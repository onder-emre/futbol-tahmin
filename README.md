# ⚽ Futbol Maç Sonucu Tahmin Sistemi

Süper Lig + Avrupa'nın 5 büyük ligi için API-Football → PostgreSQL → ML → Telegram.

## Klasör yapısı

```
futbol_tahmin/
├── config/settings.py      # .env'den okunan ayarlar, lig ID'leri
├── db/
│   ├── schema.sql          # DBeaver'da çalıştırılacak şema
│   ├── connection.py       # DatabaseManager (SQLAlchemy engine)
│   └── repository.py       # Upsert + sorgular (FootballRepository)
├── api/
│   ├── client.py           # ApiFootballClient (rate limit, retry, kota koruma)
│   └── transformers.py     # JSON → pandas flatten (FixtureTransformer)
├── pipeline/ingestion.py   # Artımlı veri çekme orkestratörü
├── scripts/run_pipeline.py # CLI giriş noktası
├── models/                 # öznitelikler, eğitim, tahmin
├── bot/                    # Telegram mesaj biçimi + gönderim
├── automation/             # launchd plist, run_daily.sh, install.sh
├── main.py                 # tek tıkla günlük iş
├── notebooks/              # Jupyter analizleri
└── tests/
```

## Kurulum

```bash
cd ~/futbol_tahmin
source .venv/bin/activate          # sanal ortam hazır, paketler kurulu
# .env dosyasını aç: DB_PASSWORD ve API_FOOTBALL_KEY'i doldur
```

### Veritabanı (DBeaver)
1. DBeaver → `postgres` bağlantısı → sağ tık **Create → Database** → ad: `futbol_tahmin`
2. `futbol_tahmin` seçiliyken **SQL Editor → Open SQL script** → `db/schema.sql`
3. **Execute SQL Script (Alt+X)** — tekrar çalıştırmak güvenlidir.

(Alternatif: `python -m scripts.run_pipeline --init-db`)

## Kullanım

```bash
python -m scripts.run_pipeline --leagues 203 --seasons 2024 --max-requests 5   # küçük deneme
python -m scripts.run_pipeline                                                  # hepsi, son 10 sezon
python -m scripts.run_pipeline --skip-stats                                     # sadece maç listeleri
pytest -q                                                                       # testler
```

## Günlük akış

```bash
python -m scripts.run_pipeline      # 1) yeni maçlar + istatistikler (artımlı)
python -m scripts.build_features    # 2) form/H2H öznitelikleri → mac_ozellikleri tablosu
python -m scripts.train_model       # 3) (haftalık yeterli) model eğit + zamana dayalı test raporu
python -m scripts.predict --days 7  # 4) yaklaşan maçlar: "Galatasaray Kazanır %65, Beraberlik %20, ..."
```

### Model (FAZ 3)
- Sınıflar: `1` ev sahibi, `X` beraberlik, `2` deplasman. Çıktı her zaman `predict_proba()` olasılıkları.
- XGBoost ve Random Forest, eğitim dönemi içinde `TimeSeriesSplit` ile karşılaştırılır; log loss'u düşük olan seçilir.
- Test = en yeni %15 maç (rastgele bölme yok, aynı gün iki tarafa bölünmez). Rapor: log loss, RPS, doğruluk ve baz çizgi.
- Canlı model test sonrası tüm geçmişle yeniden eğitilir; `models/artifacts/latest.joblib`.
- Tahminler `tahminler` tablosuna, gerçek sonuçla karşılaştırma `v_tahmin_performans` görünümüne düşer.

## Kota stratejisi (neden 10 yılı her seferinde çekmiyor?)

| Adım | İstek maliyeti |
|---|---|
| Tamamlanmış lig-sezon (`veri_cekim_durumu.tamamlandi = true`) | **0** |
| Devam eden sezonun maç listesi | 1 / lig |
| İstatistikler: sadece `istatistik_cekim_zamani IS NULL` olan bitmiş maçlar, `/fixtures?ids=` ile 20'şerli | **1 / 20 maç** |

İlk tam yükleme ≈ 60 (fikstür) + ~1.050 (istatistik) istek. Free planda (100/gün) birkaç güne yayılır;
pipeline kaldığı yerden devam eder. Sonraki günlük çalıştırmalar ≈ 6–10 istek.

> ⚠️ Free plan genellikle sadece belirli sezonlara erişim verir; erişilemeyen sezonlar boş döner ve loglanır.
> xG (`expected_goals`) API'de yalnızca yakın sezonlarda bulunur; eski maçlarda `NULL` kalır.

## Tablolar
`ligler`, `takimlar`, `maclar` (`sonuc` = H/D/A, otomatik hesaplanır), `istatistikler` (maç×takım),
`veri_cekim_durumu`, ve analiz görünümü `v_mac_detay`.

---

## FAZ 4 — Telegram botu

### 1) Botu oluştur (2 dakika)
1. Telegram'da **@BotFather** → `/newbot` → isim ver → verdiği **token**'ı `.env` → `TELEGRAM_BOT_TOKEN=` satırına yapıştır.
2. Kendi botunu Telegram'da aç ve **/start** yaz (grup kullanacaksan botu gruba ekleyip bir mesaj at).
3. `python main.py --find-chat-id` → çıkan `TELEGRAM_CHAT_ID=...` satırını `.env`'e yaz.
4. `python main.py --test-telegram` → "✅ bağlantı testi başarılı" mesajı gelmeli.

### 2) Mesaj formatı
```
⚽ Günün Güvenilir Tahminleri
📅 18.11.2026 Çarşamba
━━━━━━━━━━━━━━━━━━━━━━
🇹🇷 Süper Lig

🕘 20:00 │ Galatasaray - Fenerbahçe
🟢 %65 | ⚪ %20 | 🔴 %15
🎯 Galatasaray kazanır (MS 1) ⭐⭐
━━━━━━━━━━━━━━━━━━━━━━
🟢 Ev sahibi  ⚪ Beraberlik  🔴 Deplasman
📊 9 maçtan 3 güvenilir tahmin (eşik %55)
```
- Sadece en yüksek olasılığı **≥ %55** olan tahminler listelenir (`.env` → `CONFIDENCE_THRESHOLD`).
- ⭐ %55+, ⭐⭐ %62+, ⭐⭐⭐ %70+. Uzun listeler 4096 karakter sınırına göre otomatik bölünür.
- Aynı gün iş iki kez çalışsa bile aynı maçlar için ikinci mesaj **atılmaz** (`tahminler.bildirim_gonderildi`).
- Hata olursa Telegram'a ⚠️ uyarı gelir.

## FAZ 5 — Otomasyon

### main.py — tek tıkla her şey
```bash
python main.py                 # veri → günün fikstürü → öznitelik → model (7 günden eskiyse yeniden eğit) → tahmin → Telegram
python main.py --dry-run       # Telegram'a göndermeden mesajı ekrana bas
python main.py --skip-ingest   # API'ye gitmeden mevcut veriyle
python main.py --retrain       # modeli zorla yeniden eğit
```
Bir adım çökerse (ör. API kotası bitti) kalan adımlar mevcut veriyle devam eder. Log: `logs/daily.log` (5 MB'ta döner).

### Her sabah 09:00 — launchd (önerilen)
```bash
bash automation/install.sh            # kur
bash automation/install.sh status     # durum + son loglar
bash automation/install.sh run        # şimdi bir kez tetikle (test)
bash automation/install.sh uninstall  # kaldır
```
Ne yapar: `automation/com.futboltahmin.daily.plist` dosyasını `~/Library/LaunchAgents/` altına kopyalar ve
`launchctl bootstrap` ile yükler. Ajan her gün 09:00'da `automation/run_daily.sh` çalıştırır; o da:
1. Aynı anda iki kopya çalışmasın diye kilit alır,
2. İnternet ve PostgreSQL hazır olana kadar bekler (uykudan uyanış sonrası),
3. `caffeinate -i` ile iş bitene kadar Mac'in uykuya geçmesini engeller,
4. `.venv` Python'u ile `main.py`'yi çalıştırır.

**Neden cron değil launchd?** Mac 09:00'da uykudaysa **cron o günü atlar**; launchd ise Mac uyanınca kaçırılan işi bir kez
çalıştırır. Mac tamamen **kapalıysa** ikisi de çalışmaz.

> İsteğe bağlı: Mac'i 08:58'de otomatik uyandırmak için (şifre ister):
> `sudo pmset repeat wakeorpoweron MTWRFSU 08:58:00`  — iptal: `sudo pmset repeat cancel`

### Alternatif: crontab
```bash
crontab -e
```
şu satırı ekle (dakika saat gün ay haftanın-günü):
```
0 9 * * * /bin/bash /Users/emreonder/futbol_tahmin/automation/run_daily.sh >> /Users/emreonder/futbol_tahmin/logs/cron.log 2>&1
```
`crontab -l` ile kontrol et. Not: cron, Mac uykudaysa çalışmaz; ayrıca macOS bazı klasörler için cron'a
"Tam Disk Erişimi" isteyebilir (Sistem Ayarları → Gizlilik ve Güvenlik → Tam Disk Erişimi → `/usr/sbin/cron`).
Proje ana klasörde (Desktop/Documents dışında) olduğu için genelde gerekmez.
