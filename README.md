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
├── models/                 # FAZ 2-4: özellik mühendisliği + ML
├── bot/                    # FAZ 5: Telegram botu
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
