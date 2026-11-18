-- =====================================================================
--  Futbol Tahmin Sistemi - PostgreSQL Şeması (FAZ 1)
--  DBeaver'da: veritabanını seç → SQL Editor → bu dosyanın tamamını
--  "Execute SQL Script" (Alt+X) ile çalıştır. Tekrar çalıştırmak güvenlidir
--  (IF NOT EXISTS / OR REPLACE kullanılır).
--
--  ID'ler API-Football'un kendi ID'leridir; bu sayede upsert idempotenttir.
-- =====================================================================

-- ---------------------------------------------------------------------
-- Ortak: guncellenme_zamani kolonunu otomatik güncelleyen trigger fonksiyonu
-- ---------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_guncellenme_zamani()
RETURNS TRIGGER AS $$
BEGIN
    NEW.guncellenme_zamani := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ---------------------------------------------------------------------
-- 1) LİGLER
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ligler (
    lig_id              INTEGER      PRIMARY KEY,            -- API-Football league.id
    ad                  VARCHAR(100) NOT NULL,
    ulke                VARCHAR(100) NOT NULL,
    logo_url            TEXT,
    olusturma_zamani    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    guncellenme_zamani  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_ligler_ad_ulke UNIQUE (ad, ulke)
);

-- ---------------------------------------------------------------------
-- 2) TAKIMLAR
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS takimlar (
    takim_id            INTEGER      PRIMARY KEY,            -- API-Football team.id
    ad                  VARCHAR(100) NOT NULL,
    ulke                VARCHAR(100) NOT NULL,
    logo_url            TEXT,
    olusturma_zamani    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    guncellenme_zamani  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_takimlar_ad_ulke UNIQUE (ad, ulke)
);

-- ---------------------------------------------------------------------
-- 3) MAÇLAR
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS maclar (
    mac_id                   BIGINT       PRIMARY KEY,       -- API-Football fixture.id
    lig_id                   INTEGER      NOT NULL REFERENCES ligler(lig_id),
    sezon                    SMALLINT     NOT NULL,          -- 2024 => 2024/25 sezonu
    hafta                    VARCHAR(60),                    -- "Regular Season - 12"
    tarih                    TIMESTAMPTZ  NOT NULL,          -- UTC
    hakem                    VARCHAR(120),
    stadyum                  VARCHAR(150),
    durum                    VARCHAR(5)   NOT NULL,          -- NS, FT, AET, PEN, PST, CANC...
    ev_sahibi_id             INTEGER      NOT NULL REFERENCES takimlar(takim_id),
    deplasman_id             INTEGER      NOT NULL REFERENCES takimlar(takim_id),
    ev_gol                   SMALLINT     CHECK (ev_gol  >= 0),
    dep_gol                  SMALLINT     CHECK (dep_gol >= 0),
    ilk_yari_ev_gol          SMALLINT     CHECK (ilk_yari_ev_gol  >= 0),
    ilk_yari_dep_gol         SMALLINT     CHECK (ilk_yari_dep_gol >= 0),
    -- ML hedef değişkeni: H = ev sahibi kazandı, D = beraberlik, A = deplasman kazandı
    sonuc                    CHAR(1) GENERATED ALWAYS AS (
                                 CASE
                                     WHEN ev_gol > dep_gol THEN 'H'
                                     WHEN ev_gol < dep_gol THEN 'A'
                                     WHEN ev_gol = dep_gol THEN 'D'
                                 END
                             ) STORED,
    -- İstatistikleri API'den çektiğimiz an. NULL => henüz çekilmedi (kota için kritik)
    istatistik_cekim_zamani  TIMESTAMPTZ,
    olusturma_zamani         TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    guncellenme_zamani       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_maclar_farkli_takim CHECK (ev_sahibi_id <> deplasman_id),
    -- Doğal anahtar: aynı iki takım aynı anda iki kez oynayamaz
    CONSTRAINT uq_maclar_takimlar_tarih UNIQUE (ev_sahibi_id, deplasman_id, tarih)
);

CREATE INDEX IF NOT EXISTS ix_maclar_lig_sezon   ON maclar (lig_id, sezon);
CREATE INDEX IF NOT EXISTS ix_maclar_tarih       ON maclar (tarih);
CREATE INDEX IF NOT EXISTS ix_maclar_ev_sahibi   ON maclar (ev_sahibi_id, tarih);
CREATE INDEX IF NOT EXISTS ix_maclar_deplasman   ON maclar (deplasman_id, tarih);
-- İstatistiği eksik, bitmiş maçları hızlı bulmak için kısmi index
CREATE INDEX IF NOT EXISTS ix_maclar_istatistik_bekleyen
    ON maclar (tarih DESC)
    WHERE istatistik_cekim_zamani IS NULL AND durum IN ('FT', 'AET', 'PEN');

-- ---------------------------------------------------------------------
-- 4) İSTATİSTİKLER (maç başına takım başına 1 satır)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS istatistikler (
    mac_id               BIGINT       NOT NULL REFERENCES maclar(mac_id) ON DELETE CASCADE,
    takim_id             INTEGER      NOT NULL REFERENCES takimlar(takim_id),
    xg                   NUMERIC(5,2) CHECK (xg >= 0),
    topla_oynama         NUMERIC(5,2) CHECK (topla_oynama BETWEEN 0 AND 100),  -- yüzde
    toplam_sut           SMALLINT,
    isabetli_sut         SMALLINT,
    isabetsiz_sut        SMALLINT,
    engellenen_sut       SMALLINT,
    ceza_sahasi_ici_sut  SMALLINT,
    ceza_sahasi_disi_sut SMALLINT,
    korner               SMALLINT,
    faul                 SMALLINT,
    ofsayt               SMALLINT,
    sari_kart            SMALLINT,
    kirmizi_kart         SMALLINT,
    kaleci_kurtarisi     SMALLINT,
    toplam_pas           SMALLINT,
    isabetli_pas         SMALLINT,
    pas_isabet_yuzdesi   NUMERIC(5,2) CHECK (pas_isabet_yuzdesi BETWEEN 0 AND 100),
    olusturma_zamani     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    guncellenme_zamani   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    CONSTRAINT pk_istatistikler PRIMARY KEY (mac_id, takim_id)
);

CREATE INDEX IF NOT EXISTS ix_istatistikler_takim ON istatistikler (takim_id);

-- ---------------------------------------------------------------------
-- 5) VERİ ÇEKİM DURUMU (lig+sezon bazında senkron takibi → kota koruma)
--    Tamamlanmış sezonlar bir kez çekildikten sonra bir daha istenmez.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS veri_cekim_durumu (
    lig_id              INTEGER     NOT NULL REFERENCES ligler(lig_id),
    sezon               SMALLINT    NOT NULL,
    son_cekim_zamani    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    mac_sayisi          INTEGER     NOT NULL DEFAULT 0,
    tamamlandi          BOOLEAN     NOT NULL DEFAULT FALSE,
    CONSTRAINT pk_veri_cekim_durumu PRIMARY KEY (lig_id, sezon)
);

-- ---------------------------------------------------------------------
-- Trigger'lar (PG14+ için CREATE OR REPLACE TRIGGER)
-- ---------------------------------------------------------------------
CREATE OR REPLACE TRIGGER trg_ligler_guncellenme
    BEFORE UPDATE ON ligler FOR EACH ROW EXECUTE FUNCTION set_guncellenme_zamani();
CREATE OR REPLACE TRIGGER trg_takimlar_guncellenme
    BEFORE UPDATE ON takimlar FOR EACH ROW EXECUTE FUNCTION set_guncellenme_zamani();
CREATE OR REPLACE TRIGGER trg_maclar_guncellenme
    BEFORE UPDATE ON maclar FOR EACH ROW EXECUTE FUNCTION set_guncellenme_zamani();
CREATE OR REPLACE TRIGGER trg_istatistikler_guncellenme
    BEFORE UPDATE ON istatistikler FOR EACH ROW EXECUTE FUNCTION set_guncellenme_zamani();

-- ---------------------------------------------------------------------
-- Analiz için hazır görünüm: maç + iki takımın istatistikleri tek satırda
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW v_mac_detay AS
SELECT
    m.mac_id, m.lig_id, l.ad AS lig, m.sezon, m.hafta, m.tarih, m.durum,
    m.ev_sahibi_id, th.ad AS ev_sahibi, m.deplasman_id, ta.ad AS deplasman,
    m.ev_gol, m.dep_gol, m.sonuc,
    sh.xg AS ev_xg,               sa.xg AS dep_xg,
    sh.topla_oynama AS ev_topla_oynama, sa.topla_oynama AS dep_topla_oynama,
    sh.toplam_sut AS ev_sut,      sa.toplam_sut AS dep_sut,
    sh.isabetli_sut AS ev_isabetli_sut, sa.isabetli_sut AS dep_isabetli_sut,
    sh.korner AS ev_korner,       sa.korner AS dep_korner,
    sh.sari_kart AS ev_sari,      sa.sari_kart AS dep_sari,
    sh.kirmizi_kart AS ev_kirmizi, sa.kirmizi_kart AS dep_kirmizi
FROM maclar m
JOIN ligler   l  ON l.lig_id   = m.lig_id
JOIN takimlar th ON th.takim_id = m.ev_sahibi_id
JOIN takimlar ta ON ta.takim_id = m.deplasman_id
LEFT JOIN istatistikler sh ON sh.mac_id = m.mac_id AND sh.takim_id = m.ev_sahibi_id
LEFT JOIN istatistikler sa ON sa.mac_id = m.mac_id AND sa.takim_id = m.deplasman_id;

-- ---------------------------------------------------------------------
-- 6) TAHMİNLER (FAZ 3) — her maç için son model tahmini; FAZ 5 botu buradan okur
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tahminler (
    mac_id               BIGINT       PRIMARY KEY REFERENCES maclar(mac_id) ON DELETE CASCADE,
    model_versiyonu      VARCHAR(60)  NOT NULL,
    olasilik_1           NUMERIC(5,4) NOT NULL CHECK (olasilik_1 BETWEEN 0 AND 1),   -- ev sahibi
    olasilik_x           NUMERIC(5,4) NOT NULL CHECK (olasilik_x BETWEEN 0 AND 1),   -- beraberlik
    olasilik_2           NUMERIC(5,4) NOT NULL CHECK (olasilik_2 BETWEEN 0 AND 1),   -- deplasman
    tahmin               CHAR(1)      NOT NULL CHECK (tahmin IN ('1', 'X', '2')),
    guven                NUMERIC(5,4) NOT NULL,
    bildirim_gonderildi  BOOLEAN      NOT NULL DEFAULT FALSE,
    olusturma_zamani     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    guncellenme_zamani   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE OR REPLACE TRIGGER trg_tahminler_guncellenme
    BEFORE UPDATE ON tahminler FOR EACH ROW EXECUTE FUNCTION set_guncellenme_zamani();

-- Tahmin + gerçek sonuç yan yana (model başarısını DBeaver'dan izlemek için)
CREATE OR REPLACE VIEW v_tahmin_performans AS
SELECT
    t.mac_id, m.tarih, l.ad AS lig, th.ad AS ev_sahibi, ta.ad AS deplasman,
    ROUND(t.olasilik_1 * 100) AS yuzde_1, ROUND(t.olasilik_x * 100) AS yuzde_x, ROUND(t.olasilik_2 * 100) AS yuzde_2,
    t.tahmin,
    CASE m.sonuc WHEN 'H' THEN '1' WHEN 'D' THEN 'X' WHEN 'A' THEN '2' END AS gercek,
    CASE WHEN m.sonuc IS NULL THEN NULL
         ELSE t.tahmin = CASE m.sonuc WHEN 'H' THEN '1' WHEN 'D' THEN 'X' WHEN 'A' THEN '2' END END AS dogru_mu,
    t.model_versiyonu
FROM tahminler t
JOIN maclar   m  ON m.mac_id = t.mac_id
JOIN ligler   l  ON l.lig_id = m.lig_id
JOIN takimlar th ON th.takim_id = m.ev_sahibi_id
JOIN takimlar ta ON ta.takim_id = m.deplasman_id;
