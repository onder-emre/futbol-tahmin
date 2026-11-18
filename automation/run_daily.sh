#!/bin/bash
# =============================================================================
#  Futbol Tahmin — günlük çalıştırma sarmalayıcısı (launchd / cron bunu çağırır)
#
#  Neden doğrudan "python main.py" değil?
#   • launchd/cron minimal bir ortamla çalışır: PATH, venv, çalışma dizini yoktur.
#   • Mac uykudan yeni uyandıysa ağ ve PostgreSQL birkaç saniye hazır olmayabilir.
#   • Aynı anda iki kopya çalışmamalı (kilit).
#   • İş sürerken Mac uykuya dalmamalı (caffeinate).
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"
LOG_DIR="$PROJECT_DIR/logs"
LOCK_DIR="/tmp/futbol_tahmin.lock"
PG_ISREADY="/Library/PostgreSQL/18/bin/pg_isready"

mkdir -p "$LOG_DIR"
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') | run_daily | $*"; }

# --- 1) Tek kopya kilidi (macOS'te flock yok → atomik mkdir) -----------------
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    # 3 saatten eski kilit = önceki çalışma çökmüş, temizle
    if [ -n "$(find "$LOCK_DIR" -maxdepth 0 -mmin +180 2>/dev/null)" ]; then
        log "Eski kilit temizlendi."; rm -rf "$LOCK_DIR"; mkdir "$LOCK_DIR"
    else
        log "Başka bir çalışma sürüyor, çıkılıyor."; exit 0
    fi
fi
trap 'rm -rf "$LOCK_DIR"' EXIT

# --- 2) İnternet bağlantısını bekle (uyanış sonrası en fazla 2 dk) -----------
for i in $(seq 1 24); do
    if /usr/bin/curl -s --max-time 5 -o /dev/null https://api.telegram.org; then break; fi
    [ "$i" -eq 24 ] && { log "İnternet yok, iş iptal."; exit 1; }
    sleep 5
done

# --- 3) PostgreSQL'i bekle (en fazla 1 dk) ------------------------------------
if [ -x "$PG_ISREADY" ]; then
    for i in $(seq 1 12); do
        if "$PG_ISREADY" -h localhost -p 5432 -q; then break; fi
        [ "$i" -eq 12 ] && { log "PostgreSQL hazır değil, iş iptal."; exit 1; }
        sleep 5
    done
fi

# --- 4) Çalıştır (caffeinate -i: iş bitene kadar sistem uykusunu engelle) ----
cd "$PROJECT_DIR"
log "Başlıyor: $PYTHON main.py $*"
status=0
/usr/bin/caffeinate -i "$PYTHON" main.py "$@" || status=$?
log "Bitti (çıkış kodu: $status)"
exit $status
