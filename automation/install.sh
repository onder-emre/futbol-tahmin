#!/bin/bash
# launchd ajanını kurar / kaldırır / durumunu gösterir.
#   bash automation/install.sh            # kur (veya güncelle)
#   bash automation/install.sh uninstall  # kaldır
#   bash automation/install.sh status     # durum + son loglar
#   bash automation/install.sh run        # şimdi bir kez tetikle (launchd üzerinden)
set -euo pipefail

LABEL="com.futboltahmin.daily"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"

case "${1:-install}" in
  install)
    chmod +x "$PROJECT_DIR/automation/run_daily.sh"
    mkdir -p "$HOME/Library/LaunchAgents" "$PROJECT_DIR/logs"
    sed "s|__PROJECT_DIR__|$PROJECT_DIR|g" "$PROJECT_DIR/automation/$LABEL.plist" > "$TARGET"
    plutil -lint "$TARGET" >/dev/null
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true   # varsa önce kaldır (güncelleme)
    launchctl bootstrap "$DOMAIN" "$TARGET"
    launchctl enable "$DOMAIN/$LABEL"
    echo "✅ Kuruldu: her gün 09:00 → $TARGET"
    ;;
  uninstall)
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    rm -f "$TARGET"
    echo "🗑  Kaldırıldı."
    ;;
  status)
    launchctl print "$DOMAIN/$LABEL" 2>/dev/null | grep -E "state|last exit code|runs" || echo "Kurulu değil."
    echo "--- logs/daily.log (son 15 satır) ---"
    tail -n 15 "$PROJECT_DIR/logs/daily.log" 2>/dev/null || echo "(henüz log yok)"
    ;;
  run)
    launchctl kickstart -p "$DOMAIN/$LABEL"
    echo "▶️  Tetiklendi. İzle: tail -f $PROJECT_DIR/logs/daily.log"
    ;;
  *)
    echo "Kullanım: $0 [install|uninstall|status|run]"; exit 1 ;;
esac
