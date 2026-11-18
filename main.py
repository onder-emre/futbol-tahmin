"""Futbol Tahmin Sistemi — tek tıkla günlük çalıştırma.

    python main.py                  # tam akış: veri → fikstür → öznitelik → model → tahmin → Telegram
    python main.py --dry-run        # Telegram'a göndermez, mesajı ekrana basar
    python main.py --skip-ingest    # API'ye hiç gitme, sadece mevcut veriyle tahmin
    python main.py --retrain        # modeli yaşından bağımsız yeniden eğit
    python main.py --test-telegram  # bot bağlantısını test et
    python main.py --find-chat-id   # bota mesaj atan sohbetlerin chat_id'lerini listele
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

# Proje kökünü import yoluna ekle (launchd/cron farklı dizinden çağırsa da çalışsın)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.client import ApiFootballClient  # noqa: E402
from bot.notifier import TelegramNotifier, discover_chat_ids  # noqa: E402
from config.settings import PROJECT_ROOT, DailyJobSettings, Settings, TelegramSettings  # noqa: E402
from db.connection import DatabaseManager  # noqa: E402
from db.repository import FootballRepository  # noqa: E402
from pipeline.daily import DailyJob  # noqa: E402


def configure_logging() -> None:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    file_handler = RotatingFileHandler(log_dir / "daily.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    console = logging.StreamHandler(sys.stdout)
    for h in (file_handler, console):
        h.setFormatter(fmt)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console])
    for noisy in ("httpx", "httpcore", "telegram"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Futbol tahmin günlük işi")
    parser.add_argument("--dry-run", action="store_true", help="Telegram'a gönderme, ekrana yaz")
    parser.add_argument("--skip-ingest", action="store_true", help="API'den veri çekme")
    parser.add_argument("--retrain", action="store_true", help="Modeli zorla yeniden eğit")
    parser.add_argument("--test-telegram", action="store_true")
    parser.add_argument("--find-chat-id", action="store_true")
    args = parser.parse_args(argv)

    configure_logging()
    log = logging.getLogger("main")

    if args.find_chat_id:
        token = os.getenv("TELEGRAM_BOT_TOKEN") or ""
        chats = discover_chat_ids(token)
        if not chats:
            print("Hiç sohbet bulunamadı. Telegram'da botuna bir mesaj (/start) at ve tekrar dene.")
        for chat_id, name in chats:
            print(f"TELEGRAM_CHAT_ID={chat_id}    # {name}")
        return 0

    if args.test_telegram:
        TelegramNotifier(TelegramSettings.from_env()).send_plain("✅ Futbol Tahmin botu bağlantı testi başarılı!")
        print("Test mesajı gönderildi.")
        return 0

    settings = Settings.from_env()
    notifier = None if args.dry_run else TelegramNotifier(TelegramSettings.from_env())
    db = DatabaseManager(settings.db)
    job = DailyJob(
        settings=settings,
        daily=DailyJobSettings.from_env(),
        repository=FootballRepository(db.engine),
        client=ApiFootballClient(settings.api),
        notifier=notifier,
        dry_run=args.dry_run,
    )

    log.info("===== Günlük iş başladı =====")
    report = job.run(skip_ingest=args.skip_ingest, force_retrain=args.retrain)
    log.info("===== Günlük iş bitti =====\n%s", report)

    if not report.ok and notifier is not None:
        failed = "\n".join(f"• {k}: {v}" for k, v in report.steps.items() if v.startswith("HATA"))
        try:
            notifier.send_plain(f"⚠️ Futbol Tahmin günlük işi hata verdi:\n{failed}\n\nDetay: logs/daily.log")
        except Exception:
            log.exception("Hata bildirimi de gönderilemedi")
    db.dispose()
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
