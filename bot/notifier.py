"""Telegram'a mesaj gönderimi (python-telegram-bot v21+, async)."""
from __future__ import annotations

import asyncio
import logging

from telegram import Bot
from telegram.constants import ParseMode
from telegram.helpers import escape_markdown

from config.settings import TelegramSettings

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, settings: TelegramSettings) -> None:
        self._settings = settings

    def send(self, messages: list[str]) -> None:
        """MarkdownV2 mesajlarını sırayla gönderir (senkron arayüz)."""
        asyncio.run(self._send_all(messages))

    def send_plain(self, text: str) -> None:
        self.send([escape_markdown(text, version=2)])

    async def _send_all(self, messages: list[str]) -> None:
        chat_ids = self._settings.chat_ids
        async with Bot(self._settings.token) as bot:
            for chat_id in chat_ids:
                for text in messages:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=text,
                        parse_mode=ParseMode.MARKDOWN_V2,
                        disable_web_page_preview=True,
                    )
        logger.info("Telegram: %d mesaj × %d sohbet gönderildi.", len(messages), len(chat_ids))


def discover_chat_ids(token: str) -> list[tuple[int, str]]:
    """Bota mesaj atmış sohbetlerin (chat_id, ad) listesi — .env'deki TELEGRAM_CHAT_ID'yi bulmak için."""
    async def _run() -> list[tuple[int, str]]:
        async with Bot(token) as bot:
            updates = await bot.get_updates()
        seen: dict[int, str] = {}
        for u in updates:
            chat = u.effective_chat
            if chat:
                seen[chat.id] = chat.title or chat.full_name or chat.username or ""
        return list(seen.items())

    return asyncio.run(_run())
