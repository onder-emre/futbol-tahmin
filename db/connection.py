"""SQLAlchemy engine yönetimi."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine, text

from config.settings import DatabaseSettings

SCHEMA_FILE = Path(__file__).with_name("schema.sql")


class DatabaseManager:
    def __init__(self, settings: DatabaseSettings, echo: bool = False) -> None:
        self.engine: Engine = create_engine(settings.url, echo=echo, pool_pre_ping=True, future=True)

    def ping(self) -> str:
        with self.engine.connect() as conn:
            return conn.execute(text("SELECT version()")).scalar_one()

    def apply_schema(self, schema_file: Path = SCHEMA_FILE) -> None:
        """schema.sql'i çalıştırır (DBeaver'a alternatif). İdempotenttir."""
        sql = schema_file.read_text(encoding="utf-8")
        raw = self.engine.raw_connection()
        try:
            with raw.cursor() as cur:
                cur.execute(sql)  # psycopg2 çoklu ifadeyi ve $$ bloklarını destekler
            raw.commit()
        finally:
            raw.close()

    def dispose(self) -> None:
        self.engine.dispose()
