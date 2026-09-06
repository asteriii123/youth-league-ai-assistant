"""
数据库连接模块。

负责：

1. 创建数据库连接
2. 创建Session
3. 提供数据库依赖


被所有接口调用：

db: Session = Depends(get_db)


当前项目：

SQLAlchemy + SQLite
"""
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import ensure_data_directories, settings


ensure_data_directories()

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
