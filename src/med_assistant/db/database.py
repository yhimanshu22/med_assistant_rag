from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from med_assistant.core.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from med_assistant.models.user import User  # noqa: F401

    Base.metadata.create_all(bind=engine)
