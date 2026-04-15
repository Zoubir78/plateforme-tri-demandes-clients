from __future__ import annotations

from datetime import datetime, date
from uuid import UUID

from sqlalchemy import BigInteger, Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from src.app.db.base import Base


class RawSignalConsoComplaint(Base):
    __tablename__ = "raw_signalconso_complaints"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    external_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False, unique=True, index=True)

    category: Mapped[dict | list | None] = mapped_column(JSONB, nullable=True)
    subcategories: Mapped[dict | list | None] = mapped_column(JSONB, nullable=True)
    creationdate: Mapped[date | None] = mapped_column(Date, nullable=True)

    contactagreement: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    status: Mapped[str | None] = mapped_column(Text, nullable=True)
    forwardtoreponseconso: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    tags: Mapped[dict | list | None] = mapped_column(JSONB, nullable=True)
    signalement_transmis: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    signalement_lu: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    signalement_reponse: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    dep_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    dep_code: Mapped[str | None] = mapped_column(String(10), nullable=True)
    reg_code: Mapped[str | None] = mapped_column(String(10), nullable=True)
    reg_name: Mapped[str | None] = mapped_column(Text, nullable=True)

    raw_payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())

    clean_row: Mapped["CleanSignalConsoComplaint"] = relationship(back_populates="raw_row", uselist=False)


class CleanSignalConsoComplaint(Base):
    __tablename__ = "clean_signalconso_complaints"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    raw_signalconso_id: Mapped[int] = mapped_column(
        ForeignKey("raw_signalconso_complaints.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    clean_text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    language: Mapped[str | None] = mapped_column(String(20), nullable=True)
    is_valid: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    cleaned_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())

    raw_row: Mapped[RawSignalConsoComplaint] = relationship(back_populates="clean_row")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="clean_row")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    clean_signalconso_id: Mapped[int] = mapped_column(
        ForeignKey("clean_signalconso_complaints.id", ondelete="CASCADE"),
        nullable=False,
    )

    predicted_category: Mapped[str] = mapped_column(String(255), nullable=False)
    predicted_priority: Mapped[str] = mapped_column(String(50), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())

    clean_row: Mapped[CleanSignalConsoComplaint] = relationship(back_populates="predictions")
    
class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    clean_text: Mapped[str] = mapped_column(Text, nullable=False)
    predicted_category: Mapped[str] = mapped_column(String(255), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())

class ModelRun(Base):
    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    dataset_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    metrics: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    artifact_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())