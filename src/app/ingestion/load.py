from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from src.app.db.session import SessionLocal
from src.app.db.models import CleanComplaint, DataSource, RawComplaint


def _to_serializable(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _row_to_dict(row: pd.Series) -> dict[str, Any]:
    return {k: _to_serializable(v) for k, v in row.to_dict().items()}


def _build_match_key(row: pd.Series | dict[str, Any]) -> str:
    """
    Construit une clé de rapprochement entre raw_df et clean_df.
    Priorité aux identifiants stables, sinon aux champs métier.
    """
    getter = row.get if isinstance(row, dict) else row.get

    candidates = [
        getter("external_id"),
        getter("source_id"),
        getter("id"),
        getter("record_id"),
        getter("recordid"),
        getter("uuid"),
    ]
    for value in candidates:
        if value is not None and str(value).strip():
            return f"id:{str(value).strip()}"

    subject = getter("subject")
    description = getter("description")
    complaint_text = getter("complaint_text")
    submitted_at = getter("submitted_at")

    parts = [
        str(subject).strip().lower() if subject is not None else "",
        str(description).strip().lower() if description is not None else "",
        str(complaint_text).strip().lower() if complaint_text is not None else "",
        str(submitted_at).strip().lower() if submitted_at is not None else "",
    ]
    key = "|".join(parts).strip("|")
    return f"content:{key}"


def get_or_create_source(
    db: Session,
    name: str,
    source_type: str,
    base_url: str | None = None,
) -> DataSource:
    source = (
        db.query(DataSource)
        .filter(
            DataSource.name == name,
            DataSource.source_type == source_type,
            DataSource.base_url == base_url,
        )
        .one_or_none()
    )
    if source is not None:
        return source

    source = DataSource(name=name, source_type=source_type, base_url=base_url)
    db.add(source)
    db.flush()  # récupère l'ID sans commit
    return source


def save_raw_ticket(
    db: Session,
    source_db_id: int,
    payload: dict[str, Any],
) -> RawComplaint:
    ticket = RawComplaint(
        source_id=source_db_id,
        external_id=payload.get("external_id"),
        subject=payload.get("subject"),
        description=payload.get("description"),
        raw_category=payload.get("category"),
        raw_priority=payload.get("priority"),
        raw_payload=payload,
    )
    db.add(ticket)
    db.flush()
    return ticket


def save_clean_ticket(
    db: Session,
    raw_ticket_id: int,
    clean_text: str,
    language: str | None,
    token_count: int,
) -> CleanComplaint:
    clean_complaints = CleanComplaint(
        raw_ticket_id=raw_ticket_id,
        clean_text=clean_text,
        language=language,
        token_count=token_count,
        is_valid=True,
    )
    db.add(clean_complaints)
    db.flush()
    return clean_complaints


def load_into_db(
    db: Session,
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    source_name: str = "SignalConso",
    source_type: str = "api",
    base_url: str | None = None,
) -> tuple[int, int]:
    """
    Charge raw_df et clean_df dans la base.

    Retourne:
        (nombre_de_raw_insérés, nombre_de_clean_insérés)
    """
    source = get_or_create_source(db, name=source_name, source_type=source_type, base_url=base_url)

    raw_id_by_key: dict[str, int] = {}
    raw_inserted = 0
    clean_inserted = 0

    with db.begin():
        for _, row in raw_df.iterrows():
            payload = _row_to_dict(row)
            key = _build_match_key(payload)

            # On garde une copie stable de la clé dans external_id si possible
            if payload.get("external_id") is None:
                payload["external_id"] = key

            raw_ticket = save_raw_ticket(db, source_db_id=source.id, payload=payload)
            raw_id_by_key[key] = raw_ticket.id
            raw_inserted += 1

        for _, row in clean_df.iterrows():
            row_dict = _row_to_dict(row)
            key = _build_match_key(row_dict)
            raw_ticket_id = raw_id_by_key.get(key)

            if raw_ticket_id is None:
                continue

            clean_text = str(row_dict.get("clean_text") or "").strip()
            if not clean_text:
                continue

            token_count = row_dict.get("token_count")
            if token_count is None or pd.isna(token_count):
                token_count = len(clean_text.split())
            else:
                token_count = int(token_count)

            language = row_dict.get("language")
            if language is not None and pd.isna(language):
                language = None

            save_clean_ticket(
                db=db,
                raw_ticket_id=raw_ticket_id,
                clean_text=clean_text,
                language=language,
                token_count=token_count,
            )
            clean_inserted += 1

    return raw_inserted, clean_inserted


def main() -> None:
    # Adaptez ces imports si vos chemins changent
    from src.app.ingestion.extract import extract_from_signalconso_api
    from src.app.ingestion.transform import transform_dataframe

    API_URL = "https://data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/signalconso/records"

    raw_df = extract_from_signalconso_api(API_URL, limit=1000)
    clean_df = transform_dataframe(raw_df)

    db = SessionLocal()
    try:
        raw_count, clean_count = load_into_db(
            db=db,
            raw_df=raw_df,
            clean_df=clean_df,
            source_name="SignalConso",
            source_type="api",
            base_url=API_URL,
        )
        print(f"Lignes brutes insérées : {raw_count}")
        print(f"Lignes nettoyées insérées : {clean_count}")
    finally:
        db.close()


if __name__ == "__main__":
    main()