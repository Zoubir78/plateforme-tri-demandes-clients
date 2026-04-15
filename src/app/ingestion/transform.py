from __future__ import annotations

import ast
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CleaningConfig:
    min_text_length: int = 10
    include_category_in_text: bool = False  # laisser False pour éviter la fuite de cible


COLUMN_ALIASES = {
    "source_id": ["source_id", "id", "recordid", "record_id", "uuid"],
    "creationdate": ["creationdate", "created_at", "date", "created", "publication_date"],
    "category": ["category", "categorie", "catégorie", "theme", "main_theme"],
    "subcategories": ["subcategories", "sub_category", "subcategory", "sub_theme", "sous_categorie", "sous_theme"],
    "tags": ["tags", "tag"],
    "status": ["status", "statut", "state"],
    "dep_name": ["dep_name", "department_name", "departement_name", "department"],
    "dep_code": ["dep_code", "department_code"],
    "reg_name": ["reg_name", "region_name", "région"],
    "reg_code": ["reg_code", "region_code"],
    "complaint_text": ["complaint_text", "description", "narrative", "message", "content", "body", "details"],
}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float) and pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_colname(name: str) -> str:
    return normalize_text(name).replace(" ", "_")


def _find_actual_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized_to_actual = {_normalize_colname(col): col for col in df.columns}
    for candidate in candidates:
        key = _normalize_colname(candidate)
        if key in normalized_to_actual:
            return normalized_to_actual[key]
    return None


def _is_missing(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, str):
        return not value.strip()

    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0

    try:
        result = pd.isna(value)
        if isinstance(result, bool):
            return result
        return False
    except (TypeError, ValueError):
        return False


def _get_first_value(row: pd.Series, candidates: list[str]) -> Any:
    for candidate in candidates:
        if candidate in row.index:
            value = row[candidate]
            if not _is_missing(value):
                return value
    return None


def _parse_multivalue(value: Any) -> list[str]:
    """
    Gère :
    - list/tuple/set Python
    - chaînes du type "['AchatMagasin', 'Autre']"
    - chaînes simples
    """
    if _is_missing(value):
        return []

    if isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        raw = str(value).strip()
        if not raw:
            return []

        if raw.startswith("[") or raw.startswith("(") or raw.startswith("{"):
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, (list, tuple, set)):
                    items = list(parsed)
                else:
                    items = [parsed]
            except (ValueError, SyntaxError):
                items = [part.strip() for part in raw.strip("[](){}").split(",")]
        else:
            items = [raw]

    cleaned: list[str] = []
    for item in items:
        txt = normalize_text(item)
        if txt:
            cleaned.append(txt)
    return cleaned


def _extract_text_parts(row: pd.Series, candidates: list[str]) -> list[str]:
    value = _get_first_value(row, candidates)
    if value is None:
        return []
    return _parse_multivalue(value)


def build_clean_text(row: pd.Series, config: CleaningConfig | None = None) -> str:
    config = config or CleaningConfig()

    parts: list[str] = []

    # Ne pas inclure category par défaut pour éviter la fuite de cible
    if config.include_category_in_text:
        parts.extend(_extract_text_parts(row, COLUMN_ALIASES["category"]))

    parts.extend(_extract_text_parts(row, COLUMN_ALIASES["subcategories"]))
    parts.extend(_extract_text_parts(row, COLUMN_ALIASES["tags"]))

    dep_name = normalize_text(_get_first_value(row, COLUMN_ALIASES["dep_name"]))
    reg_name = normalize_text(_get_first_value(row, COLUMN_ALIASES["reg_name"]))
    status = normalize_text(_get_first_value(row, COLUMN_ALIASES["status"]))
    complaint_text = normalize_text(_get_first_value(row, COLUMN_ALIASES["complaint_text"]))

    for value in [dep_name, reg_name, status, complaint_text]:
        if value:
            parts.append(value)

    seen = set()
    unique_parts = []
    for part in parts:
        part = normalize_text(part)
        if part and part not in seen:
            seen.add(part)
            unique_parts.append(part)

    return normalize_text(" ".join(unique_parts))


def transform_dataframe(df: pd.DataFrame, config: CleaningConfig | None = None) -> pd.DataFrame:
    config = config or CleaningConfig()
    out = df.copy()

    creationdate_col = _find_actual_column(out, COLUMN_ALIASES["creationdate"])
    if creationdate_col is not None:
        out[creationdate_col] = pd.to_datetime(out[creationdate_col], errors="coerce")

    out["clean_text"] = out.apply(build_clean_text, axis=1, config=config)
    out["token_count"] = out["clean_text"].str.split().str.len().fillna(0).astype(int)
    out["is_valid"] = out["clean_text"].str.len() >= config.min_text_length

    out = out[out["is_valid"]].copy()

    source_id_col = _find_actual_column(out, COLUMN_ALIASES["source_id"])
    if source_id_col is not None:
        out = out.drop_duplicates(subset=[source_id_col], keep="first")
    else:
        out = out.drop_duplicates(keep="first")

    return out.reset_index(drop=True)