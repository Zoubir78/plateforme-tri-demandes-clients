from __future__ import annotations

import pandas as pd
import requests


def extract_from_signalconso_api(api_url: str, limit: int = 10000) -> pd.DataFrame:
    page_size = 100  # limite max de l'API
    offset = 0
    rows = []

    while len(rows) < limit:
        params = {
            "limit": page_size,
            "offset": offset,
        }

        response = requests.get(api_url, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()

        records = payload.get("results", [])
        if not records:
            break

        rows.extend(records)
        offset += page_size

    rows = rows[:limit]

    if not isinstance(rows, list):
        raise ValueError("Format inattendu retourné par l'API SignalConso")

    return pd.json_normalize(rows)