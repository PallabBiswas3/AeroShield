# backend/app/db.py
"""
Lightweight SQLite persistence for dispatched enforcement cases.

Why this exists: the frontend's "Approve & Dispatch Field Squad" button
used to just fake a 1.8s network delay in React state and forget the case
on refresh. This module gives it a real, durable log — good enough for a
demo/prototype, swap for Postgres later if this goes to production.

Table: enforcement_cases
    case_id             auto-increment primary key
    cell_id             grid cell that was clicked
    lat / lon           exact coordinates analyzed
    predicted_aqi        model's PM2.5 prediction at dispatch time
    aqi_label           e.g. "Severe"
    primary_violator    top-ranked attributed source name
    confidence_score    attribution confidence % for that source
    escalation_level    CRITICAL / HIGH / MEDIUM / LOW (from planner agent)
    dispatch_priority   CRITICAL / HIGH / MEDIUM (from legal drafter agent)
    statute_violated    e.g. "Section 21, Air Act 1981"
    legal_notice_draft  full drafted notice text
    case_summary        short field-alert summary
    attribution_matrix  JSON-encoded full ranked source list
    status              DISPATCHED (only status for now; extend later)
    created_at          ISO-8601 UTC timestamp
"""
import os
import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "aeroshield.db"))


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Call once at app startup. Safe to call repeatedly (IF NOT EXISTS)."""
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS enforcement_cases (
            case_id             INTEGER PRIMARY KEY AUTOINCREMENT,
            cell_id             INTEGER NOT NULL,
            lat                 REAL,
            lon                 REAL,
            predicted_aqi       REAL,
            aqi_label           TEXT,
            primary_violator    TEXT,
            confidence_score    REAL,
            escalation_level    TEXT,
            dispatch_priority   TEXT,
            statute_violated    TEXT,
            legal_notice_draft  TEXT,
            case_summary        TEXT,
            attribution_matrix  TEXT,
            status              TEXT DEFAULT 'DISPATCHED',
            created_at          TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def insert_case(payload: dict) -> int:
    conn = _connect()
    cur = conn.execute("""
        INSERT INTO enforcement_cases
        (cell_id, lat, lon, predicted_aqi, aqi_label, primary_violator, confidence_score,
         escalation_level, dispatch_priority, statute_violated, legal_notice_draft,
         case_summary, attribution_matrix, status, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        payload["cell_id"], payload.get("lat"), payload.get("lon"),
        payload.get("predicted_aqi"), payload.get("aqi_label"),
        payload.get("primary_violator"), payload.get("confidence_score"),
        payload.get("escalation_level"), payload.get("dispatch_priority"),
        payload.get("statute_violated"), payload.get("legal_notice_draft"),
        payload.get("case_summary"), json.dumps(payload.get("attribution_matrix", [])),
        "DISPATCHED", datetime.now(timezone.utc).isoformat(),
    ))
    conn.commit()
    case_id = cur.lastrowid
    conn.close()
    return case_id


def list_cases(limit: int = 50) -> list:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM enforcement_cases ORDER BY case_id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["attribution_matrix"] = json.loads(d["attribution_matrix"])
        except Exception:
            d["attribution_matrix"] = []
        out.append(d)
    return out


def get_case(case_id: int) -> Optional[dict]:
    conn = _connect()
    row = conn.execute("SELECT * FROM enforcement_cases WHERE case_id = ?", (case_id,)).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    try:
        d["attribution_matrix"] = json.loads(d["attribution_matrix"])
    except Exception:
        d["attribution_matrix"] = []
    return d