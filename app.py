# app.py
from __future__ import annotations

"""
CEO Execution Tracker (Streamlit + SQLite)

Zero-cloud-DB version:
- Daily/Weekly/Monthly checklists
- Notes per item
- Edit items on main page
- One-click seed starter lists
- Google Calendar link (New York)

Run locally:
  streamlit run app.py

Tip for phone:
  Run on laptop, then open the Network URL shown by Streamlit on your phone.
"""

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, time as dtime
from typing import Dict, List, Tuple
from urllib.parse import urlencode, quote

import streamlit as st

PERIODS = ["Daily", "Weekly", "Monthly"]
USER_ID = "me"
NY_TZ = "America/New_York"


# ----------------------------
# SQLite DB
# ----------------------------

@dataclass(frozen=True)
class DbConfig:
    path: str = "ceo_tracker.db"


def conn(cfg: DbConfig) -> sqlite3.Connection:
    c = sqlite3.connect(cfg.path, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_db(cfg: DbConfig) -> None:
    with conn(cfg) as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS items (
                period TEXT NOT NULL,
                position INTEGER NOT NULL,
                text TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (period, position)
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                user_id TEXT NOT NULL,
                bucket TEXT NOT NULL,
                period TEXT NOT NULL,
                item TEXT NOT NULL,
                completed INTEGER NOT NULL,
                note TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (user_id, bucket, period, item)
            )
            """
        )
        c.commit()


# ----------------------------
# Period keys
# ----------------------------

def monday_of_week(d: date) -> date:
    return d - timedelta(days=d.weekday())


def bucket_key(period: str, d: date) -> str:
    if period == "Daily":
        return d.isoformat()
    if period == "Weekly":
        y, w, _ = d.isocalendar()
        return f"{y}-W{w:02d}"
    if period == "Monthly":
        return f"{d.year:04d}-{d.month:02d}"
    raise ValueError("Unknown period")


# ----------------------------
# Items
# ----------------------------

def seed_items(cfg: DbConfig, period: str, items: List[str]) -> None:
    cleaned = [x.strip() for x in items if x and x.strip()]
    with conn(cfg) as c:
        c.execute("DELETE FROM items WHERE period=?", (period,))
        c.executemany(
            "INSERT INTO items (period, position, text, priority) VALUES (?, ?, ?, 0)",
            [(period, i + 1, t) for i, t in enumerate(cleaned)],
        )
        c.commit()


def get_items(cfg: DbConfig, period: str) -> List[str]:
    with conn(cfg) as c:
        rows = c.execute(
            "SELECT text FROM items WHERE period=? ORDER BY priority DESC, position ASC",
            (period,),
        ).fetchall()
    return [r["text"] for r in rows]


# ----------------------------
# Entries
# ----------------------------

def load_entries(cfg: DbConfig, user_id: str, bucket: str, period: str) -> Dict[str, Tuple[bool, str]]:
    with conn(cfg) as c:
        rows = c.execute(
            """
            SELECT item, completed, note
            FROM entries
            WHERE user_id=? AND bucket=? AND period=?
            """,
            (user_id, bucket, period),
        ).fetchall()
    return {r["item"]: (bool(r["completed"]), r["note"] or "") for r in rows}


def save_entries(cfg: DbConfig, user_id: str, bucket: str, period: str, entries: Dict[str, Tuple[bool, str]]) -> None:
    with conn(cfg) as c:
        c.execute(
            "DELETE FROM entries WHERE user_id=? AND bucket=? AND period=?",
            (user_id, bucket, period),
        )
        c.executemany(
            """
            INSERT INTO entries (user_id, bucket, period, item, completed, note)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [(user_id, bucket, period, item, int(done), note or "") for item, (done, note) in entries.items()],
        )
        c.commit()


def top_focus_item(cfg: DbConfig, user_id: str, day: date) -> str:
    items = get_items(cfg, "Daily")
    if not items:
        return "Plan / Review"
    saved = load_entries(cfg, user_id, day.isoformat(), "Daily")
    for it in items:
        done, _ = saved.get(it, (False, ""))
        if not done:
            return it
    return "Plan / Review"


# ----------------------------
# Google Calendar link
# ----------------------------

def gcal_dt(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M%S")


def gcal_link(title: str, start_local: datetime, end_local: datetime, details: str, tz: str = NY_TZ) -> str:
    params = {
        "action": "TEMPLATE",
        "text": title,
        "dates": f"{gcal_dt(start_local)}/{gcal_dt(end_local)}",
        "ctz": tz,
        "details": details or "",
    }
    return "https://calendar.google.com/calendar/render?" + urlencode(params, quote_via=quote)


# ----------------------------
# UI
# ----------------------------

STARTER_DAILY = [
    "Top 1 outcome (ship today)",
    "Revenue / pipeline action",
    "Customer / client value action",
    "Team / people action",
    "Ops / systems action",
    "Deep work block (no distractions)",
    "Workout / health non-negotiable",
    "Learning (15–30 min)",
    "Inbox sweep (15 min) + next actions",
    "End-of-day review + plan tomorrow",
]

STARTER_WEEKLY = [
    "Set weekly #1 outcome + success metric",
    "Review KPIs (revenue, pipeline, retention)",
    "Plan deep work blocks on calendar",
    "Prioritize top 3 initiatives",
    "1:1s / key conversations scheduled",
    "Process improvement (remove 1 bottleneck)",
    "Customer feedback review + action",
    "Finance review (cash, runway, invoices)",
    "Systems cleanup (docs, tools, automations)",
    "Weekly review + next week plan",
]

STARTER_MONTHLY = [
    "Review monthly goals vs actuals",
    "Define next month top 3 priorities",
    "Budget + financial review",
    "Customer/market insights review",
    "Team performance + hiring plan",
    "Product/ops roadmap update",
    "Systems & process audit",
    "Risk review + mitigation plan",
    "Personal performance review",
    "Reset habits + schedule big rocks",
]


def main() -> None:
    st.set_page_config(page_title="CEO Execution Tracker (SQLite)", layout="wide")
    st.title("📈 CEO Execution Tracker (SQLite)")

    cfg = DbConfig()
    init_db(cfg)

    with st.sidebar:
        st.header("Setup")
        if st.button("🚀 First-time setup (seed starter lists)", use_container_width=True):
            seed_items(cfg, "Daily", STARTER_DAILY)
            seed_items(cfg, "Weekly", STARTER_WEEKLY)
            seed_items(cfg, "Monthly", STARTER_MONTHLY)
            st.success("Seeded starter lists.")
            st.rerun()

    tab_track, tab_calendar = st.tabs(["✅ Track", "📅 Calendar (Android)"])

    with tab_track:
        period = st.selectbox("Period", PERIODS, key="track_period")
        d = st.date_input("Date", value=date.today(), key="track_date")
        bucket = bucket_key(period, d)

        items = get_items(cfg, period)
        if not items:
            st.warning("No items yet. Click sidebar: First-time setup.")
            st.stop()

        if st.toggle("✏️ Edit items (this period)", value=False, key="edit_items_toggle"):
            edited = st.text_area(
                "One item per line (saving replaces the whole list).",
                value="\n".join(items),
                height=200,
                key="edit_items_text",
            )
            if st.button("Save item list", use_container_width=True, key="edit_items_save"):
                seed_items(cfg, period, [x.strip() for x in edited.splitlines() if x.strip()])
                st.success("Saved items.")
                st.rerun()

        saved = load_entries(cfg, USER_ID, bucket, period)

        current: Dict[str, Tuple[bool, str]] = {}
        for it in items:
            done0, note0 = saved.get(it, (False, ""))
            done = st.checkbox(it, value=done0, key=f"cb::{bucket}::{it}")
            note = st.text_area(f"Note — {it}", value=note0, height=90, key=f"note::{bucket}::{it}")
            current[it] = (bool(done), str(note))

        if st.button("Save", type="primary", use_container_width=True, key="save_entries"):
            save_entries(cfg, USER_ID, bucket, period, current)
            st.success("Saved.")
            st.rerun()

    with tab_calendar:
        st.subheader("📱 Google Calendar Daily Link (New York)")
        d = st.date_input("Day for event", value=date.today(), key="cal_day")
        start_t = st.time_input("Start time", value=dtime(9, 0), key="cal_start")
        minutes = st.number_input("Duration (minutes)", min_value=5, max_value=600, value=60, step=5, key="cal_dur")

        focus = top_focus_item(cfg, USER_ID, d)
        title = f"Deep Work — {focus} ({minutes}m)"
        start_dt = datetime.combine(d, start_t)
        end_dt = start_dt + timedelta(minutes=int(minutes))
        details = "One outcome. No distractions."

        url = gcal_link(title, start_dt, end_dt, details, tz=NY_TZ)
        st.link_button("Add to Google Calendar", url, use_container_width=True)
        st.caption("Tap on Android → Google Calendar opens with the event prefilled.")


if __name__ == "__main__":
    main()
