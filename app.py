import socket
from urllib.parse import urlparse
"""
app.py

CEO Execution Tracker (Streamlit + Supabase Postgres)

What you get:
- Daily/Weekly/Monthly checklists
- Notes per item
- Base priority per item
- Weekly priority overrides (per week)
- Google Calendar daily event links (New York) using "top focus item"
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, time as dtime
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlencode, quote

import psycopg2
import streamlit as st

PERIODS = ["Daily", "Weekly", "Monthly"]
USER_ID = "me"
NY_TZ = "America/New_York"


# ----------------------------
# DB
# ----------------------------

@dataclass(frozen=True)
class DbConfig:
    database_url: str


def get_database_url() -> str:
    if "DATABASE_URL" in st.secrets:
        return str(st.secrets["DATABASE_URL"])
    if "DATABASE_URL" in os.environ:
        return os.environ["DATABASE_URL"]
    raise RuntimeError("Missing DATABASE_URL. Add it in Streamlit Secrets.")


def conn(cfg: DbConfig):
    """
    Force IPv4 because Streamlit Cloud sometimes can't connect to Supabase over IPv6.
    Works with DATABASE_URL in either URL or DSN format.
    """
    dsn = cfg.database_url.strip()

    # DSN format: "dbname=... user=... password=... host=... port=... sslmode=require"
    if "://" not in dsn:
        # If DSN includes host=..., force hostaddr (IPv4)
        m = re.search(r"\bhost=([^\s]+)", dsn)
        if m:
            host = m.group(1)
            hostaddr = socket.gethostbyname(host)  # IPv4
            if "hostaddr=" not in dsn:
                dsn = f"{dsn} hostaddr={hostaddr}"
        return psycopg2.connect(dsn)

    # URL format: "postgresql://user:pass@host:port/db?sslmode=require"
    u = urlparse(dsn)
    host = u.hostname or ""
    hostaddr = socket.gethostbyname(host)  # IPv4

    return psycopg2.connect(
        dbname=(u.path or "/postgres").lstrip("/"),
        user=u.username or "postgres",
        password=u.password or "",
        host=host,
        hostaddr=hostaddr,          # <-- forces IPv4
        port=u.port or 5432,
        sslmode="require",          # Supabase requires SSL
    )


def init_db(cfg: DbConfig) -> None:
    with conn(cfg) as c:
        with c.cursor() as cur:
            cur.execute(
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
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS entries (
                    user_id TEXT NOT NULL,
                    bucket TEXT NOT NULL,
                    period TEXT NOT NULL,
                    item TEXT NOT NULL,
                    completed BOOLEAN NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (user_id, bucket, period, item)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS item_priority_overrides (
                    user_id TEXT NOT NULL,
                    period TEXT NOT NULL,
                    week_monday TEXT NOT NULL,
                    item TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    PRIMARY KEY (user_id, period, week_monday, item)
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
        iso_year, iso_week, _ = d.isocalendar()
        return f"{iso_year}-W{iso_week:02d}"
    if period == "Monthly":
        return f"{d.year:04d}-{d.month:02d}"
    raise ValueError("Unknown period")


# ----------------------------
# Items + priorities
# ----------------------------

def seed_items(cfg: DbConfig, period: str, items: List[str]) -> None:
    cleaned = [x.strip() for x in items if x and x.strip()]
    with conn(cfg) as c:
        with c.cursor() as cur:
            cur.execute("DELETE FROM items WHERE period=%s", (period,))
            cur.executemany(
                "INSERT INTO items (period, position, text, priority) VALUES (%s, %s, %s, 0)",
                [(period, i + 1, t) for i, t in enumerate(cleaned)],
            )
        c.commit()


def get_items_with_effective_priority(cfg: DbConfig, user_id: str, period: str, for_date: date) -> List[str]:
    wk = monday_of_week(for_date).isoformat()
    with conn(cfg) as c:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT i.text
                FROM items i
                LEFT JOIN item_priority_overrides o
                  ON o.user_id=%s AND o.period=i.period AND o.week_monday=%s AND o.item=i.text
                WHERE i.period=%s
                ORDER BY COALESCE(o.priority, i.priority) DESC, i.position ASC
                """,
                (user_id, wk, period),
            )
            return [r[0] for r in cur.fetchall()]


def get_items_for_edit(cfg: DbConfig, period: str) -> List[Tuple[int, str, int]]:
    with conn(cfg) as c:
        with c.cursor() as cur:
            cur.execute(
                "SELECT position, text, priority FROM items WHERE period=%s ORDER BY position ASC",
                (period,),
            )
            return [(int(p), str(t), int(pr)) for p, t, pr in cur.fetchall()]


def save_base_priorities(cfg: DbConfig, period: str, updates: List[Tuple[int, int]]) -> None:
    if not updates:
        return
    with conn(cfg) as c:
        with c.cursor() as cur:
            cur.executemany(
                "UPDATE items SET priority=%s WHERE period=%s AND position=%s",
                [(prio, period, pos) for pos, prio in updates],
            )
        c.commit()


def get_weekly_overrides(cfg: DbConfig, user_id: str, period: str, week_monday: date) -> Dict[str, int]:
    wk = week_monday.isoformat()
    with conn(cfg) as c:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT item, priority
                FROM item_priority_overrides
                WHERE user_id=%s AND period=%s AND week_monday=%s
                """,
                (user_id, period, wk),
            )
            return {str(item): int(p) for item, p in cur.fetchall()}


def save_weekly_overrides(cfg: DbConfig, user_id: str, period: str, week_monday: date, overrides: Dict[str, int]) -> None:
    wk = week_monday.isoformat()
    with conn(cfg) as c:
        with c.cursor() as cur:
            for item, prio in overrides.items():
                cur.execute(
                    """
                    INSERT INTO item_priority_overrides (user_id, period, week_monday, item, priority)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, period, week_monday, item)
                    DO UPDATE SET priority=EXCLUDED.priority
                    """,
                    (user_id, period, wk, item, int(prio)),
                )
        c.commit()


def clear_weekly_overrides(cfg: DbConfig, user_id: str, period: str, week_monday: date) -> None:
    wk = week_monday.isoformat()
    with conn(cfg) as c:
        with c.cursor() as cur:
            cur.execute(
                "DELETE FROM item_priority_overrides WHERE user_id=%s AND period=%s AND week_monday=%s",
                (user_id, period, wk),
            )
        c.commit()


# ----------------------------
# Entries
# ----------------------------

def load_entries(cfg: DbConfig, user_id: str, bucket: str, period: str) -> Dict[str, Tuple[bool, str]]:
    with conn(cfg) as c:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT item, completed, note
                FROM entries
                WHERE user_id=%s AND bucket=%s AND period=%s
                """,
                (user_id, bucket, period),
            )
            rows = cur.fetchall()
    return {item: (bool(done), note or "") for item, done, note in rows}


def save_entries(cfg: DbConfig, user_id: str, bucket: str, period: str, entries: Dict[str, Tuple[bool, str]]) -> None:
    with conn(cfg) as c:
        with c.cursor() as cur:
            cur.execute(
                "DELETE FROM entries WHERE user_id=%s AND bucket=%s AND period=%s",
                (user_id, bucket, period),
            )
            cur.executemany(
                """
                INSERT INTO entries (user_id, bucket, period, item, completed, note)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                [(user_id, bucket, period, item, done, note or "") for item, (done, note) in entries.items()],
            )
        c.commit()


def top_focus_item(cfg: DbConfig, user_id: str, day: date) -> str:
    items = get_items_with_effective_priority(cfg, user_id, "Daily", day)
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
    st.set_page_config(page_title="CEO Execution Tracker", layout="wide")
    st.title("📈 CEO Execution Tracker")

    cfg = DbConfig(database_url=get_database_url())
    init_db(cfg)

    # Sidebar: one-click setup + admin
    with st.sidebar:
        st.header("Setup")
        if st.button("🚀 First-time setup (seed starter lists)", use_container_width=True):
            seed_items(cfg, "Daily", STARTER_DAILY)
            seed_items(cfg, "Weekly", STARTER_WEEKLY)
            seed_items(cfg, "Monthly", STARTER_MONTHLY)
            st.success("Seeded starter lists.")
            st.rerun()

        st.divider()
        st.header("Admin")

        with st.expander("Edit base priorities", expanded=False):
            p = st.selectbox("Period", PERIODS, key="baseprio_period")
            rows = get_items_for_edit(cfg, p)
            updates: List[Tuple[int, int]] = []
            for pos, text, prio in rows:
                new_prio = st.number_input(
                    f"{pos}. {text}",
                    min_value=-10,
                    max_value=10,
                    value=int(prio),
                    step=1,
                    key=f"baseprio::{p}::{pos}",
                )
                if int(new_prio) != int(prio):
                    updates.append((pos, int(new_prio)))
            if st.button("Save base priorities", use_container_width=True, key="save_base"):
                save_base_priorities(cfg, p, updates)
                st.success("Saved.")
                st.rerun()

        with st.expander("Weekly priority overrides (this week)", expanded=False):
            p = st.selectbox("Period (overrides)", PERIODS, key="wkprio_period")
            wk = monday_of_week(date.today())
            st.caption(f"Week starts: {wk.isoformat()}")
            base = get_items_for_edit(cfg, p)
            ov = get_weekly_overrides(cfg, USER_ID, p, wk)

            to_save: Dict[str, int] = {}
            for pos, text, base_pr in base:
                effective = ov.get(text, int(base_pr))
                new_pr = st.number_input(
                    f"{pos}. {text}",
                    min_value=-10,
                    max_value=10,
                    value=int(effective),
                    step=1,
                    key=f"wkprio::{p}::{wk.isoformat()}::{pos}",
                )
                if int(new_pr) != int(base_pr):
                    to_save[text] = int(new_pr)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Save overrides", use_container_width=True, key="wk_save"):
                    save_weekly_overrides(cfg, USER_ID, p, wk, to_save)
                    st.success("Saved overrides.")
                    st.rerun()
            with c2:
                if st.button("Clear overrides", use_container_width=True, key="wk_clear"):
                    clear_weekly_overrides(cfg, USER_ID, p, wk)
                    st.success("Cleared overrides.")
                    st.rerun()

    tab_track, tab_calendar = st.tabs(["✅ Track", "📅 Calendar (Android)"])

    with tab_track:
        period = st.selectbox("Period", PERIODS, key="track_period")
        d = st.date_input("Date", value=date.today(), key="track_date")
        bucket = bucket_key(period, d)

        items = get_items_with_effective_priority(cfg, USER_ID, period, d)
        if not items:
            st.warning("No items yet. Click sidebar: First-time setup.")
            st.stop()

        # Main-page item editor
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
