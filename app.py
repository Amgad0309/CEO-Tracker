# app.py
from __future__ import annotations

"""
CEO Execution Tracker (Streamlit + SQLite)

Advanced-but-simple local version:
- Daily / Weekly / Monthly checklists
- Notes per item
- Focus Mode (single item) with Prev/Next + auto-save + auto-advance on completion
- Pomodoro timer (work sessions logged to SQLite)
- Basic dashboards (today completion + focus minutes)
- Google Calendar event link (New York) with top focus item

Run:
  python -m pip install -r requirements.txt
  streamlit run app.py --server.address 0.0.0.0 --server.port 8501
"""

import math
import sqlite3
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, time as dtime
from typing import Dict, List, Tuple
from urllib.parse import urlencode, quote

import streamlit as st


# -------------------------
# Config
# -------------------------

DB_PATH = "ceo_tracker.db"
USER_ID = "me"
PERIODS = ["Daily", "Weekly", "Monthly"]
NY_TZ = "America/New_York"


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


# -------------------------
# Helpers
# -------------------------

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


def completion_pct(done: int, total: int) -> float:
    return (done / total * 100.0) if total else 0.0


def ceil_div(a: int, b: int) -> int:
    return int(math.ceil(a / max(1, b)))


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


# -------------------------
# DB
# -------------------------

@dataclass(frozen=True)
class DbConfig:
    path: str = DB_PATH


def get_conn(cfg: DbConfig) -> sqlite3.Connection:
    c = sqlite3.connect(cfg.path, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_db(cfg: DbConfig) -> None:
    with get_conn(cfg) as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS items (
                period TEXT NOT NULL,
                position INTEGER NOT NULL,
                text TEXT NOT NULL,
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
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS pomodoro_daily (
                user_id TEXT NOT NULL,
                day TEXT NOT NULL,
                work_sessions INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (user_id, day)
            )
            """
        )
        c.commit()


# -------------------------
# Items
# -------------------------

def seed_items(cfg: DbConfig, period: str, items: List[str]) -> None:
    cleaned = [x.strip() for x in items if x and x.strip()]
    with get_conn(cfg) as c:
        c.execute("DELETE FROM items WHERE period=?", (period,))
        c.executemany(
            "INSERT INTO items (period, position, text) VALUES (?, ?, ?)",
            [(period, i + 1, t) for i, t in enumerate(cleaned)],
        )
        c.commit()


def get_items(cfg: DbConfig, period: str) -> List[str]:
    with get_conn(cfg) as c:
        rows = c.execute(
            "SELECT text FROM items WHERE period=? ORDER BY position ASC",
            (period,),
        ).fetchall()
    return [r["text"] for r in rows]


# -------------------------
# Entries
# -------------------------

def load_entries(cfg: DbConfig, user_id: str, bucket: str, period: str) -> Dict[str, Tuple[bool, str]]:
    with get_conn(cfg) as c:
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
    with get_conn(cfg) as c:
        c.execute(
            "DELETE FROM entries WHERE user_id=? AND bucket=? AND period=?",
            (user_id, bucket, period),
        )
        c.executemany(
            """
            INSERT INTO entries (user_id, bucket, period, item, completed, note)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [(user_id, bucket, period, item, int(done), (note or "").rstrip()) for item, (done, note) in entries.items()],
        )
        c.commit()


def get_today_done_total(cfg: DbConfig, user_id: str) -> Tuple[int, int]:
    b = date.today().isoformat()
    with get_conn(cfg) as c:
        row = c.execute(
            """
            SELECT SUM(completed) AS done, COUNT(*) AS total
            FROM entries
            WHERE user_id=? AND period='Daily' AND bucket=?
            """,
            (user_id, b),
        ).fetchone()
    done = int(row["done"] or 0)
    total = int(row["total"] or 0)
    return done, total


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


# -------------------------
# Pomodoro
# -------------------------

def record_work_session(cfg: DbConfig, user_id: str, day: date, inc: int = 1) -> None:
    with get_conn(cfg) as c:
        c.execute(
            """
            INSERT INTO pomodoro_daily (user_id, day, work_sessions)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, day) DO UPDATE SET
              work_sessions = work_sessions + excluded.work_sessions
            """,
            (user_id, day.isoformat(), int(inc)),
        )
        c.commit()


def get_work_sessions(cfg: DbConfig, user_id: str, day: date) -> int:
    with get_conn(cfg) as c:
        row = c.execute(
            "SELECT work_sessions FROM pomodoro_daily WHERE user_id=? AND day=?",
            (user_id, day.isoformat()),
        ).fetchone()
    return int(row["work_sessions"]) if row else 0


def init_pomodoro_state() -> None:
    st.session_state.setdefault("pomo_running", False)
    st.session_state.setdefault("pomo_end_ts", 0.0)
    st.session_state.setdefault("pomo_work_min", 25)
    st.session_state.setdefault("pomo_break_min", 5)
    st.session_state.setdefault("pomo_mode", "work")  # work|break


def start_pomodoro() -> None:
    init_pomodoro_state()
    minutes = st.session_state.pomo_work_min if st.session_state.pomo_mode == "work" else st.session_state.pomo_break_min
    st.session_state.pomo_running = True
    st.session_state.pomo_end_ts = time.time() + int(minutes) * 60


def stop_pomodoro() -> None:
    st.session_state.pomo_running = False
    st.session_state.pomo_end_ts = 0.0


def render_pomodoro(cfg: DbConfig) -> None:
    init_pomodoro_state()

    st.markdown("### 🍅 Pomodoro")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state.pomo_work_min = st.number_input("Work (min)", 5, 90, int(st.session_state.pomo_work_min), 5)
    with c2:
        st.session_state.pomo_break_min = st.number_input("Break (min)", 3, 30, int(st.session_state.pomo_break_min), 1)
    with c3:
        st.selectbox("Mode", ["work", "break"], key="pomo_mode")

    if st.session_state.pomo_running:
        remaining = int(st.session_state.pomo_end_ts - time.time())
        if remaining <= 0:
            if st.session_state.pomo_mode == "work":
                record_work_session(cfg, USER_ID, date.today(), inc=1)
                st.toast("✅ Work session logged (+1)", icon="🍅")
            stop_pomodoro()
            st.session_state.pomo_mode = "break" if st.session_state.pomo_mode == "work" else "work"
            st.rerun()
        mm, ss = divmod(max(0, remaining), 60)
        st.metric("Time left", f"{mm:02d}:{ss:02d}")
        st.progress(min(1.0, remaining / max(1, int((st.session_state.pomo_work_min if st.session_state.pomo_mode=='work' else st.session_state.pomo_break_min) * 60)))))
        time.sleep(1)
        st.rerun()
    else:
        st.caption("Start a timer. Work sessions auto-log when they finish.")

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Start", use_container_width=True):
            start_pomodoro()
            st.rerun()
    with b2:
        if st.button("Stop", use_container_width=True):
            stop_pomodoro()
            st.rerun()


# -------------------------
# Focus Mode
# -------------------------

def focus_key(period: str, bucket: str) -> str:
    return f"focus_idx::{period}::{bucket}"


def get_focus_idx(period: str, bucket: str, n: int) -> int:
    k = focus_key(period, bucket)
    st.session_state.setdefault(k, 0)
    return max(0, min(int(st.session_state[k]), max(0, n - 1)))


def set_focus_idx(period: str, bucket: str, idx: int, n: int) -> None:
    st.session_state[focus_key(period, bucket)] = max(0, min(int(idx), max(0, n - 1)))


def render_focus_mode(cfg: DbConfig, period: str, bucket: str, items: List[str], saved: Dict[str, Tuple[bool, str]]) -> Dict[str, Tuple[bool, str]]:
    n = len(items)
    idx = get_focus_idx(period, bucket, n)
    item = items[idx]

    def build_state() -> Dict[str, Tuple[bool, str]]:
        out: Dict[str, Tuple[bool, str]] = {}
        for it in items:
            cbk = f"cb::{period}::{bucket}::{it}"
            nk = f"note::{period}::{bucket}::{it}"
            d0, n0 = saved.get(it, (False, ""))
            out[it] = (bool(st.session_state.get(cbk, d0)), str(st.session_state.get(nk, n0)))
        return out

    def autosave() -> None:
        save_entries(cfg, USER_ID, bucket, period, build_state())

    nav1, nav2, nav3 = st.columns([0.2, 0.6, 0.2])
    with nav1:
        if st.button("⬅️ Prev", disabled=(idx == 0), use_container_width=True):
            autosave()
            set_focus_idx(period, bucket, idx - 1, n)
            st.rerun()
    with nav3:
        if st.button("Next ➡️", disabled=(idx == n - 1), use_container_width=True):
            autosave()
            set_focus_idx(period, bucket, idx + 1, n)
            st.rerun()
    with nav2:
        st.progress((idx + 1) / max(1, n))
        st.caption(f"Item {idx + 1} / {n}")

    st.markdown(f"## {item}")

    d0, n0 = saved.get(item, (False, ""))
    cbk = f"cb::{period}::{bucket}::{item}"
    nk = f"note::{period}::{bucket}::{item}"
    st.session_state.setdefault(cbk, d0)
    st.session_state.setdefault(nk, n0)

    prev_done = bool(st.session_state[cbk])
    done = st.checkbox("Completed", key=cbk)
    note = st.text_area("Note", height=260, key=nk)

    st.caption("Auto-saves when you navigate or when you mark completed.")

    if done and not prev_done:
        autosave()
        if idx < n - 1:
            set_focus_idx(period, bucket, idx + 1, n)
            st.rerun()

    return build_state()


# -------------------------
# UI
# -------------------------

def main() -> None:
    st.set_page_config(page_title="CEO Execution Tracker", layout="wide")
    st.title("📈 CEO Execution Tracker (Local)")

    cfg = DbConfig()
    init_db(cfg)

    with st.sidebar:
        st.header("Setup")
        if st.button("🚀 Seed starter lists", use_container_width=True):
            seed_items(cfg, "Daily", STARTER_DAILY)
            seed_items(cfg, "Weekly", STARTER_WEEKLY)
            seed_items(cfg, "Monthly", STARTER_MONTHLY)
            st.success("Seeded.")
            st.rerun()

        st.divider()
        st.header("Today")
        done, total = get_today_done_total(cfg, USER_ID)
        sessions = get_work_sessions(cfg, USER_ID, date.today())
        st.metric("Daily completion", f"{completion_pct(done, total):.1f}%")
        st.metric("Work sessions (today)", sessions)
        st.caption("If this is your first run, click: Seed starter lists.")

    tab_track, tab_calendar, tab_stats = st.tabs(["✅ Track", "📅 Calendar", "📊 Stats"])

    with tab_track:
        period = st.selectbox("Period", PERIODS, key="period")
        d = st.date_input("Date", value=date.today(), key="day")
        bucket = bucket_key(period, d)

        items = get_items(cfg, period)
        if not items:
            st.warning("No items yet. Click sidebar → Seed starter lists.")
            st.stop()

        if st.toggle("✏️ Edit items (this period)", value=False):
            edited = st.text_area("One per line. Saving replaces the list.", value="\n".join(items), height=200)
            if st.button("Save item list", use_container_width=True):
                seed_items(cfg, period, [x.strip() for x in edited.splitlines() if x.strip()])
                st.success("Saved.")
                st.rerun()

        saved = load_entries(cfg, USER_ID, bucket, period)

        st.divider()
        render_pomodoro(cfg)
        st.divider()

        focus = st.toggle("🎯 Focus Mode (single item)", value=True)

        if focus:
            current = render_focus_mode(cfg, period, bucket, items, saved)
            if st.button("Save now", use_container_width=True):
                save_entries(cfg, USER_ID, bucket, period, current)
                st.success("Saved.")
                st.rerun()
        else:
            current: Dict[str, Tuple[bool, str]] = {}
            for it in items:
                d0, n0 = saved.get(it, (False, ""))
                done = st.checkbox(it, value=d0, key=f"cb::{period}::{bucket}::{it}")
                note = st.text_area(f"Note — {it}", value=n0, height=90, key=f"note::{period}::{bucket}::{it}")
                current[it] = (bool(done), str(note))

            done_count = sum(1 for v in current.values() if v[0])
            st.metric("Completion", f"{completion_pct(done_count, len(items)):.1f}%")

            if st.button("Save", type="primary", use_container_width=True):
                save_entries(cfg, USER_ID, bucket, period, current)
                st.success("Saved.")
                st.rerun()

    with tab_calendar:
        st.subheader("Google Calendar Event Link (New York)")
        day = st.date_input("Event day", value=date.today(), key="cal_day")
        start_t = st.time_input("Start time", value=dtime(9, 0), key="cal_start")
        minutes = st.number_input("Duration (minutes)", 5, 600, 60, 5, key="cal_minutes")

        focus_item = top_focus_item(cfg, USER_ID, day)
        title = f"Deep Work — {focus_item} ({int(minutes)}m)"
        start_dt = datetime.combine(day, start_t)
        end_dt = start_dt + timedelta(minutes=int(minutes))
        details = "One outcome. No distractions."

        st.link_button("Add to Google Calendar", gcal_link(title, start_dt, end_dt, details), use_container_width=True)
        st.caption("On Samsung: tap button → Google Calendar opens with event prefilled.")

    with tab_stats:
        st.subheader("Today snapshot")
        done, total = get_today_done_total(cfg, USER_ID)
        sessions = get_work_sessions(cfg, USER_ID, date.today())
        c1, c2, c3 = st.columns(3)
        c1.metric("Done / Total", f"{done} / {total}")
        c2.metric("Completion %", f"{completion_pct(done, total):.1f}%")
        c3.metric("Focus minutes", sessions * int(st.session_state.get("pomo_work_min", 25)))


if __name__ == "__main__":
    main()
