"""
CEO Execution Tracker (Streamlit + SQLite)

Features
- Daily / Weekly / Monthly checklists
- Markdown notes (safe render) + bare URL linkify
- Focus Mode (single item) with Prev/Next, auto-save, auto-advance on completion
- Pomodoro timer (in Focus Mode) logs:
    - daily totals (pomodoro_daily)
    - event timestamps (pomodoro_events) for "recent pace" ETA
- Priorities:
    - Base priority per item (items.priority)
    - Weekly priority overrides per item (item_priority_overrides)
- Deep Work Score:
    - 2-component: Daily completion % + Focus minutes
    - Optional 3rd component: Weekly completion % OR Weekly focused minutes
- Dashboards:
    - Today: score, remaining minutes, remaining items, ETA (today avg or recent pace)
    - Weekly: weekly score, minutes remaining, completion gap
- Weekly plan:
    - Mon–Fri-only option
    - Availability caps per day, optional AM/PM caps
    - Suggested AM/PM split snapped to whole pomodoros, optional 1 partial block
    - Copyable text plan + calendar-friendly schedule
    - Google Calendar "daily big block" links (New York) with top focus item in title
- Notes viewer with search
- Admin tools in sidebar:
    - Seed/Replace items per period
    - Edit base priorities
    - Edit weekly priority overrides (current week)

Run locally: streamlit run app.py
Deploy: Streamlit Community Cloud
"""

from __future__ import annotations

import math
import re
import psycopg2
import os
from typing import Any
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone, time as dtime
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlencode, quote

import streamlit as st


# -------------------------
# Config
# -------------------------

DB_NAME = "checklist.db"
SINGLE_USER_ID = "me"
PERIODS = ["Daily", "Weekly", "Monthly"]
NY_TZ = "America/New_York"

BARE_URL_REGEX = re.compile(r"(?<![\(\[<])(https?://[^\s\)>\]]+)")


# -------------------------
# Helpers
# -------------------------

def ceil_div(a: int, b: int) -> int:
    return int(math.ceil(a / max(1, b)))


def monday_of_week(d: date) -> date:
    return d - timedelta(days=d.weekday())


def week_days(monday: date) -> List[date]:
    return [monday + timedelta(days=i) for i in range(7)]


def remaining_days_in_week(today: date, weekdays_only: bool) -> List[date]:
    m = monday_of_week(today)
    days = [d for d in week_days(m) if d >= today]
    if weekdays_only:
        days = [d for d in days if d.weekday() < 5]  # Mon..Fri
    return days


def completion_pct(done: int, total: int) -> float:
    return (done / total * 100.0) if total else 0.0


def linkify_bare_urls(text: str) -> str:
    if not text:
        return ""

    def repl(m: re.Match[str]) -> str:
        url = m.group(1)
        return f"[{url}]({url})"

    return BARE_URL_REGEX.sub(repl, text)


def render_markdown_note(note: str) -> None:
    note = (note or "").strip()
    if not note:
        st.caption("_No content_")
        return
    st.markdown(linkify_bare_urls(note), unsafe_allow_html=False)


def bucket_key(period: str, d: date) -> str:
    if period == "Daily":
        return d.isoformat()
    if period == "Weekly":
        iso_year, iso_week, _ = d.isocalendar()
        return f"{iso_year}-W{iso_week:02d}"
    if period == "Monthly":
        return f"{d.year:04d}-{d.month:02d}"
    raise ValueError("Unknown period")


def bucket_to_anchor_date(period: str, key: str) -> date:
    if period == "Daily":
        return datetime.strptime(key, "%Y-%m-%d").date()
    if period == "Weekly":
        y, w = key.split("-W")
        return date.fromisocalendar(int(y), int(w), 1)
    if period == "Monthly":
        y, m = key.split("-")
        return date(int(y), int(m), 1)
    raise ValueError("Unknown period")


# -------------------------
# DB (Postgres via Supabase) - SQLite-compatible adapter
# -------------------------

@dataclass(frozen=True)
class DbConfig:
    database_url: str


def _get_database_url() -> str:
    # Prefer Streamlit secrets (Streamlit Cloud), then env var fallback.
    if "DATABASE_URL" in st.secrets:
        return str(st.secrets["DATABASE_URL"])
    return os.environ["DATABASE_URL"]


class PgConn:
    """
    Tiny adapter so existing sqlite-style code works:
    - conn.execute(sql, params) -> cursor
    - conn.executemany(sql, seq_of_params)
    - conn.commit()
    """
    def __init__(self, database_url: str):
        self._database_url = database_url
        self._conn: psycopg2.extensions.connection | None = None

    def __enter__(self) -> "PgConn":
        self._conn = psycopg2.connect(self._database_url)
        self._conn.autocommit = False
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self._conn is not None
        try:
            if exc_type is None:
                self._conn.commit()
            else:
                self._conn.rollback()
        finally:
            self._conn.close()
            self._conn = None

    def commit(self) -> None:
        assert self._conn is not None
        self._conn.commit()

    @staticmethod
    def _sql(sql: str) -> str:
        # Convert sqlite qmark params "?" -> psycopg2 "%s"
        return sql.replace("?", "%s")

    def execute(self, sql: str, params: Tuple[Any, ...] | List[Any] | None = None):
        assert self._conn is not None
        cur = self._conn.cursor()
        cur.execute(self._sql(sql), params or ())
        return cur

    def executemany(self, sql: str, seq_of_params):
        assert self._conn is not None
        cur = self._conn.cursor()
        cur.executemany(self._sql(sql), seq_of_params)
        return cur


def get_conn(cfg: DbConfig) -> PgConn:
    return PgConn(cfg.database_url)


def init_db(cfg: DbConfig) -> None:
    with get_conn(cfg) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                user TEXT NOT NULL,
                bucket TEXT NOT NULL,
                period TEXT NOT NULL,
                item TEXT NOT NULL,
                completed INTEGER NOT NULL,
                note TEXT,
                PRIMARY KEY (user, bucket, period, item)
            )
            """
        )

        conn.execute(
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

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS item_priority_overrides (
                user TEXT NOT NULL,
                period TEXT NOT NULL,
                week_monday TEXT NOT NULL,
                item TEXT NOT NULL,
                priority INTEGER NOT NULL,
                PRIMARY KEY (user, period, week_monday, item)
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pomodoro_daily (
                user TEXT NOT NULL,
                day TEXT NOT NULL,
                work_sessions INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (user, day)
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pomodoro_events (
                user TEXT NOT NULL,
                ts_utc INTEGER NOT NULL,
                day TEXT NOT NULL
            )
            """
        )

        conn.commit()


def save_entries(
    cfg: DbConfig,
    user: str,
    bucket: str,
    period: str,
    entries: Dict[str, Tuple[bool, str]],
) -> None:
    with get_conn(cfg) as conn:
        conn.execute(
            "DELETE FROM entries WHERE user=? AND bucket=? AND period=?",
            (user, bucket, period),
        )
        conn.executemany(
            """
            INSERT INTO entries (user, bucket, period, item, completed, note)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (user, bucket, period, item, int(done), (note or "").rstrip())
                for item, (done, note) in entries.items()
            ],
        )
        conn.commit()


def load_entries(
    cfg: DbConfig,
    user: str,
    bucket: str,
    period: str,
) -> Dict[str, Tuple[bool, str]]:
    with get_conn(cfg) as conn:
        rows = conn.execute(
            """
            SELECT item, completed, COALESCE(note, '')
            FROM entries
            WHERE user=? AND bucket=? AND period=?
            """,
            (user, bucket, period),
        ).fetchall()
    return {item: (bool(c), n or "") for item, c, n in rows}


def fetch_notes(cfg: DbConfig, user: str, period: str) -> List[Tuple[str, str, str]]:
    with get_conn(cfg) as conn:
        rows = conn.execute(
            """
            SELECT bucket, item, COALESCE(note, '')
            FROM entries
            WHERE user=? AND period=? AND note IS NOT NULL AND note != ''
            ORDER BY bucket DESC
            """,
            (user, period),
        ).fetchall()
    return rows


def get_daily_completion_for_day(cfg: DbConfig, user: str, d: date) -> Tuple[float, int, int]:
    b = d.isoformat()
    with get_conn(cfg) as conn:
        row = conn.execute(
            """
            SELECT SUM(completed) AS done, COUNT(*) AS total
            FROM entries
            WHERE user=? AND period='Daily' AND bucket=?
            """,
            (user, b),
        ).fetchone()
    done = int((row[0] or 0) if row else 0)
    total = int((row[1] or 0) if row else 0)
    return completion_pct(done, total), done, total


def get_daily_completion_history(cfg: DbConfig, user: str, days: int) -> List[Tuple[date, float, int, int]]:
    end = date.today()
    start = end - timedelta(days=days - 1)

    with get_conn(cfg) as conn:
        rows = conn.execute(
            """
            SELECT bucket, SUM(completed) AS done, COUNT(*) AS total
            FROM entries
            WHERE user=? AND period='Daily' AND bucket BETWEEN ? AND ?
            GROUP BY bucket
            ORDER BY bucket ASC
            """,
            (user, start.isoformat(), end.isoformat()),
        ).fetchall()

    by_day: Dict[date, Tuple[int, int]] = {}
    for b, done, total in rows:
        dd = datetime.strptime(b, "%Y-%m-%d").date()
        by_day[dd] = (int(done or 0), int(total or 0))

    out: List[Tuple[date, float, int, int]] = []
    for i in range(days):
        dd = start + timedelta(days=i)
        done, total = by_day.get(dd, (0, 0))
        out.append((dd, completion_pct(done, total), done, total))
    return out


def get_weekly_completion_pct_for_week(cfg: DbConfig, user: str, week_monday: date) -> float:
    week_sunday = week_monday + timedelta(days=6)
    with get_conn(cfg) as conn:
        row = conn.execute(
            """
            SELECT SUM(completed) AS done, COUNT(*) AS total
            FROM entries
            WHERE user=? AND period='Daily' AND bucket BETWEEN ? AND ?
            """,
            (user, week_monday.isoformat(), week_sunday.isoformat()),
        ).fetchone()
    done = int((row[0] or 0) if row else 0)
    total = int((row[1] or 0) if row else 0)
    return completion_pct(done, total)


# -------------------------
# Items + Priorities
# -------------------------

def get_items(
    cfg: DbConfig,
    period: str,
    *,
    user: str,
    for_date: date,
) -> List[str]:
    """
    Items ordered by effective priority:
      effective_priority = COALESCE(weekly_override, base_priority)
    """
    wk = monday_of_week(for_date).isoformat()
    with get_conn(cfg) as conn:
        rows = conn.execute(
            """
            SELECT i.text
            FROM items i
            LEFT JOIN item_priority_overrides o
              ON o.user = ?
             AND o.period = i.period
             AND o.week_monday = ?
             AND o.item = i.text
            WHERE i.period = ?
            ORDER BY COALESCE(o.priority, i.priority) DESC, i.position ASC
            """,
            (user, wk, period),
        ).fetchall()
    return [r[0] for r in rows]


def seed_items(cfg: DbConfig, period: str, items: List[str]) -> None:
    cleaned = [x.strip() for x in items if x and x.strip()]
    with get_conn(cfg) as conn:
        conn.execute("DELETE FROM items WHERE period=?", (period,))
        conn.executemany(
            "INSERT INTO items (period, position, text, priority) VALUES (?, ?, ?, 0)",
            [(period, idx, text) for idx, text in enumerate(cleaned, start=1)],
        )
        conn.commit()


def get_items_for_edit(cfg: DbConfig, period: str) -> List[Tuple[int, str, int]]:
    with get_conn(cfg) as conn:
        rows = conn.execute(
            """
            SELECT position, text, priority
            FROM items
            WHERE period=?
            ORDER BY position ASC
            """,
            (period,),
        ).fetchall()
    return [(int(p), str(t), int(pr)) for p, t, pr in rows]


def update_base_priorities(cfg: DbConfig, period: str, updates: List[Tuple[int, int]]) -> None:
    """
    updates: [(position, priority)]
    """
    with get_conn(cfg) as conn:
        conn.executemany(
            "UPDATE items SET priority=? WHERE period=? AND position=?",
            [(prio, period, pos) for pos, prio in updates],
        )
        conn.commit()


def set_weekly_overrides(
    cfg: DbConfig,
    user: str,
    period: str,
    week_monday: date,
    overrides: Dict[str, int],
) -> None:
    wk = week_monday.isoformat()
    with get_conn(cfg) as conn:
        for item, prio in overrides.items():
            conn.execute(
                """
                INSERT INTO item_priority_overrides (user, period, week_monday, item, priority)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user, period, week_monday, item)
                DO UPDATE SET priority=excluded.priority
                """,
                (user, period, wk, item, int(prio)),
            )
        conn.commit()


def clear_weekly_overrides(cfg: DbConfig, user: str, period: str, week_monday: date) -> None:
    with get_conn(cfg) as conn:
        conn.execute(
            "DELETE FROM item_priority_overrides WHERE user=? AND period=? AND week_monday=?",
            (user, period, week_monday.isoformat()),
        )
        conn.commit()


def get_weekly_overrides_map(cfg: DbConfig, user: str, period: str, week_monday: date) -> Dict[str, int]:
    with get_conn(cfg) as conn:
        rows = conn.execute(
            """
            SELECT item, priority
            FROM item_priority_overrides
            WHERE user=? AND period=? AND week_monday=?
            """,
            (user, period, week_monday.isoformat()),
        ).fetchall()
    return {str(item): int(p) for item, p in rows}


def get_top_focus_item_for_day(cfg: DbConfig, user: str, day: date) -> str:
    items = get_items(cfg, "Daily", user=user, for_date=day)
    if not items:
        return "Plan / Review"
    saved = load_entries(cfg, user, day.isoformat(), "Daily")
    for it in items:
        done, _ = saved.get(it, (False, ""))
        if not done:
            return it
    return "Plan / Review"


# -------------------------
# Pomodoro tracking
# -------------------------

def record_work_session(cfg: DbConfig, user: str, day: date, inc: int = 1) -> None:
    day_s = day.isoformat()
    now_utc = int(datetime.now(timezone.utc).timestamp())

    with get_conn(cfg) as conn:
        conn.execute(
            """
            INSERT INTO pomodoro_daily (user, day, work_sessions)
            VALUES (?, ?, ?)
            ON CONFLICT(user, day) DO UPDATE SET
              work_sessions = work_sessions + excluded.work_sessions
            """,
            (user, day_s, int(inc)),
        )
        for _ in range(int(inc)):
            conn.execute(
                "INSERT INTO pomodoro_events (user, ts_utc, day) VALUES (?, ?, ?)",
                (user, now_utc, day_s),
            )
        conn.commit()


def get_pomodoro_sessions_for_day(cfg: DbConfig, user: str, d: date) -> int:
    with get_conn(cfg) as conn:
        row = conn.execute(
            "SELECT work_sessions FROM pomodoro_daily WHERE user=? AND day=?",
            (user, d.isoformat()),
        ).fetchone()
    return int(row[0]) if row else 0


def get_pomodoro_history(cfg: DbConfig, user: str, days: int) -> List[Tuple[date, int]]:
    end = date.today()
    start = end - timedelta(days=days - 1)
    with get_conn(cfg) as conn:
        rows = conn.execute(
            """
            SELECT day, work_sessions
            FROM pomodoro_daily
            WHERE user=? AND day BETWEEN ? AND ?
            ORDER BY day ASC
            """,
            (user, start.isoformat(), end.isoformat()),
        ).fetchall()
    by_day = {datetime.strptime(d, "%Y-%m-%d").date(): int(c) for d, c in rows}
    out: List[Tuple[date, int]] = []
    for i in range(days):
        dd = start + timedelta(days=i)
        out.append((dd, by_day.get(dd, 0)))
    return out


def compute_pomodoro_streak(history: List[Tuple[date, int]], target_per_day: int) -> int:
    if not history:
        return 0
    by_day = {d: c for d, c in history}
    cur = date.today()
    streak = 0
    while by_day.get(cur, 0) >= target_per_day:
        streak += 1
        cur -= timedelta(days=1)
    return streak


def weekly_totals_from_daily(history: List[Tuple[date, int]]) -> List[Tuple[date, int]]:
    buckets: Dict[date, int] = {}
    for d, sessions in history:
        m = monday_of_week(d)
        buckets[m] = buckets.get(m, 0) + int(sessions)
    return sorted(buckets.items(), key=lambda x: x[0])


def get_weekly_sessions(cfg: DbConfig, user: str, week_monday: date) -> int:
    week_sunday = week_monday + timedelta(days=6)
    with get_conn(cfg) as conn:
        row = conn.execute(
            """
            SELECT SUM(work_sessions) AS total
            FROM pomodoro_daily
            WHERE user=? AND day BETWEEN ? AND ?
            """,
            (user, week_monday.isoformat(), week_sunday.isoformat()),
        ).fetchone()
    return int((row[0] or 0) if row else 0)


def get_recent_sessions_count(cfg: DbConfig, user: str, window_minutes: int) -> int:
    cutoff_utc = int(datetime.now(timezone.utc).timestamp()) - int(window_minutes) * 60
    with get_conn(cfg) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM pomodoro_events
            WHERE user=? AND ts_utc >= ?
            """,
            (user, cutoff_utc),
        ).fetchone()
    return int(row[0]) if row else 0


# -------------------------
# Pomodoro timer (UI)
# -------------------------

def init_pomodoro_state() -> None:
    st.session_state.setdefault("pomodoro_running", False)
    st.session_state.setdefault("pomodoro_mode", "work")  # work | break
    st.session_state.setdefault("pomodoro_end_time", None)
    st.session_state.setdefault("pomodoro_work_min", 25)
    st.session_state.setdefault("pomodoro_break_min", 5)


def start_pomodoro(duration_seconds: int) -> None:
    st.session_state.pomodoro_running = True
    st.session_state.pomodoro_end_time = time.time() + duration_seconds


def stop_pomodoro() -> None:
    st.session_state.pomodoro_running = False
    st.session_state.pomodoro_end_time = None


def render_pomodoro(cfg: DbConfig, user: str) -> None:
    init_pomodoro_state()

    st.markdown("### 🍅 Pomodoro")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.pomodoro_work_min = st.number_input(
            "Work (min)", min_value=1, max_value=90, value=int(st.session_state.pomodoro_work_min)
        )
    with c2:
        st.session_state.pomodoro_break_min = st.number_input(
            "Break (min)", min_value=1, max_value=30, value=int(st.session_state.pomodoro_break_min)
        )

    mode: str = st.session_state.pomodoro_mode
    duration = int(st.session_state.pomodoro_work_min * 60) if mode == "work" else int(st.session_state.pomodoro_break_min * 60)

    if st.session_state.pomodoro_running and st.session_state.pomodoro_end_time is not None:
        seconds_left = int(st.session_state.pomodoro_end_time - time.time())

        if seconds_left <= 0:
            if mode == "work":
                record_work_session(cfg, user, date.today(), inc=1)
                st.toast("✅ Work session logged (+1)", icon="🍅")

            st.session_state.pomodoro_mode = "break" if mode == "work" else "work"
            stop_pomodoro()
            st.rerun()
        else:
            mm = seconds_left // 60
            ss = seconds_left % 60
            st.metric(f"{mode.capitalize()} Remaining", f"{mm:02d}:{ss:02d}")
            st.progress(seconds_left / max(1, duration))
            time.sleep(1)
            st.rerun()
    else:
        st.metric(f"{mode.capitalize()} Duration", f"{duration // 60} min")

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Start", use_container_width=True):
            start_pomodoro(duration)
            st.rerun()
    with b2:
        if st.button("Pause", use_container_width=True):
            stop_pomodoro()
            st.rerun()
    with b3:
        if st.button("Reset", use_container_width=True):
            stop_pomodoro()
            st.session_state.pomodoro_mode = "work"
            st.rerun()


# -------------------------
# Focus Mode
# -------------------------

def _focus_index_key(period: str, bucket: str) -> str:
    return f"focus_idx::{period}::{bucket}"


def get_focus_index(period: str, bucket: str, n: int) -> int:
    key = _focus_index_key(period, bucket)
    if key not in st.session_state:
        st.session_state[key] = 0
    return max(0, min(int(st.session_state[key]), max(0, n - 1)))


def set_focus_index(period: str, bucket: str, idx: int, n: int) -> None:
    st.session_state[_focus_index_key(period, bucket)] = max(0, min(int(idx), max(0, n - 1)))


def render_focus_mode(
    cfg: DbConfig,
    user: str,
    period: str,
    bucket: str,
    items: List[str],
    saved: Dict[str, Tuple[bool, str]],
    preview_enabled: bool,
) -> Dict[str, Tuple[bool, str]]:
    n = len(items)
    idx = get_focus_index(period, bucket, n)

    st.markdown(
        """
        <style>
        .block-container {padding-top: 1rem;}
        textarea {font-size: 18px !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    render_pomodoro(cfg, user)
    st.divider()

    def build_current_state() -> Dict[str, Tuple[bool, str]]:
        current: Dict[str, Tuple[bool, str]] = {}
        for it in items:
            dk = f"cb::{period}::{bucket}::{it}"
            nk = f"note::{period}::{bucket}::{it}"
            d0, n0 = saved.get(it, (False, ""))
            done_val = bool(st.session_state.get(dk, d0))
            note_val = str(st.session_state.get(nk, n0))
            current[it] = (done_val, note_val)
        return current

    def auto_save() -> None:
        save_entries(cfg, user, bucket, period, build_current_state())

    nav1, nav2, nav3 = st.columns([0.2, 0.6, 0.2])
    with nav1:
        if st.button("⬅️ Prev", disabled=(idx == 0), use_container_width=True):
            auto_save()
            set_focus_index(period, bucket, idx - 1, n)
            st.rerun()
    with nav3:
        if st.button("Next ➡️", disabled=(idx == n - 1), use_container_width=True):
            auto_save()
            set_focus_index(period, bucket, idx + 1, n)
            st.rerun()
    with nav2:
        st.progress((idx + 1) / max(1, n))
        st.caption(f"Item {idx + 1} / {n}")

    item = items[idx]
    done_default, note_default = saved.get(item, (False, ""))

    done_key = f"cb::{period}::{bucket}::{item}"
    note_key = f"note::{period}::{bucket}::{item}"

    st.session_state.setdefault(done_key, done_default)
    st.session_state.setdefault(note_key, note_default)

    st.markdown(f"## {item}")

    previous_done = bool(st.session_state[done_key])
    done_now = st.checkbox("Completed", value=previous_done, key=done_key)

    note = st.text_area(
        "Markdown Editor",
        value=str(st.session_state[note_key]),
        height=360,
        key=note_key,
    )

    if preview_enabled:
        st.markdown("### Preview")
        render_markdown_note(note)

    if done_now and not previous_done:
        auto_save()
        if idx < n - 1:
            set_focus_index(period, bucket, idx + 1, n)
            st.rerun()

    return build_current_state()


# -------------------------
# Deep Work Score + Dashboards
# -------------------------

def _normalize_weights(weights: List[float]) -> List[float]:
    cleaned = [max(0.0, float(w)) for w in weights]
    s = sum(cleaned)
    if s <= 0:
        return [1.0 / len(cleaned) for _ in cleaned]
    return [w / s for w in cleaned]


def deep_work_score_2(
    daily_completion_pct: float,
    focused_minutes: int,
    target_completion_pct: float,
    target_minutes: int,
    weight_completion: float,
) -> float:
    w_c, w_m = _normalize_weights([weight_completion, 1.0 - weight_completion])
    comp_component = min(1.0, max(0.0, daily_completion_pct / max(1.0, target_completion_pct)))
    min_component = min(1.0, max(0.0, focused_minutes / max(1, target_minutes)))
    return 100.0 * (w_c * comp_component + w_m * min_component)


def deep_work_score_3(
    daily_completion_pct: float,
    focused_minutes: int,
    third_value: float,
    target_completion_pct: float,
    target_minutes: int,
    third_target: float,
    w_daily: float,
    w_minutes: float,
    w_third: float,
) -> float:
    w1, w2, w3 = _normalize_weights([w_daily, w_minutes, w_third])
    comp_component = min(1.0, max(0.0, daily_completion_pct / max(1.0, target_completion_pct)))
    min_component = min(1.0, max(0.0, focused_minutes / max(1, target_minutes)))
    third_component = min(1.0, max(0.0, third_value / max(1.0, third_target)))
    return 100.0 * (w1 * comp_component + w2 * min_component + w3 * third_component)


def remaining_items_to_target(done: int, total: int, target_pct: float) -> int:
    if total <= 0:
        return 0
    target = (target_pct / 100.0) * total
    target_done = int(target)
    if target > target_done:
        target_done += 1
    return max(0, target_done - done)


def eta_from_pace(remaining_minutes: int, pace_minutes_per_hour: float, now: datetime) -> str:
    if remaining_minutes <= 0:
        return "Target reached"
    if pace_minutes_per_hour <= 0.0:
        return ""
    hours_needed = remaining_minutes / pace_minutes_per_hour
    eta = now + timedelta(hours=hours_needed)
    return eta.strftime("%I:%M %p").lstrip("0")


def render_today_dashboard(
    cfg: DbConfig,
    user: str,
    *,
    session_minutes: int,
    target_completion: float,
    target_minutes: int,
    use_third: bool,
    third_type: str,
    third_target: float,
    w_daily: float,
    w_minutes: float,
    w_third: float,
    eta_mode: str,
    recent_window_min: int,
) -> None:
    st.subheader("📍 Today Dashboard")

    today = date.today()
    now = datetime.now()

    daily_pct, done, total = get_daily_completion_for_day(cfg, user, today)
    sessions_today = get_pomodoro_sessions_for_day(cfg, user, today)
    focused_minutes_today = sessions_today * int(session_minutes)

    week_monday = monday_of_week(today)
    week_completion_pct = get_weekly_completion_pct_for_week(cfg, user, week_monday)
    week_sessions = get_weekly_sessions(cfg, user, week_monday)
    week_minutes = week_sessions * int(session_minutes)

    if use_third:
        third_value = week_completion_pct if third_type == "Weekly completion %" else float(week_minutes)
        score = deep_work_score_3(
            daily_completion_pct=daily_pct,
            focused_minutes=focused_minutes_today,
            third_value=float(third_value),
            target_completion_pct=float(target_completion),
            target_minutes=int(target_minutes),
            third_target=float(third_target),
            w_daily=float(w_daily),
            w_minutes=float(w_minutes),
            w_third=float(w_third),
        )
    else:
        score = deep_work_score_2(
            daily_completion_pct=daily_pct,
            focused_minutes=focused_minutes_today,
            target_completion_pct=float(target_completion),
            target_minutes=int(target_minutes),
            weight_completion=float(w_daily),
        )

    remaining_minutes = max(0, int(target_minutes) - focused_minutes_today)
    remaining_items = remaining_items_to_target(done, total, float(target_completion))

    if eta_mode == "Recent pace":
        recent_sessions = get_recent_sessions_count(cfg, user, window_minutes=int(recent_window_min))
        recent_focused = recent_sessions * int(session_minutes)
        pace_mph = (recent_focused / (max(1, int(recent_window_min)) / 60.0)) if recent_focused > 0 else 0.0
        pace_label = f"{pace_mph:.1f} (last {recent_window_min}m)"
    else:
        start_of_day = datetime(now.year, now.month, now.day, 0, 0, 0)
        hours_elapsed = max(0.01, (now - start_of_day).total_seconds() / 3600.0)
        pace_mph = focused_minutes_today / hours_elapsed if focused_minutes_today > 0 else 0.0
        pace_label = f"{pace_mph:.1f} (today avg)"

    eta = eta_from_pace(remaining_minutes, pace_mph, now)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Deep Work Score", f"{score:.1f}/100")
    c2.metric("Focused minutes (today)", focused_minutes_today)
    c3.metric("Minutes to target", remaining_minutes)
    c4.metric("Items to target", remaining_items)
    c5.metric("Pace (min/hr)", pace_label)

    if eta:
        st.caption(f"ETA to minutes target: **{eta}**")
    else:
        st.caption("ETA: not enough data yet (log at least 1 work session).")

    st.caption(
        f"Daily completion: {daily_pct:.1f}% ({done}/{total}) • Pomodoros today: {sessions_today} • "
        f"Week: {week_monday.isoformat()} — completion {week_completion_pct:.1f}% • focused minutes {week_minutes}"
    )


def render_weekly_dashboard(
    cfg: DbConfig,
    user: str,
    *,
    session_minutes: int,
    target_weekly_minutes: int,
    target_weekly_completion: float,
    weekly_score_mode: str,
) -> None:
    st.subheader("🗓️ Weekly Dashboard")

    today = date.today()
    wk = monday_of_week(today)

    week_completion = get_weekly_completion_pct_for_week(cfg, user, wk)
    week_sessions = get_weekly_sessions(cfg, user, wk)
    week_minutes = week_sessions * int(session_minutes)

    weekly_minutes_remaining = max(0, int(target_weekly_minutes) - week_minutes)
    weekly_completion_gap = max(0.0, float(target_weekly_completion) - week_completion)

    if weekly_score_mode == "Minutes only":
        weekly_score = 100.0 * min(1.0, week_minutes / max(1, int(target_weekly_minutes)))
    elif weekly_score_mode == "Completion only":
        weekly_score = 100.0 * min(1.0, week_completion / max(1.0, float(target_weekly_completion)))
    else:
        r1 = min(1.0, week_minutes / max(1, int(target_weekly_minutes)))
        r2 = min(1.0, week_completion / max(1.0, float(target_weekly_completion)))
        weekly_score = 100.0 * ((r1 + r2) / 2.0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Weekly score", f"{weekly_score:.1f}/100")
    c2.metric("Weekly focused minutes", week_minutes)
    c3.metric("Minutes remaining", weekly_minutes_remaining)
    c4.metric("Completion gap", f"{weekly_completion_gap:.1f}%")

    st.caption(f"Week starting Monday: {wk.isoformat()} • Mode: {weekly_score_mode}")


# -------------------------
# Weekly Plan (allocation + AM/PM split)
# -------------------------

def allocate_minutes_with_caps(
    total_minutes: int,
    day_caps: List[int],
    weights: List[float],
) -> Tuple[List[int], int]:
    n = len(day_caps)
    if n == 0:
        return [], total_minutes

    weights = [max(0.0, float(w)) for w in weights]
    wsum = sum(weights) or 1.0
    weights = [w / wsum for w in weights]

    alloc = [0] * n
    remaining = int(total_minutes)
    desired = [weights[i] * total_minutes for i in range(n)]

    while remaining > 0:
        candidates = [i for i in range(n) if alloc[i] < day_caps[i]]
        if not candidates:
            break
        best_i = max(candidates, key=lambda i: (desired[i] - alloc[i], day_caps[i] - alloc[i]))
        alloc[best_i] += 1
        remaining -= 1

    return alloc, remaining


def weekly_plan_minutes(
    weekly_target_minutes: int,
    minutes_done_so_far: int,
    days: List[date],
    strategy: str,
    day_caps: Optional[List[int]] = None,
) -> Tuple[List[Tuple[str, int]], int]:
    remaining = max(0, int(weekly_target_minutes) - int(minutes_done_so_far))
    n = len(days)
    if n == 0:
        return [], remaining
    if remaining == 0:
        return [(d.strftime("%a %m/%d"), 0) for d in days], 0

    if day_caps is None:
        day_caps = [10**9] * n
    else:
        day_caps = [max(0, int(c)) for c in day_caps]

    if sum(day_caps) < remaining:
        alloc = day_caps[:]
        return [(days[i].strftime("%a %m/%d"), int(alloc[i])) for i in range(n)], remaining - sum(alloc)

    if strategy == "Even":
        weights = [1.0] * n
    elif strategy == "Front-load":
        weights = [float(n - i) for i in range(n)]
    else:
        weights = [float(i + 1) for i in range(n)]

    alloc, unallocated = allocate_minutes_with_caps(remaining, day_caps, weights)
    return [(days[i].strftime("%a %m/%d"), int(alloc[i])) for i in range(n)], int(unallocated)


def split_minutes_am_pm_snapped(
    minutes_needed: int,
    cap_am: int | None,
    cap_pm: int | None,
    bias: str,
    session_minutes: int,
    allow_one_partial: bool,
) -> Tuple[int, int, int, int]:
    """
    Returns (am_minutes, pm_minutes, unallocated_minutes, extra_pomodoros_needed).
    - Whole pomodoros snapped; optional single partial remainder if allowed.
    """
    need = max(0, int(minutes_needed))
    s = max(1, int(session_minutes))

    cap_am = 10**9 if cap_am is None else max(0, int(cap_am))
    cap_pm = 10**9 if cap_pm is None else max(0, int(cap_pm))

    if need == 0:
        return 0, 0, 0, 0

    if bias == "AM-heavy":
        target_am = int(round(need * 0.7))
    elif bias == "PM-heavy":
        target_am = int(round(need * 0.3))
    else:
        target_am = need // 2

    cap_am_snapped = (cap_am // s) * s
    cap_pm_snapped = (cap_pm // s) * s
    target_am_snapped = (target_am // s) * s

    am = min(target_am_snapped, cap_am_snapped)
    remaining = need - am

    pm = min((remaining // s) * s, cap_pm_snapped)
    remaining -= pm

    if remaining >= s and am < cap_am_snapped:
        add = min((remaining // s) * s, cap_am_snapped - am)
        am += add
        remaining -= add

    if remaining >= s and pm < cap_pm_snapped:
        add = min((remaining // s) * s, cap_pm_snapped - pm)
        pm += add
        remaining -= add

    if allow_one_partial and 0 < remaining < s:
        am_space = cap_am - am
        pm_space = cap_pm - pm
        prefer_am = bias in ("AM-heavy", "Balanced")
        first = "am" if prefer_am else "pm"
        second = "pm" if prefer_am else "am"

        def try_place(which: str) -> bool:
            nonlocal am, pm, remaining
            if which == "am" and am_space >= remaining:
                am += remaining
                remaining = 0
                return True
            if which == "pm" and pm_space >= remaining:
                pm += remaining
                remaining = 0
                return True
            return False

        if not try_place(first):
            try_place(second)

    unallocated = remaining
    extra_pomodoros_needed = (unallocated + s - 1) // s if unallocated > 0 else 0
    return int(am), int(pm), int(unallocated), int(extra_pomodoros_needed)


def _fmt_time(dt: datetime) -> str:
    return dt.strftime("%I:%M %p").lstrip("0").lower()


def _format_range(start_dt: datetime, minutes: int) -> str:
    end_dt = start_dt + timedelta(minutes=int(minutes))
    return f"{_fmt_time(start_dt)}–{_fmt_time(end_dt)}"


def _build_session_ranges_for_block(
    day: date,
    block_start: dtime,
    total_minutes: int,
    session_minutes: int,
    gap_minutes: int,
    allow_partial: bool,
) -> List[str]:
    total_minutes = max(0, int(total_minutes))
    s = max(1, int(session_minutes))
    gap = max(0, int(gap_minutes))
    if total_minutes == 0:
        return []

    sessions: List[int] = []
    full = total_minutes // s
    rem = total_minutes % s
    sessions.extend([s] * full)
    if allow_partial and rem > 0:
        sessions.append(rem)

    start_dt = datetime.combine(day, block_start)
    out: List[str] = []
    for mins in sessions:
        out.append(_format_range(start_dt, mins))
        start_dt = start_dt + timedelta(minutes=mins + gap)
    return out


# -------------------------
# Google Calendar links (Android-friendly)
# -------------------------

def _gcal_dt_local(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M%S")


def build_google_calendar_link(
    *,
    title: str,
    start_local: datetime,
    end_local: datetime,
    details: str = "",
    location: str = "",
    tz: str = NY_TZ,
) -> str:
    params = {
        "action": "TEMPLATE",
        "text": title,
        "dates": f"{_gcal_dt_local(start_local)}/{_gcal_dt_local(end_local)}",
        "ctz": tz,
    }
    if details:
        params["details"] = details
    if location:
        params["location"] = location
    return "https://calendar.google.com/calendar/render?" + urlencode(params, quote_via=quote)


# -------------------------
# Admin UI (sidebar)
# -------------------------

def sidebar_admin(cfg: DbConfig, user: str) -> None:
    st.sidebar.header("⚙️ Admin")

    # Seed items
    with st.sidebar.expander("Seed / Replace items", expanded=False):
        p = st.selectbox("Period", PERIODS, key="seed_period")
        st.caption("Paste one item per line. This replaces the list for that period.")
        raw = st.text_area("Items", height=180, key="seed_items_text")
        if st.button("Replace items for period", use_container_width=True, key="seed_btn"):
            items = [line.strip() for line in raw.splitlines() if line.strip()]
            seed_items(cfg, p, items)
            st.success("Items updated.")
            st.rerun()

    # Base priorities
    with st.sidebar.expander("Edit base priorities", expanded=False):
        p = st.selectbox("Period (priorities)", PERIODS, key="prio_period")
        rows = get_items_for_edit(cfg, p)
        updates: List[Tuple[int, int]] = []
        for pos, text, pr in rows:
            new_pr = st.number_input(
                f"{pos}. {text}",
                min_value=-10,
                max_value=10,
                value=int(pr),
                step=1,
                key=f"baseprio::{p}::{pos}",
            )
            if int(new_pr) != int(pr):
                updates.append((pos, int(new_pr)))
        if st.button("Save base priorities", use_container_width=True, key="save_base_prio") and updates:
            update_base_priorities(cfg, p, updates)
            st.success("Base priorities saved.")
            st.rerun()

    # Weekly overrides
    with st.sidebar.expander("Weekly priority overrides", expanded=False):
        p = st.selectbox("Period (weekly overrides)", PERIODS, key="wk_prio_period")
        wk = monday_of_week(date.today())
        st.caption(f"Week (Mon start): {wk.isoformat()}")
        base_rows = get_items_for_edit(cfg, p)  # position order; we display base + override
        ov_map = get_weekly_overrides_map(cfg, user, p, wk)

        overrides_to_save: Dict[str, int] = {}
        for pos, text, base_pr in base_rows:
            effective = ov_map.get(text, int(base_pr))
            new_pr = st.number_input(
                f"{pos}. {text}",
                min_value=-10,
                max_value=10,
                value=int(effective),
                step=1,
                key=f"wkprio::{p}::{wk.isoformat()}::{pos}",
            )
            if int(new_pr) != int(base_pr):
                overrides_to_save[text] = int(new_pr)

        c1, c2 = st.sidebar.columns(2)
        with c1:
            if st.button("Save overrides", use_container_width=True, key="wk_save"):
                set_weekly_overrides(cfg, user, p, wk, overrides_to_save)
                st.success("Weekly overrides saved.")
                st.rerun()
        with c2:
            if st.button("Clear overrides", use_container_width=True, key="wk_clear"):
                clear_weekly_overrides(cfg, user, p, wk)
                st.success("Weekly overrides cleared.")
                st.rerun()


# -------------------------
# Focus Stats UI (charts + dashboards + weekly plan)
# -------------------------

def render_weekly_plan(cfg: DbConfig, user: str, *, session_minutes: int, target_weekly_minutes: int) -> None:
    st.subheader("🧭 Weekly Plan (Minutes)")

    today = date.today()
    wk = monday_of_week(today)

    week_sessions = get_weekly_sessions(cfg, user, wk)
    week_minutes_done = week_sessions * int(session_minutes)
    remaining = max(0, int(target_weekly_minutes) - week_minutes_done)

    c1, c2, c3 = st.columns(3)
    c1.metric("Weekly focused minutes (so far)", week_minutes_done)
    c2.metric("Weekly minutes target", int(target_weekly_minutes))
    c3.metric("Minutes remaining", remaining)

    weekdays_only = st.toggle("Mon–Fri only (skip weekends)", value=False, key="wkplan_weekdays_only")
    days = remaining_days_in_week(today, weekdays_only=weekdays_only)
    if not days:
        st.info("No remaining days in this week window.")
        return

    strategy = st.selectbox("Split strategy", ["Even", "Front-load", "Back-load"], index=0, key="wkplan_strategy")

    use_availability = st.toggle("Use availability (caps)", value=True, key="wkplan_use_caps")
    two_blocks = False
    if use_availability:
        two_blocks = st.toggle("Two blocks per day (AM + PM)", value=True, key="wkplan_two_blocks")

    caps: Optional[List[int]] = None
    caps_am: Optional[List[int]] = None
    caps_pm: Optional[List[int]] = None

    if use_availability:
        st.caption("Set your maximum available focus minutes per remaining day.")
        caps = []
        if two_blocks:
            st.caption("AM + PM caps are summed into a daily total cap.")
            caps_am, caps_pm = [], []

        cols = st.columns(min(4, len(days)))
        for i, d in enumerate(days):
            with cols[i % len(cols)]:
                if two_blocks:
                    am = st.number_input(
                        f"{d.strftime('%a %m/%d')} AM cap",
                        min_value=0,
                        max_value=600,
                        value=60,
                        step=10,
                        key=f"cap_am::{wk.isoformat()}::{d.isoformat()}",
                    )
                    pm = st.number_input(
                        f"{d.strftime('%a %m/%d')} PM cap",
                        min_value=0,
                        max_value=600,
                        value=60,
                        step=10,
                        key=f"cap_pm::{wk.isoformat()}::{d.isoformat()}",
                    )
                    caps_am.append(int(am))
                    caps_pm.append(int(pm))
                    caps.append(int(am) + int(pm))
                    st.caption(f"Total cap: {int(am)+int(pm)} min")
                else:
                    cap = st.number_input(
                        f"{d.strftime('%a %m/%d')} cap",
                        min_value=0,
                        max_value=600,
                        value=90,
                        step=10,
                        key=f"cap::{wk.isoformat()}::{d.isoformat()}",
                    )
                    caps.append(int(cap))

    plan, unallocated = weekly_plan_minutes(
        weekly_target_minutes=int(target_weekly_minutes),
        minutes_done_so_far=week_minutes_done,
        days=days,
        strategy=strategy,
        day_caps=caps,
    )

    am_pm_bias = "Balanced"
    allow_one_partial = False
    if use_availability and two_blocks:
        am_pm_bias = st.selectbox("Suggested split bias", ["AM-heavy", "Balanced", "PM-heavy"], index=1, key="wkplan_bias")
        allow_one_partial = st.toggle("Allow one partial session (last block can be shorter)", value=False, key="wkplan_partial")

    st.write("**Required minutes per remaining day (including today):**")

    rows: List[Dict] = []
    for i, (day_label, minutes_needed) in enumerate(plan):
        cap_total = caps[i] if caps is not None else None
        am_cap = caps_am[i] if caps_am is not None else None
        pm_cap = caps_pm[i] if caps_pm is not None else None

        am_suggested = None
        pm_suggested = None
        day_shortfall = 0
        extra_pomos = 0

        if use_availability and two_blocks and caps_am is not None and caps_pm is not None:
            am_suggested, pm_suggested, day_shortfall, extra_pomos = split_minutes_am_pm_snapped(
                minutes_needed=int(minutes_needed),
                cap_am=int(caps_am[i]),
                cap_pm=int(caps_pm[i]),
                bias=str(am_pm_bias),
                session_minutes=int(session_minutes),
                allow_one_partial=bool(allow_one_partial),
            )

        rows.append(
            {
                "day": day_label,
                "minutes_needed": int(minutes_needed),
                "min_pomodoros_needed": ceil_div(int(minutes_needed), int(session_minutes)),
                "cap_total": cap_total,
                "cap_am": am_cap,
                "cap_pm": pm_cap,
                "suggest_am": am_suggested,
                "suggest_pm": pm_suggested,
                "day_shortfall": int(day_shortfall),
                "extra_pomodoros_needed": int(extra_pomos),
            }
        )

    st.dataframe(rows, use_container_width=True)

    if remaining == 0:
        st.success("Weekly target reached. Anything extra is bonus.")
    elif unallocated > 0:
        st.error(
            f"Your availability caps cannot fit the weekly target. Shortfall: **{unallocated} minutes**. "
            f"Increase caps or lower weekly target."
        )
    else:
        avg = remaining / max(1, len(plan))
        st.caption(
            f"Plan uses **{strategy}** split. Average needed per remaining day ≈ **{avg:.0f} min/day** "
            f"(min ≈ **{ceil_div(int(avg), int(session_minutes))} pomodoros/day**)."
        )

    # Copyable plan text
    if use_availability and two_blocks:
        st.subheader("📋 Copy Weekly Plan")

        lines = []
        lines.append(f"Weekly Plan (week of {wk.isoformat()})")
        lines.append(f"Session length: {int(session_minutes)} min | Strategy: {strategy} | Bias: {am_pm_bias}")
        lines.append("Partial session: allowed (max 1/day)" if allow_one_partial else "Partial session: not allowed")
        lines.append("")
        for r in rows:
            day = r["day"]
            need = r["minutes_needed"]
            am = r.get("suggest_am")
            pm = r.get("suggest_pm")
            shortfall = r.get("day_shortfall", 0)
            if am is None and pm is None:
                lines.append(f"- {day}: {need} min")
            else:
                am_txt = f"AM {am}m" if am else "AM —"
                pm_txt = f"PM {pm}m" if pm else "PM —"
                sf_txt = f" (shortfall {shortfall}m)" if shortfall else ""
                lines.append(f"- {day}: {am_txt}, {pm_txt} | total {need}m{sf_txt}")

        st.text_area("Copy/paste into notes", value="\n".join(lines), height=220, key="wkplan_copy")

        # Calendar-friendly schedule text
        st.subheader("📅 Calendar-Friendly Plan")

        colT1, colT2, colT3 = st.columns(3)
        with colT1:
            am_start = st.time_input("AM start time", value=dtime(9, 0), key="wkplan_am_start")
        with colT2:
            pm_start = st.time_input("PM start time", value=dtime(14, 0), key="wkplan_pm_start")
        with colT3:
            gap_minutes = st.number_input("Gap between sessions (min)", min_value=0, max_value=60, value=5, step=5, key="wkplan_gap")

        cal_lines = []
        cal_lines.append(f"Weekly Plan (calendar format) — week of {wk.isoformat()}")
        cal_lines.append(f"Session: {int(session_minutes)}m | Gap: {int(gap_minutes)}m | Bias: {am_pm_bias}")
        cal_lines.append("")

        for i, d in enumerate(days):
            day_label = d.strftime("%a %m/%d")
            am_mins = rows[i].get("suggest_am") or 0
            pm_mins = rows[i].get("suggest_pm") or 0

            am_ranges = _build_session_ranges_for_block(
                day=d,
                block_start=am_start,
                total_minutes=int(am_mins),
                session_minutes=int(session_minutes),
                gap_minutes=int(gap_minutes),
                allow_partial=bool(allow_one_partial),
            )
            pm_ranges = _build_session_ranges_for_block(
                day=d,
                block_start=pm_start,
                total_minutes=int(pm_mins),
                session_minutes=int(session_minutes),
                gap_minutes=int(gap_minutes),
                allow_partial=bool(allow_one_partial),
            )

            parts = []
            if am_ranges:
                parts.append(", ".join(am_ranges))
            if pm_ranges:
                parts.append(", ".join(pm_ranges))

            cal_lines.append(f"{day_label}: " + (" ; ".join(parts) if parts else "—"))

        st.text_area("Copy/paste into calendar notes", value="\n".join(cal_lines), height=260, key="wkplan_cal_copy")

    # Google Calendar daily big-block links (Android)
    st.subheader("📱 Google Calendar Links (Daily Big Block) — Android")

    c1, c2 = st.columns(2)
    with c1:
        daily_start = st.time_input("Start time (each day)", value=dtime(9, 0), key="gcal_big_start")
    with c2:
        title_prefix = st.text_input("Title prefix", value="Deep Work Block", key="gcal_big_title")

    details = st.text_area(
        "Details (optional)",
        value="Planned deep work block from Weekly Plan. One outcome. No distractions.",
        height=80,
        key="gcal_big_details",
    ).strip()

    include_focus_item = st.toggle("Include top focus item in title", value=True, key="gcal_include_focus")

    st.caption("Tap a button on Android → opens Google Calendar with the event prefilled (New York time).")

    for i, d in enumerate(days):
        minutes_needed = int(rows[i].get("minutes_needed") or 0)
        label = d.strftime("%a %m/%d")

        if minutes_needed <= 0:
            st.link_button(f"{label}: No block needed", "https://calendar.google.com", use_container_width=True)
            continue

        focus_item = get_top_focus_item_for_day(cfg, user, d) if include_focus_item else ""
        focus_prefix = f" — {focus_item}" if focus_item else ""

        start_dt = datetime.combine(d, daily_start)
        end_dt = start_dt + timedelta(minutes=minutes_needed)

        url = build_google_calendar_link(
            title=f"{title_prefix}{focus_prefix} — {label} ({minutes_needed}m)",
            start_local=start_dt,
            end_local=end_dt,
            details=(details + (f"\n\nTop focus item:\n- {focus_item}" if focus_item else "")).strip(),
            tz=NY_TZ,
        )
        st.link_button(f"Add {label} ({minutes_needed} min) to Google Calendar", url, use_container_width=True)


def render_focus_stats(cfg: DbConfig, user: str) -> None:
    import matplotlib.pyplot as plt

    st.subheader("🍅 Focus Stats")

    days = st.slider("Window (days)", 7, 180, 30, 1, key="stats_days")
    daily_pomo_target = st.number_input("Daily target (work sessions)", min_value=1, max_value=20, value=4, step=1, key="stats_target_pomos")
    session_minutes = st.number_input("Minutes per work session (for stats)", min_value=5, max_value=90, value=25, step=5, key="stats_session_min")

    st.divider()
    st.write("### Deep Work Score settings")

    use_third = st.toggle("Enable 3rd component", value=True, key="score_use_third")
    third_type = "Weekly completion %"
    third_target = 85.0

    eta_mode = st.selectbox("ETA pace mode", ["Today average", "Recent pace"], key="eta_mode")
    recent_window_min = 60
    if eta_mode == "Recent pace":
        recent_window_min = st.slider("Recent pace window (minutes)", 15, 180, 60, 5, key="eta_window")

    cA, cB, cC = st.columns(3)
    with cA:
        target_completion = st.slider("Target daily completion (%)", 10, 100, 80, 5, key="score_target_completion")
    with cB:
        target_minutes = st.number_input("Target focused minutes/day", min_value=10, max_value=600, value=100, step=10, key="score_target_minutes")
    with cC:
        if use_third:
            third_type = st.selectbox("3rd component type", ["Weekly completion %", "Weekly focused minutes"], key="score_third_type")
            if third_type == "Weekly completion %":
                third_target = float(st.slider("Target weekly completion (%)", 10, 100, 85, 5, key="score_third_target_pct"))
            else:
                third_target = float(st.number_input("Target weekly focused minutes", min_value=10, max_value=3000, value=300, step=10, key="score_third_target_min"))

    st.write("### Weights")
    if use_third:
        w1, w2, w3 = st.columns(3)
        with w1:
            w_daily = st.slider("Weight: Daily completion", 0.0, 1.0, 0.45, 0.05, key="w_daily")
        with w2:
            w_minutes = st.slider("Weight: Focus minutes", 0.0, 1.0, 0.35, 0.05, key="w_minutes")
        with w3:
            w_third = st.slider("Weight: 3rd component", 0.0, 1.0, 0.20, 0.05, key="w_third")
    else:
        w_daily = st.slider("Weight: Daily completion", 0.0, 1.0, 0.60, 0.05, key="w_daily_2")
        w_minutes = 1.0 - w_daily
        w_third = 0.0

    st.divider()

    render_today_dashboard(
        cfg,
        user,
        session_minutes=int(session_minutes),
        target_completion=float(target_completion),
        target_minutes=int(target_minutes),
        use_third=bool(use_third),
        third_type=str(third_type),
        third_target=float(third_target),
        w_daily=float(w_daily),
        w_minutes=float(w_minutes),
        w_third=float(w_third),
        eta_mode=("Recent pace" if eta_mode == "Recent pace" else "Today average"),
        recent_window_min=int(recent_window_min),
    )

    st.divider()

    weekly_mode = st.selectbox("Weekly dashboard score mode", ["Balanced", "Minutes only", "Completion only"], key="weekly_mode")
    weekly_target_minutes = st.number_input("Target weekly focused minutes", min_value=10, max_value=6000, value=300, step=10, key="weekly_target_minutes")
    weekly_target_completion = st.slider("Target weekly completion (%)", 10, 100, 85, 5, key="weekly_target_completion")

    render_weekly_dashboard(
        cfg,
        user,
        session_minutes=int(session_minutes),
        target_weekly_minutes=int(weekly_target_minutes),
        target_weekly_completion=float(weekly_target_completion),
        weekly_score_mode=str(weekly_mode),
    )

    st.divider()
    render_weekly_plan(cfg, user, session_minutes=int(session_minutes), target_weekly_minutes=int(weekly_target_minutes))

    st.divider()

    pomo_hist = get_pomodoro_history(cfg, user, days=int(days))
    completion_hist = get_daily_completion_history(cfg, user, days=int(days))
    completion_by_day = {d: (pct, done, total) for d, pct, done, total in completion_hist}

    rows: List[Tuple[date, int, int, float, float]] = []
    for d, sessions in pomo_hist:
        pct, _, _ = completion_by_day.get(d, (0.0, 0, 0))
        focused_min = int(sessions) * int(session_minutes)

        if use_third:
            wk = monday_of_week(d)
            if third_type == "Weekly completion %":
                third_value = get_weekly_completion_pct_for_week(cfg, user, wk)
            else:
                third_value = float(get_weekly_sessions(cfg, user, wk) * int(session_minutes))

            score = deep_work_score_3(
                daily_completion_pct=pct,
                focused_minutes=focused_min,
                third_value=float(third_value),
                target_completion_pct=float(target_completion),
                target_minutes=int(target_minutes),
                third_target=float(third_target),
                w_daily=float(w_daily),
                w_minutes=float(w_minutes),
                w_third=float(w_third),
            )
        else:
            score = deep_work_score_2(
                daily_completion_pct=pct,
                focused_minutes=focused_min,
                target_completion_pct=float(target_completion),
                target_minutes=int(target_minutes),
                weight_completion=float(w_daily),
            )

        rows.append((d, int(sessions), focused_min, pct, score))

    total_sessions = sum(s for _, s in pomo_hist)
    today_sessions = pomo_hist[-1][1] if pomo_hist else 0
    streak = compute_pomodoro_streak(pomo_hist, target_per_day=int(daily_pomo_target))
    total_minutes = sum(m for _, _, m, _, _ in rows)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Today work sessions", today_sessions)
    c2.metric(f"Pomodoro streak (≥ {int(daily_pomo_target)}/day)", streak)
    c3.metric(f"Total sessions (last {days}d)", total_sessions)
    c4.metric(f"Total focused minutes (last {days}d)", total_minutes)

    xs = [d for d, *_ in rows]
    ys_sessions = [s for _, s, *_ in rows]
    ys_score = [sc for *_, sc in rows]

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.bar(xs, ys_sessions)
    ax1.set_title(f"Work Sessions per Day (last {days} days)")
    ax1.set_ylabel("Sessions")
    ax1.set_xlabel("Date")
    fig1.autofmt_xdate()
    st.pyplot(fig1)

    weekly_sessions = weekly_totals_from_daily([(d, s) for d, s, *_ in rows])
    weekly_minutes = [(wk, total * int(session_minutes)) for wk, total in weekly_sessions]

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.bar([wk for wk, _ in weekly_sessions], [v for _, v in weekly_sessions])
    ax2.set_title("Weekly Total Work Sessions (Mon-start weeks)")
    ax2.set_ylabel("Sessions / week")
    ax2.set_xlabel("Week (Monday)")
    fig2.autofmt_xdate()
    st.pyplot(fig2)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.bar([wk for wk, _ in weekly_minutes], [v for _, v in weekly_minutes])
    ax3.set_title("Weekly Total Focused Minutes (Mon-start weeks)")
    ax3.set_ylabel("Minutes / week")
    ax3.set_xlabel("Week (Monday)")
    fig3.autofmt_xdate()
    st.pyplot(fig3)

    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.plot(xs, ys_score, marker="o")
    ax4.set_title("Deep Work Score (0–100)")
    ax4.set_ylabel("Score")
    ax4.set_xlabel("Date")
    ax4.set_ylim(0, 100)
    fig4.autofmt_xdate()
    st.pyplot(fig4)

    st.write("**Daily detail**")
    table = [
        {
            "day": d.isoformat(),
            "sessions": s,
            "focused_minutes": fm,
            "daily_completion_%": round(pct, 1),
            "deep_work_score": round(sc, 1),
        }
        for d, s, fm, pct, sc in rows[::-1]
    ]
    st.dataframe(table, use_container_width=True)


# -------------------------
# Main UI
# -------------------------

def main() -> None:
    st.set_page_config(page_title="CEO Execution Tracker", layout="wide")
    st.title("📈 CEO Execution Tracker")

    cfg = DbConfig(database_url=_get_database_url())
    init_db(cfg)
    user = SINGLE_USER_ID

    sidebar_admin(cfg, user)

    tab_track, tab_notes, tab_stats = st.tabs(["✅ Track", "📝 Notes", "🍅 Focus Stats"])

    with tab_track:
        period = st.selectbox("Period", PERIODS, key="track_period")
        selected_date = st.date_input("Select date", date.today(), key="track_date")
        bkey = bucket_key(period, selected_date)

        items = get_items(cfg, period, user=user, for_date=selected_date)
        edit_mode = st.toggle("✏️ Edit items (this period)", value=False, key="main_edit_items")

        if edit_mode:
            edited = st.text_area(
                "Edit items (one per line). Saving replaces the whole list for this period.",
                value="\n".join(items),
                height=220,
                key="main_edit_items_text",
            )
            if st.button("Save item list", use_container_width=True, key="main_edit_items_save"):
                new_items = [x.strip() for x in edited.splitlines() if x.strip()]
                seed_items(cfg, period, new_items)
                st.success("Items updated.")
                st.rerun()
        if not items:
            st.warning("No items yet. Open the sidebar → 'Seed / Replace items' and add your list.")
            st.stop()

        saved = load_entries(cfg, user, bkey, period)

        focus_mode = st.toggle("🎯 Focus Single Item Mode", value=True, key="focus_mode_toggle")
        preview_enabled = st.toggle("Enable Live Markdown Preview", value=True, key="preview_toggle")

        if focus_mode:
            current = render_focus_mode(cfg, user, period, bkey, items, saved, preview_enabled)
            if st.button("Save (manual)", type="secondary", key="manual_save_focus"):
                save_entries(cfg, user, bkey, period, current)
                st.success("Saved.")
                st.rerun()
        else:
            current: Dict[str, Tuple[bool, str]] = {}
            for item in items:
                done_default, note_default = saved.get(item, (False, ""))
                done = st.checkbox(item, value=done_default, key=f"cb::{period}::{bkey}::{item}")
                note = st.text_area(
                    f"Note (Markdown) — {item}",
                    value=note_default,
                    height=90,
                    key=f"note::{period}::{bkey}::{item}",
                )
                current[item] = (done, note)

            done_count = sum(1 for v in current.values() if v[0])
            st.metric("Completion %", f"{completion_pct(done_count, len(items)):.1f}%")

            if st.button("Save", type="primary", key="save_standard"):
                save_entries(cfg, user, bkey, period, current)
                st.success("Saved.")
                st.rerun()

    with tab_notes:
        st.subheader("📝 Notes Viewer (Markdown)")
        period_filter = st.selectbox("Filter by period", PERIODS, key="notes_period")
        search_query = st.text_input("Search (bucket/item/note)", key="notes_search").strip().lower()

        rows = fetch_notes(cfg, user, period_filter)
        shown = 0
        for bucket, item, note in rows:
            hay = f"{bucket}\n{item}\n{note}".lower()
            if search_query and search_query not in hay:
                continue
            shown += 1
            anchor = bucket_to_anchor_date(period_filter, bucket).isoformat()
            st.markdown(f"### {bucket} — {item}")
            st.caption(f"Anchor date: {anchor}")
            render_markdown_note(note)
            st.divider()

        if shown == 0:
            st.info("No notes found.")

    with tab_stats:
        render_focus_stats(cfg, user)


if __name__ == "__main__":
    main()
