from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from prediction_data import (
    as_export_csv,
    available_teams,
    confidence_label,
    data_as_of,
    filter_predictions,
    load_predictions,
)

PROJECT_ROOT = Path(__file__).resolve().parent
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "2027_upcoming_predictions.csv"
SCHEDULE_PATH = PROJECT_ROOT / "data" / "schedule_2027_upcoming.csv"


st.set_page_config(
    page_title="College Basketball Win Projections",
    page_icon="🏀",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .eyebrow {
            color: #d97706;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        .hero-title {
            font-size: clamp(2rem, 5vw, 4.25rem);
            font-weight: 800;
            letter-spacing: -0.05em;
            line-height: 1.02;
            margin-bottom: 0.5rem;
        }
        .hero-copy {
            color: #64748b;
            font-size: 1.05rem;
            max-width: 720px;
            margin-bottom: 1.5rem;
        }
        .matchup-card {
            background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.07);
            padding: 1.1rem 1.15rem;
            min-height: 220px;
        }
        .matchup-card-date {
            color: #64748b;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
        }
        .matchup-card-title {
            font-size: 1.1rem;
            font-weight: 750;
            margin: 0.4rem 0 0.65rem;
        }
        .winner {
            color: #0f766e;
            font-size: 1.45rem;
            font-weight: 800;
        }
        .card-detail {
            color: #475569;
            font-size: 0.88rem;
        }
        .neutral-badge {
            background: #fff7ed;
            border-radius: 999px;
            color: #c2410c;
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.2rem 0.55rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_predictions(
    path: Path,
    schedule_path: Path,
    predictions_modified_at: float,
    schedule_modified_at: float,
) -> pd.DataFrame:
    del predictions_modified_at, schedule_modified_at
    return load_predictions(path, schedule_path)


try:
    predictions = get_predictions(
        PREDICTIONS_PATH,
        SCHEDULE_PATH,
        PREDICTIONS_PATH.stat().st_mtime,
        SCHEDULE_PATH.stat().st_mtime,
    )
except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
    st.error(str(error))
    st.stop()


teams = available_teams(predictions)
minimum_date = predictions["game_date"].min()
maximum_date = predictions["game_date"].max()

with st.sidebar:
    st.header("Explore projections")
    st.page_link(
        "pages/How_It_Works.py",
        label="How the model works",
        icon="📘",
    )
    st.divider()
    selected_teams = st.multiselect(
        "Search and select teams",
        options=teams,
        placeholder="Type a team name...",
        help="Leave blank to show every matchup.",
    )
    selected_dates = st.date_input(
        "Date range",
        value=(minimum_date, maximum_date),
        min_value=minimum_date,
        max_value=maximum_date,
    )
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date = minimum_date
        end_date = maximum_date

    min_confidence = st.slider(
        "Minimum predicted win probability",
        min_value=50,
        max_value=95,
        value=50,
        step=1,
        format="%d%%",
    )
    neutral_filter = st.radio(
        "Venue type",
        options=["All games", "Neutral-site only", "Non-neutral only"],
    )
    sort_by = st.selectbox(
        "Sort results by",
        options=[
            "Tip-off time",
            "Highest win probability",
            "Largest projected spread",
            "Team name",
        ],
    )

st.markdown('<div class="eyebrow">2026–27 season</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-title">College basketball, projected.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-copy">Search a team, narrow the schedule, and explore '
    "the model's predicted winner, win probability, and point spread.</div>",
    unsafe_allow_html=True,
)
st.page_link(
    "pages/How_It_Works.py",
    label="New here? Read how the model works",
    icon="📘",
)

filtered = filter_predictions(
    predictions,
    selected_teams=selected_teams,
    start_date=start_date,
    end_date=end_date,
    min_confidence=float(min_confidence),
    neutral_filter=neutral_filter,
)

if sort_by == "Highest win probability":
    filtered = filtered.sort_values("win_pct", ascending=False)
elif sort_by == "Largest projected spread":
    filtered = filtered.sort_values("point_spread", ascending=False)
elif sort_by == "Team name":
    filtered = filtered.sort_values(["team_1", "team_2"])
else:
    filtered = filtered.sort_values("tipoff_datetime")

metric_columns = st.columns(4)
metric_columns[0].metric("Matching games", f"{len(filtered):,}")
metric_columns[1].metric("Teams available", f"{len(teams):,}")
metric_columns[2].metric(
    "Average win chance",
    f"{filtered['win_pct'].mean():.1f}%" if not filtered.empty else "—",
)
metric_columns[3].metric(
    "Neutral-site games",
    f"{int(filtered['neutral_site'].sum()):,}" if not filtered.empty else "0",
)

st.caption(
    f"Data export updated {data_as_of(PREDICTIONS_PATH):%B %d, %Y at %I:%M %p}. "
    "Times are shown as provided in the source schedule."
)

if filtered.empty:
    st.info("No games match these filters. Try lowering the confidence or widening the date range.")
    st.stop()

st.subheader("Featured matchups")
card_rows = filtered.head(6).to_dict("records")
for row_start in range(0, len(card_rows), 3):
    for column, game in zip(st.columns(3), card_rows[row_start : row_start + 3]):
        with column:
            game_date = (
                game["game_date"].strftime("%a, %b %d")
                if isinstance(game["game_date"], date)
                else str(game["game_date"])
            )
            neutral_badge = (
                '<span class="neutral-badge">Neutral site</span>'
                if game["neutral_site"]
                else ""
            )
            st.markdown(
                f"""
                <div class="matchup-card">
                    <div class="matchup-card-date">{game_date} · {game["game_time"]}</div>
                    <div class="matchup-card-title">{game["matchup"]}</div>
                    <div class="card-detail">{game["venue"]}</div>
                    <div style="margin-top: 1rem;">
                        <div class="winner">{game["predicted_team"]} {game["win_pct"]:.1f}%</div>
                        <div class="card-detail">
                            {confidence_label(game["win_pct"])} · Projected margin {game["point_spread"]:.1f}
                        </div>
                    </div>
                    <div style="margin-top: 0.65rem;">{neutral_badge}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(int(game["win_pct"]), text=f"{game['win_pct']:.1f}% model win chance")

st.subheader("All matching games")
table = filtered[
    [
        "game_date",
        "game_time",
        "matchup",
        "venue",
        "neutral_site",
        "predicted_team",
        "win_pct",
        "point_spread",
    ]
].rename(
    columns={
        "game_date": "Date",
        "game_time": "Time",
        "matchup": "Matchup",
        "venue": "Venue",
        "neutral_site": "Neutral",
        "predicted_team": "Predicted winner",
        "win_pct": "Win probability",
        "point_spread": "Projected margin",
    }
)
st.dataframe(
    table,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Date": st.column_config.DateColumn(format="MMM D, YYYY"),
        "Win probability": st.column_config.NumberColumn(format="%.1f%%"),
        "Projected margin": st.column_config.NumberColumn(format="%.1f pts"),
    },
)

st.download_button(
    "Download filtered results",
    data=as_export_csv(filtered),
    file_name="filtered_basketball_projections.csv",
    mime="text/csv",
)

st.caption(
    "Model outputs are estimates for exploration, not guarantees or betting advice. "
    "The projected margin belongs to the displayed predicted winner."
)
