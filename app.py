from __future__ import annotations

from datetime import date
from html import escape
from pathlib import Path
import pandas as pd
import streamlit as st

from odds_data import normalize_team_name
from prediction_data import (
    as_export_csv,
    available_teams,
    data_as_of,
    filter_predictions,
    load_predictions,
    load_yesterday_scorecard,
)

PROJECT_ROOT = Path(__file__).resolve().parent
PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "2027_upcoming_with_benchmarks.csv"
)
SCHEDULE_PATH = PROJECT_ROOT / "data" / "schedule_2027_upcoming.csv"
YESTERDAY_PATH = PROJECT_ROOT / "data" / "yesterday_model_scorecard.csv"


def format_number(value: object, digits: int = 1, signed: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(number):
        return "—"
    return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"


def probability_color(value: object) -> str:
    try:
        return "positive" if float(value) >= 50 else "negative"
    except (TypeError, ValueError):
        return "neutral"


def render_delta(
    value: object,
    tooltip: str,
    *,
    neutral: bool = False,
    positive_is_good: bool = True,
    suffix: str = " pts",
    threshold: float = 0.0,
) -> str:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        numeric_value = 0.0
    if neutral or numeric_value == 0:
        color = "neutral"
    elif (numeric_value >= threshold) == positive_is_good:
        color = "positive"
    else:
        color = "negative"
    return (
        '<span class="delta-tooltip ' + color + '" tabindex="0" '
        'role="note" data-tooltip="' + escape(tooltip, quote=True) + '">'
        + f"{numeric_value:+.1f}{suffix} <span aria-hidden='true'>ⓘ</span>"
        + "</span>"
    )


def render_card_metric(
    column: st.delta_generator.DeltaGenerator,
    label: str,
    value: str,
    tooltip: str,
    *,
    value_color: str = "neutral",
    delta_html: str = "",
) -> None:
    column.markdown(
        '<div class="compact-metric">'
        f'<div class="metric-tip" tabindex="0" data-tooltip="{escape(tooltip, quote=True)}">'
        f'{escape(label)} <span aria-hidden="true">ⓘ</span></div>'
        f'<div class="compact-value {value_color}">{escape(value)}</div>'
        f'<div class="compact-delta">{delta_html}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


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
        [data-testid="stVerticalBlockBorderWrapper"] h3 {
            font-size: 1.05rem;
            line-height: 1.2;
        }
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMetricLabel"] {
            font-size: 0.72rem;
            line-height: 1.2;
            min-height: 2.1rem;
            overflow-wrap: anywhere;
            white-space: normal;
        }
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMetricValue"] {
            font-size: clamp(0.95rem, 1.4vw, 1.25rem);
            overflow-wrap: anywhere;
        }
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"] p {
            overflow-wrap: anywhere;
        }
        [data-testid="stVerticalBlock"]:has(.card-state-underdog) {
            background: linear-gradient(145deg, #ecfdf5 0%, #f0fdf4 100%);
            border-color: #16a34a;
        }
        [data-testid="stVerticalBlock"]:has(.card-state-fade) {
            background: linear-gradient(145deg, #fff7f7 0%, #fef2f2 100%);
            border-color: #dc2626;
        }
        .card-state {
            border-radius: 999px;
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 800;
            margin-bottom: 0.45rem;
            padding: 0.25rem 0.55rem;
        }
        .card-state-underdog {
            background: #dcfce7;
            color: #166534;
        }
        .card-state-fade {
            background: #fee2e2;
            color: #991b1b;
        }
        .compact-metric {
            min-width: 0;
        }
        .metric-tip {
            color: #64748b;
            font-size: 0.68rem;
            line-height: 1.15;
            min-height: 2rem;
            overflow-wrap: anywhere;
            position: relative;
        }
        .compact-value {
            font-size: clamp(0.95rem, 1.4vw, 1.22rem);
            font-weight: 750;
            line-height: 1.15;
            overflow-wrap: anywhere;
        }
        .positive {
            color: #15803d;
        }
        .negative {
            color: #b91c1c;
        }
        .neutral {
            color: #0f172a;
        }
        .compact-delta {
            min-height: 1.2rem;
        }
        .delta-tooltip {
            border-radius: 999px;
            display: inline-block;
            font-size: 0.7rem;
            font-weight: 750;
            position: relative;
        }
        .metric-tip::after, .delta-tooltip::after {
            background: #0f172a;
            border-radius: 8px;
            bottom: calc(100% + 0.4rem);
            color: #fff;
            content: attr(data-tooltip);
            font-size: 0.72rem;
            font-weight: 500;
            left: 50%;
            line-height: 1.35;
            max-width: 14rem;
            opacity: 0;
            padding: 0.45rem 0.55rem;
            pointer-events: none;
            position: absolute;
            transform: translateX(-50%);
            transition: opacity 120ms ease;
            visibility: hidden;
            white-space: normal;
            width: max-content;
            z-index: 20;
        }
        .metric-tip:hover::after, .metric-tip:focus::after,
        .delta-tooltip:hover::after, .delta-tooltip:focus::after {
            opacity: 1;
            visibility: visible;
        }
        @media (max-width: 900px) {
            .block-container {
                padding-left: 0.7rem;
                padding-right: 0.7rem;
                max-width: 100% !important;
                min-width: 0 !important;
                width: 100% !important;
            }
            [data-testid="stMainBlockContainer"],
            [data-testid="stMain"] {
                max-width: 100% !important;
                min-width: 0 !important;
                overflow-x: hidden;
                width: 100% !important;
            }
            [data-testid="stVerticalBlock"]:has(
                > [data-testid="stElementContainer"] .card-state
            ) [data-testid="stHorizontalBlock"] {
                flex-wrap: nowrap !important;
                gap: 0.3rem;
                width: 100% !important;
            }
            [data-testid="stVerticalBlock"]:has(
                > [data-testid="stElementContainer"] .card-state
            ) [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
                min-width: 0 !important;
            }
            [data-testid="stVerticalBlock"]:has(
                > [data-testid="stElementContainer"] .card-state
            ) {
                padding: 0.65rem !important;
            }
            [data-testid="stVerticalBlock"]:has(.card-state) h3 {
                font-size: 0.92rem;
                overflow-wrap: anywhere;
            }
            [data-testid="stVerticalBlock"]:has(
                > [data-testid="stElementContainer"] .card-state
            ) [data-testid="stHorizontalBlock"]
            > [data-testid="stColumn"]:first-child {
                flex: 0.72 1 0 !important;
            }
            [data-testid="stVerticalBlock"]:has(
                > [data-testid="stElementContainer"] .card-state
            ) [data-testid="stHorizontalBlock"]
            > [data-testid="stColumn"]:not(:first-child) {
                flex: 2 1 0 !important;
            }
            .compact-value {
                font-size: 0.92rem;
            }
            .metric-tip {
                font-size: 0.61rem;
                min-height: 1.8rem;
            }
            .delta-tooltip {
                font-size: 0.62rem;
            }
            .card-state {
                font-size: 0.6rem;
                line-height: 1.2;
                white-space: normal;
            }
            .metric-tip::after, .delta-tooltip::after {
                max-width: 10rem;
                width: 10rem;
            }
        }
        @media (max-width: 600px) {
            .hero-title {
                font-size: 2.1rem;
            }
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
            min-height: 330px;
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
        .comparison {
            border-top: 1px solid #e2e8f0;
            margin-top: 0.95rem;
            padding-top: 0.8rem;
        }
        .comparison-row {
            align-items: center;
            display: grid;
            gap: 0.45rem;
            grid-template-columns: 4.1rem 1fr 1fr;
            margin-top: 0.42rem;
        }
        .comparison-label {
            color: #64748b;
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        .comparison-cell {
            background: #f8fafc;
            border-radius: 10px;
            color: #0f172a;
            font-size: 0.82rem;
            padding: 0.58rem;
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


@st.cache_data(show_spinner=False)
def get_yesterday_scorecard(path: Path, modified_at: float) -> pd.DataFrame:
    del modified_at
    return load_yesterday_scorecard(path)


try:
    predictions = get_predictions(
        PREDICTIONS_PATH,
        SCHEDULE_PATH,
        PREDICTIONS_PATH.stat().st_mtime,
        SCHEDULE_PATH.stat().st_mtime,
    )
    yesterday = get_yesterday_scorecard(
        YESTERDAY_PATH, YESTERDAY_PATH.stat().st_mtime
    )
except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
    st.error(str(error))
    st.stop()

if yesterday.empty:
    st.warning("No completed scorecard data is available yet.")
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
    benchmark_only = st.checkbox(
        "DraftKings benchmark available",
        value=True,
        help="Keep enabled to show only games with current DraftKings markets.",
    )
    sort_by = st.selectbox(
        "Sort results by",
        options=[
            "Tip-off time",
            "Highest win probability",
            "Largest projected spread",
            "Largest model edge",
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
    "the model's projections, with DraftKings as the side-by-side benchmark.</div>",
    unsafe_allow_html=True,
)
st.page_link(
    "pages/How_It_Works.py",
    label="New here? Read how the model works",
    icon="📘",
)

st.subheader("Choose teams")
selected_teams = st.multiselect(
    "Teams shown in the matchup cards",
    options=teams,
    placeholder="Search for one or more teams...",
    help=(
        "Choose any number of teams. Leave this blank to show every matchup "
        "that meets the filters."
    ),
)

filtered = filter_predictions(
    predictions,
    selected_teams=selected_teams,
    start_date=start_date,
    end_date=end_date,
    min_confidence=float(min_confidence),
    neutral_filter=neutral_filter,
)
if benchmark_only:
    filtered = filtered[
        filtered[["market_favorite", "market_spread", "market_total"]]
        .notna()
        .all(axis=1)
    ]

if sort_by == "Highest win probability":
    filtered = filtered.sort_values("win_pct", ascending=False)
elif sort_by == "Largest projected spread":
    filtered = filtered.sort_values("point_spread", ascending=False)
elif sort_by == "Largest model edge":
    filtered = filtered.sort_values(
        "model_win_probability_edge", ascending=False, na_position="last"
    )
elif sort_by == "Team name":
    filtered = filtered.sort_values(["team_1", "team_2"])
else:
    filtered = filtered.sort_values("tipoff_datetime")

st.subheader(
    f"Yesterday's scorecard · {yesterday['game_date'].iloc[0]:%b %d, %Y}"
)
benchmarked_yesterday = yesterday[
    yesterday["market_favorite"].notna()
] if "market_favorite" in yesterday else yesterday.iloc[0:0]
model_wins = int(
    benchmarked_yesterday["model_correct"].fillna(False).sum()
)
market_wins = int(
    benchmarked_yesterday["market_correct"].fillna(False).sum()
)
scorecard_games = len(benchmarked_yesterday)
metric_columns = st.columns(4)
metric_columns[0].metric(
    "Model winner picks",
    f"{model_wins}/{scorecard_games}",
    help=(
        "The large value is the number of correct model winner picks from "
        "the previous day. The percentage below shows its winner-pick accuracy."
    ),
)
if scorecard_games:
    metric_columns[0].markdown(
        render_delta(
            100 * model_wins / scorecard_games,
            "Winner-pick accuracy: correct model picks divided by scored games.",
            suffix="%",
            threshold=50,
        ),
        unsafe_allow_html=True,
    )
metric_columns[1].metric(
    "DraftKings winner picks",
    f"{market_wins}/{scorecard_games}",
    help=(
        "The large value is DraftKings' correct favorite picks from the "
        "previous day. The delta compares model wins minus DraftKings wins."
    ),
)
if scorecard_games:
    metric_columns[1].markdown(
        render_delta(
            model_wins - market_wins,
            "Model wins minus DraftKings wins. Positive means the model picked more winners yesterday; negative means DraftKings did.",
            suffix=" wins",
        ),
        unsafe_allow_html=True,
    )
metric_columns[2].metric(
    "Model avg. total miss (MAE)",
    f"{benchmarked_yesterday['model_total_error'].mean():.1f} pts"
    if not benchmarked_yesterday.empty
    else "—",
    help="MAE means mean absolute error: the average number of points the projected total missed the final total. Lower is better.",
)
metric_columns[3].metric(
    "DraftKings avg. total miss (MAE)",
    f"{benchmarked_yesterday['market_total_error'].mean():.1f} pts"
    if not benchmarked_yesterday.empty
    else "—",
    help="MAE means mean absolute error: the average number of points the DraftKings total missed the final total. Lower is better.",
)
st.caption(
    "MAE = mean absolute error, or average points missed. A lower total MAE is better."
)

st.caption(
    f"Upcoming benchmarks refreshed {data_as_of(PREDICTIONS_PATH):%B %d, %Y} at "
    f"{data_as_of(PREDICTIONS_PATH):%I:%M %p}. "
    f"{len(filtered):,} games match the current filters. "
    "The scorecard compares completed games from the previous day."
)

if filtered.empty:
    st.info("No games match these filters. Try lowering the confidence or widening the date range.")
    st.stop()

st.subheader("Featured matchups")
st.caption(
    "H2H shows the predicted winner. Spreads are shown for each favorite, "
    "and totals are the projected combined score."
)
card_rows = filtered.head(6).to_dict("records")
for row_start in range(0, len(card_rows), 2):
    for column, game in zip(st.columns(2), card_rows[row_start : row_start + 2]):
        with column:
            game_date = (
                game["game_date"].strftime("%a, %b %d")
                if isinstance(game["game_date"], date)
                else str(game["game_date"])
            )
            neutral_badge = (
                "Neutral site" if game["neutral_site"] else ""
            )
            has_benchmark = bool(pd.notna(game.get("market_favorite")))
            with st.container(border=True):
                st.caption(f"{game_date} · {game['game_time']}")
                st.markdown(f"### {game['matchup']}")
                st.caption(
                    f"{game['venue']}"
                    + (f" · {neutral_badge}" if neutral_badge else "")
                )
                if not has_benchmark:
                    st.info("No complete DraftKings H2H, spread, and total market found.")
                    continue

                model_is_underdog = (
                    normalize_team_name(game["predicted_team"])
                    != normalize_team_name(game["market_favorite"])
                )
                model_edge = float(game.get("model_win_probability_edge") or 0)
                if model_is_underdog:
                    st.markdown(
                        '<span class="card-state card-state-underdog" '
                        'title="The model projects a different winner than DraftKings, making the model pick the market underdog.">'
                        "Model underdog opportunity</span>",
                        unsafe_allow_html=True,
                    )
                elif model_edge < 0:
                    st.markdown(
                        '<span class="card-state card-state-fade" '
                        'title="The model and DraftKings agree on the winner, but the model gives that winner a lower chance than the market implies.">'
                        "Model below market confidence</span>",
                        unsafe_allow_html=True,
                    )

                label, model_column, market_column = st.columns([0.9, 2.1, 2.1])
                label.markdown("**Market**")
                model_column.markdown("**Our model**")
                market_column.markdown("**DraftKings**")

                label, model_column, market_column = st.columns([0.9, 2.1, 2.1])
                label.markdown("**H2H**")
                render_card_metric(
                    model_column,
                    game["predicted_team"],
                    f"{format_number(game['win_pct'])}%",
                    "H2H is head-to-head. The model estimates this team's probability of winning. Values at or above 50% are green; below 50% are red.",
                    value_color=probability_color(game["win_pct"]),
                )
                render_card_metric(
                    market_column,
                    game["market_favorite"],
                    f"{format_number(game['market_win_pct'])}%",
                    "DraftKings' no-vig moneyline probability for its favorite. Values at or above 50% are green; below 50% are red.",
                    value_color=probability_color(game["market_win_pct"]),
                    delta_html=render_delta(
                        model_edge,
                        "Model probability minus DraftKings' implied probability for the model's selected team. Positive means the model sees more value; negative means less.",
                    ),
                )

                label, model_column, market_column = st.columns([0.9, 2.1, 2.1])
                label.markdown("**Spread**")
                spread_edge = float(game["market_spread"]) - float(game["point_spread"])
                render_card_metric(
                    model_column,
                    game["predicted_team"],
                    f"−{format_number(game['point_spread'])}",
                    "The model's expected winning margin for its projected winner. A minus sign denotes the favorite.",
                    value_color="neutral",
                )
                render_card_metric(
                    market_column,
                    game["market_favorite"],
                    f"−{format_number(game['market_spread'])}",
                    "DraftKings' spread for its market favorite.",
                    value_color="neutral",
                    delta_html=render_delta(
                        spread_edge,
                        "DraftKings spread minus the model margin. Positive is red because the market asks for a larger margin; negative is green because the model margin is larger.",
                        positive_is_good=False,
                    ),
                )

                label, model_column, market_column = st.columns([0.9, 2.1, 2.1])
                label.markdown("**Total**")
                total_edge = float(game["market_total"]) - float(game["projected_total"])
                render_card_metric(
                    model_column,
                    "Combined score",
                    f"{format_number(game['projected_total'])}",
                    "The model's estimate of both teams' combined final score.",
                    value_color="neutral",
                )
                render_card_metric(
                    market_column,
                    "Combined score",
                    f"{format_number(game['market_total'])}",
                    "DraftKings' over/under line. The delta is directional only and remains black because a higher total is not inherently better.",
                    value_color="neutral",
                    delta_html=render_delta(
                        total_edge,
                        "DraftKings total minus the model total. Positive means the market total is higher; negative means it is lower. This comparison is informational, not a green/red value signal.",
                        neutral=True,
                    ),
                )

st.download_button(
    "Download filtered results",
    data=as_export_csv(filtered),
    file_name="filtered_basketball_projections.csv",
    mime="text/csv",
)

st.caption(
    "Model outputs are estimates for exploration, not guarantees or betting advice. "
    "The projected margin belongs to the displayed predicted winner, and the "
    "projected total estimates both teams' combined points."
)
