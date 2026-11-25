from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from prediction_data import (
    available_teams,
    load_predictions,
    load_yesterday_scorecard,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "2027_upcoming_with_benchmarks.csv"
SCHEDULE_PATH = PROJECT_ROOT / "data" / "schedule_2027_upcoming.csv"
SCORECARD_PATH = PROJECT_ROOT / "data" / "yesterday_model_scorecard.csv"


st.set_page_config(
    page_title="Grady's Projections | Full Details",
    page_icon="🔎",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_details(
    predictions_path: Path,
    schedule_path: Path,
    scorecard_path: Path,
    predictions_modified: float,
    schedule_modified: float,
    scorecard_modified: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del predictions_modified, schedule_modified, scorecard_modified
    predictions = load_predictions(predictions_path, schedule_path)
    scorecard = load_yesterday_scorecard(scorecard_path)
    return predictions, scorecard


try:
    upcoming, scorecard = load_details(
        PREDICTIONS_PATH,
        SCHEDULE_PATH,
        SCORECARD_PATH,
        PREDICTIONS_PATH.stat().st_mtime,
        SCHEDULE_PATH.stat().st_mtime,
        SCORECARD_PATH.stat().st_mtime,
    )
except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
    st.error(str(error))
    st.stop()

st.markdown(
    '<div style="color:#d97706;font-size:0.78rem;font-weight:700;'
    'letter-spacing:0.12em;text-transform:uppercase;">CALCULATION AUDIT</div>',
    unsafe_allow_html=True,
)
st.title("Full projection details")
st.write(
    "Inspect every model and DraftKings input used in the cards, including "
    "the exact actual-result errors behind the summary scorecard."
)

with st.expander("What do the columns mean?", expanded=False):
    st.markdown(
        """
        **Model vs. DraftKings terms**

        - **Model win % / Model winner:** The model's favorite and estimated win probability.
        - **DK favorite %:** DraftKings' no-vig implied chance for its favorite.
        - **H2H edge:** Model win probability minus the DraftKings probability for the model's pick.
        - **Model spread:** How many points the model expects its picked winner to win by.
        - **DK spread:** How many points DraftKings expects its favorite to win by.
        - **Spread edge:** DK spread minus the model spread; positive means DraftKings has the larger margin.
        - **Model total / DK total:** The projected combined points from each source.
        - **Total edge:** Model total minus the DraftKings total.

        **Completed-game terms**

        - **Actual margin:** Absolute distance between the final scores.
        - **Margin for market favorite:** Final margin, positive when the DraftKings favorite won.
        - **Model spread for DK favorite:** The model spread converted to the DraftKings favorite's perspective. Positive favors that team; negative favors the opponent.
        - **Spread error:** Absolute difference between a source's margin estimate and the final margin. Lower is better.
        - **Total error:** Absolute difference between a source's projected total and the final combined score. Lower is better.
        - **Model / DK closer:** Checked when that source's estimate was closer to the actual result than the other source.
        - **Tie:** Both sources missed the actual result by exactly the same amount.
        """
    )

upcoming_tab, completed_tab = st.tabs(
    ["Upcoming projections", "Completed scorecard"]
)

with upcoming_tab:
    st.subheader("Upcoming model vs. DraftKings")
    teams = st.multiselect(
        "Filter teams",
        options=available_teams(upcoming),
        placeholder="Leave blank for all teams",
    )
    benchmark_only = st.checkbox(
        "Only games with a complete DraftKings market",
        value=True,
    )

    upcoming_detail = upcoming.copy()
    if teams:
        team_set = {team.casefold() for team in teams}
        upcoming_detail = upcoming_detail[
            upcoming_detail["team_1"].str.casefold().isin(team_set)
            | upcoming_detail["team_2"].str.casefold().isin(team_set)
            | upcoming_detail["predicted_team"].str.casefold().isin(team_set)
        ]
    if benchmark_only:
        upcoming_detail = upcoming_detail[
            upcoming_detail[["market_favorite", "market_spread", "market_total"]]
            .notna()
            .all(axis=1)
        ]

    upcoming_detail["model_spread_edge"] = (
        upcoming_detail["market_spread"] - upcoming_detail["point_spread"]
    )
    upcoming_columns = [
        "game_date",
        "game_time",
        "matchup",
        "venue",
        "neutral_site",
        "predicted_team",
        "win_pct",
        "market_favorite",
        "market_win_pct",
        "model_win_probability_edge",
        "point_spread",
        "market_spread",
        "model_spread_edge",
        "model_total_edge",
        "projected_total",
        "market_total",
    ]
    st.caption(f"{len(upcoming_detail):,} rows")
    st.dataframe(
        upcoming_detail[
            [column for column in upcoming_columns if column in upcoming_detail]
        ],
        hide_index=True,
        use_container_width=True,
        column_config={
            "game_date": st.column_config.DateColumn("Date", format="MMM D, YYYY"),
            "game_time": "Tip-off",
            "matchup": "Matchup",
            "venue": "Venue",
            "neutral_site": st.column_config.CheckboxColumn("Neutral site"),
            "predicted_team": "Model winner",
            "win_pct": st.column_config.NumberColumn("Model win %", format="%.1f"),
            "market_favorite": "DraftKings favorite",
            "market_win_pct": st.column_config.NumberColumn(
                "DK favorite %", format="%.1f"
            ),
            "model_win_probability_edge": st.column_config.NumberColumn(
                "H2H edge", format="%+.1f"
            ),
            "point_spread": st.column_config.NumberColumn(
                "Model spread", format="%.1f"
            ),
            "market_spread": st.column_config.NumberColumn(
                "DK spread", format="%.1f"
            ),
            "model_spread_edge": st.column_config.NumberColumn(
                "Spread edge", format="%+.1f"
            ),
            "projected_total": st.column_config.NumberColumn(
                "Model total", format="%.1f"
            ),
            "market_total": st.column_config.NumberColumn(
                "DK total", format="%.1f"
            ),
            "model_total_edge": st.column_config.NumberColumn(
                "Total edge", format="%+.1f"
            ),
        },
    )
    st.download_button(
        "Download upcoming detail",
        data=upcoming_detail[
            [column for column in upcoming_columns if column in upcoming_detail]
        ].to_csv(index=False).encode("utf-8"),
        file_name="upcoming_projection_detail.csv",
        mime="text/csv",
    )

    st.caption(
        "Tip: use Show/hide columns in the table toolbar to keep only the "
        "fields relevant to the question you are auditing."
    )

with completed_tab:
    scorecard_date = scorecard["game_date"].iloc[0] if not scorecard.empty else None
    st.subheader(
        f"Completed scorecard · {scorecard_date:%b %d, %Y}"
        if scorecard_date
        else "Completed scorecard"
    )

    if scorecard.empty:
        st.info("No completed scorecard rows are available yet.")
        st.stop()

    benchmarked = scorecard[scorecard["market_favorite"].notna()].copy()
    completed_teams = sorted(
        pd.concat([benchmarked["team_1"], benchmarked["team_2"]])
        .dropna()
        .astype(str)
        .unique(),
        key=str.casefold,
    )
    selected_completed_teams = st.multiselect(
        "Filter completed teams",
        options=completed_teams,
        placeholder="Leave blank for all completed games",
    )
    if selected_completed_teams:
        team_set = {team.casefold() for team in selected_completed_teams}
        benchmarked = benchmarked[
            benchmarked["team_1"].str.casefold().isin(team_set)
            | benchmarked["team_2"].str.casefold().isin(team_set)
        ]

    completed_columns = [
        "game_date",
        "matchup",
        "home_score",
        "away_score",
        "actual_total",
        "actual_margin",
        "predicted_team",
        "market_favorite",
        "actual_winner",
        "model_correct",
        "market_correct",
        "point_spread",
        "market_spread",
        "model_margin_for_market",
        "actual_margin_for_market",
        "model_spread_error",
        "market_spread_error",
        "model_spread_correct",
        "market_spread_correct",
        "spread_tie",
        "projected_total",
        "market_total",
        "model_total_error",
        "market_total_error",
        "model_total_correct",
        "market_total_correct",
        "total_tie",
    ]
    st.caption(
        f"{len(benchmarked):,} scored games with DraftKings matchups. "
        "For spread, positive margins are expressed for the DraftKings favorite."
    )
    st.dataframe(
        benchmarked[
            [column for column in completed_columns if column in benchmarked]
        ],
        hide_index=True,
        use_container_width=True,
        column_config={
            "game_date": st.column_config.DateColumn("Date", format="MMM D, YYYY"),
            "matchup": "Matchup",
            "home_score": "Home score",
            "away_score": "Away score",
            "actual_total": st.column_config.NumberColumn(
                "Actual combined score", format="%.0f"
            ),
            "actual_margin": st.column_config.NumberColumn(
                "Actual winning margin", format="%.0f"
            ),
            "predicted_team": "Model winner",
            "market_favorite": "DraftKings favorite",
            "actual_winner": "Actual winner",
            "model_correct": st.column_config.CheckboxColumn(
                "Model picked winner"
            ),
            "market_correct": st.column_config.CheckboxColumn(
                "DK picked winner"
            ),
            "point_spread": st.column_config.NumberColumn(
                "Model spread", format="%.1f"
            ),
            "market_spread": st.column_config.NumberColumn(
                "DK spread", format="%.1f"
            ),
            "model_margin_for_market": st.column_config.NumberColumn(
                "Model spread for DK favorite", format="%.1f"
            ),
            "actual_margin_for_market": st.column_config.NumberColumn(
                "Actual margin for DK favorite", format="%.1f"
            ),
            "projected_total": st.column_config.NumberColumn(
                "Model total", format="%.1f"
            ),
            "market_total": st.column_config.NumberColumn(
                "DK total", format="%.1f"
            ),
            "spread_tie": st.column_config.CheckboxColumn("Spread tie"),
            "total_tie": st.column_config.CheckboxColumn("Total tie"),
            "model_spread_error": st.column_config.NumberColumn(
                "Model spread error", format="%.1f"
            ),
            "market_spread_error": st.column_config.NumberColumn(
                "DK spread error", format="%.1f"
            ),
            "model_total_error": st.column_config.NumberColumn(
                "Model total error", format="%.1f"
            ),
            "market_total_error": st.column_config.NumberColumn(
                "DK total error", format="%.1f"
            ),
            "model_spread_correct": st.column_config.CheckboxColumn(
                "Model spread closer"
            ),
            "market_spread_correct": st.column_config.CheckboxColumn(
                "DK spread closer"
            ),
            "model_total_correct": st.column_config.CheckboxColumn(
                "Model total closer"
            ),
            "market_total_correct": st.column_config.CheckboxColumn(
                "DK total closer"
            ),
        },
    )
    st.download_button(
        "Download completed detail",
        data=benchmarked.to_csv(index=False).encode("utf-8"),
        file_name=f"completed_projection_detail_{scorecard_date}.csv",
        mime="text/csv",
    )
