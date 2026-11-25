from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = {
    "game_date",
    "game_time",
    "matchup",
    "venue",
    "neutral_site",
    "predicted_team",
    "win_pct",
    "point_spread",
    "projected_total",
}
SCORECARD_COLUMNS = {
    "game_date",
    "matchup",
    "predicted_team",
    "model_correct",
    "market_correct",
    "model_total_error",
    "market_total_error",
    "model_spread_correct",
    "market_spread_correct",
    "model_total_correct",
    "market_total_correct",
    "model_spread_error",
    "market_spread_error",
    "spread_tie",
    "total_tie",
}


def split_matchup(matchup: str) -> tuple[str, str]:
    for separator in (" @ ", " vs. "):
        if separator in matchup:
            first, second = matchup.split(separator, maxsplit=1)
            return first.strip(), second.strip()
    raise ValueError(f"Unsupported matchup format: {matchup!r}")


def load_predictions(path: Path, schedule_path: Path | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Prediction file not found at {path}. Run the model notebook to generate it."
        )

    predictions = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS.difference(predictions.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Prediction file is missing required columns: {missing}")

    teams = predictions["matchup"].map(split_matchup)
    predictions["team_1"] = teams.map(lambda pair: pair[0])
    predictions["team_2"] = teams.map(lambda pair: pair[1])

    if schedule_path is not None:
        if not schedule_path.is_file():
            raise FileNotFoundError(
                f"Schedule file not found at {schedule_path}. "
                "It is required to normalize team names."
            )
        schedule = pd.read_csv(schedule_path)
        schedule_columns = {
            "game_date",
            "game_time",
            "matchup",
            "venue",
            "team_1",
            "team_2",
        }
        missing_schedule_columns = schedule_columns.difference(schedule.columns)
        if missing_schedule_columns:
            missing = ", ".join(sorted(missing_schedule_columns))
            raise ValueError(f"Schedule file is missing required columns: {missing}")

        join_keys = ["game_date", "game_time", "matchup", "venue"]
        canonical_teams = schedule[list(schedule_columns)].drop_duplicates(join_keys)
        predictions = predictions.merge(
            canonical_teams,
            on=join_keys,
            how="left",
            suffixes=("_raw", "_canonical"),
            validate="many_to_one",
            indicator=True,
        )
        missing_matches = int((predictions["_merge"] != "both").sum())
        if missing_matches:
            raise ValueError(
                f"{missing_matches} prediction rows did not match the schedule "
                "when normalizing team names."
            )
        predictions["team_1"] = predictions["team_1_canonical"]
        predictions["team_2"] = predictions["team_2_canonical"]
        predictions = predictions.drop(
            columns=["_merge", "team_1_canonical", "team_2_canonical"]
        )
    for display_column, target_column in (
        ("matchup_display", "matchup"),
        ("team_1_display", "team_1"),
        ("team_2_display", "team_2"),
        ("predicted_team_display", "predicted_team"),
    ):
        if display_column in predictions:
            predictions[display_column] = predictions[display_column].where(
                predictions[display_column].notna(), predictions[target_column]
            )
            predictions[target_column] = predictions[display_column]
    predictions["game_date"] = pd.to_datetime(
        predictions["game_date"], format="%Y-%m-%d", errors="raise"
    ).dt.date
    predictions["tipoff_datetime"] = pd.to_datetime(
        predictions["game_date"].astype(str)
        + " "
        + predictions["game_time"].astype(str),
        format="%Y-%m-%d %I:%M %p",
        errors="raise",
    )
    predictions["neutral_site"] = (
        predictions["neutral_site"].astype(str).str.lower().map({"true": True, "false": False})
    )
    if predictions["neutral_site"].isna().any():
        raise ValueError("Prediction file contains invalid neutral_site values")

    predictions["win_pct"] = pd.to_numeric(predictions["win_pct"], errors="raise")
    predictions["point_spread"] = pd.to_numeric(
        predictions["point_spread"], errors="raise"
    )
    predictions["projected_total"] = pd.to_numeric(
        predictions["projected_total"], errors="raise"
    )
    for column in (
        "market_win_pct",
        "market_spread",
        "market_total",
        "over_price",
        "under_price",
    ):
        if column in predictions:
            predictions[column] = pd.to_numeric(predictions[column], errors="coerce")
    return predictions.sort_values(
        ["tipoff_datetime", "matchup"], na_position="last"
    ).reset_index(drop=True)


def available_teams(predictions: pd.DataFrame) -> list[str]:
    teams = pd.concat(
        [predictions["team_1"], predictions["team_2"]],
        ignore_index=True,
    )
    return sorted(
        teams.dropna().astype(str).str.strip().unique(),
        key=str.casefold,
    )


def filter_predictions(
    predictions: pd.DataFrame,
    selected_teams: Iterable[str] = (),
    start_date: date | None = None,
    end_date: date | None = None,
    min_confidence: float = 50.0,
    neutral_filter: str = "All games",
) -> pd.DataFrame:
    filtered = predictions
    teams = {team.casefold() for team in selected_teams}

    if teams:
        team_1_matches = filtered["team_1"].str.casefold().isin(teams)
        team_2_matches = filtered["team_2"].str.casefold().isin(teams)
        predicted_matches = filtered["predicted_team"].str.casefold().isin(teams)
        filtered = filtered[team_1_matches | team_2_matches | predicted_matches]

    if start_date:
        filtered = filtered[filtered["game_date"] >= start_date]
    if end_date:
        filtered = filtered[filtered["game_date"] <= end_date]

    filtered = filtered[filtered["win_pct"] >= min_confidence]

    if neutral_filter == "Neutral-site only":
        filtered = filtered[filtered["neutral_site"]]
    elif neutral_filter == "Non-neutral only":
        filtered = filtered[~filtered["neutral_site"]]
    elif neutral_filter != "All games":
        raise ValueError(f"Unknown neutral-site filter: {neutral_filter!r}")

    return filtered.copy().reset_index(drop=True)


def confidence_label(win_pct: float) -> str:
    if win_pct >= 75:
        return "Strong favorite"
    if win_pct >= 65:
        return "Clear favorite"
    if win_pct >= 55:
        return "Lean"
    return "Close matchup"


def as_export_csv(predictions: pd.DataFrame) -> bytes:
    columns = [
        "game_date",
        "game_time",
        "matchup",
        "venue",
        "neutral_site",
        "predicted_team",
        "win_pct",
        "point_spread",
        "projected_total",
    ]
    benchmark_columns = [
        "market_favorite",
        "market_win_pct",
        "market_spread",
        "market_total",
        "model_win_probability_edge",
        "model_total_edge",
    ]
    columns.extend(
        column for column in benchmark_columns if column in predictions.columns
    )
    return predictions[columns].to_csv(index=False).encode("utf-8")


def load_yesterday_scorecard(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Prior-day scorecard not found at {path}. Run the model notebook to "
            "refresh benchmark data."
        )
    scorecard = pd.read_csv(path)
    missing_columns = SCORECARD_COLUMNS.difference(scorecard.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Prior-day scorecard is missing required columns: {missing}")

    scorecard["game_date"] = pd.to_datetime(
        scorecard["game_date"], format="%Y-%m-%d", errors="raise"
    ).dt.date
    for column in (
        "model_correct",
        "market_correct",
        "model_spread_correct",
        "market_spread_correct",
        "model_total_correct",
        "market_total_correct",
        "spread_tie",
        "total_tie",
    ):
        values = scorecard[column].astype(str).str.lower().map(
            {"true": True, "false": False}
        )
        scorecard[column] = values
    for column in (
        "model_total_error",
        "market_total_error",
        "model_spread_error",
        "market_spread_error",
    ):
        scorecard[column] = pd.to_numeric(scorecard[column], errors="coerce")
    return scorecard


def data_as_of(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime)
