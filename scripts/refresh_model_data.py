from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SOURCE_ROOT = (
    "https://raw.githubusercontent.com/"
    "sportsdataverse/ncaa-mbb-hoops-data/main/mbb"
)
SEASONS = tuple(range(2021, 2028))
ROLLING_GAMES = 10
CORE_FEATURES = ["netrtg_strength_edge", "is_home"]
TOTAL_FEATURES = [
    "home_prior_pts",
    "away_prior_pts",
    "home_prior_d_pts",
    "away_prior_d_pts",
]


def download_training_data(data_root: Path) -> None:
    for season in SEASONS:
        for dataset in ("team_box", "schedule"):
            source = (
                f"{SOURCE_ROOT}/{dataset}/parquet/"
                f"ncaa_mbb_{dataset}_{season}.parquet"
            )
            destination = (
                data_root / dataset / "parquet"
                / f"ncaa_mbb_{dataset}_{season}.parquet"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".parquet.tmp")
            request = urllib.request.Request(
                source, headers={"User-Agent": "ncaa-basketball-model-refresh"}
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    with temporary.open("wb") as output:
                        shutil.copyfileobj(response, output)
                temporary.replace(destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
            print(f"Downloaded {dataset} {season}")


def read_frames(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    team_box = pd.concat(
        [
            pd.read_parquet(
                data_root
                / "team_box"
                / "parquet"
                / f"ncaa_mbb_team_box_{season}.parquet"
            )
            for season in SEASONS
        ],
        ignore_index=True,
    )
    schedule = pd.concat(
        [
            pd.read_parquet(
                data_root
                / "schedule"
                / "parquet"
                / f"ncaa_mbb_schedule_{season}.parquet"
            )[["contest_id", "game_date"]]
            for season in SEASONS
        ],
        ignore_index=True,
    )
    return team_box, schedule


def build_model_games(
    team_box: pd.DataFrame, schedule: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    history = team_box.merge(
        schedule, on="contest_id", how="left", validate="many_to_one"
    )
    history["game_date"] = pd.to_datetime(history["game_date"])
    history["team_key"] = history["team"].astype(str)
    history["win"] = (history["pts"] > history["d_pts"]).astype(int)
    history = history.sort_values(["team_key", "game_date", "contest_id"])

    for column in ("netrtg", "pts", "d_pts"):
        history[f"prior_{column}"] = history.groupby("team_key")[column].transform(
            lambda values: values.shift(1).rolling(
                ROLLING_GAMES, min_periods=1
            ).mean()
        )

    history["opponent_key"] = np.where(
        history["team"].eq(history["home"]), history["away"], history["home"]
    )
    opponent_netrtg = history[
        ["contest_id", "team_key", "prior_netrtg"]
    ].rename(
        columns={
            "team_key": "opponent_key",
            "prior_netrtg": "opponent_prior_netrtg",
        }
    )
    history = history.merge(
        opponent_netrtg,
        on=["contest_id", "opponent_key"],
        how="left",
        validate="many_to_one",
    )
    history["prior_opponent_netrtg"] = history.groupby("team_key")[
        "opponent_prior_netrtg"
    ].transform(
        lambda values: values.shift(1).rolling(
            ROLLING_GAMES, min_periods=1
        ).mean()
    )
    history["adjusted_netrtg_strength"] = (
        history["prior_netrtg"] + history["prior_opponent_netrtg"]
    )

    home_history = history.query("team == home")[
        [
            "contest_id",
            "game_date",
            "season",
            "home",
            "away",
            "win",
            "pts",
            "d_pts",
            "prior_netrtg",
            "prior_opponent_netrtg",
            "adjusted_netrtg_strength",
            "prior_pts",
            "prior_d_pts",
        ]
    ].rename(
        columns={
            "prior_netrtg": "home_prior_netrtg",
            "prior_opponent_netrtg": "home_prior_opponent_netrtg",
            "adjusted_netrtg_strength": "home_adjusted_netrtg_strength",
            "prior_pts": "home_prior_pts",
            "prior_d_pts": "home_prior_d_pts",
        }
    )
    away_history = history.query("team == away")[
        [
            "contest_id",
            "prior_netrtg",
            "prior_opponent_netrtg",
            "adjusted_netrtg_strength",
            "prior_pts",
            "prior_d_pts",
        ]
    ].rename(
        columns={
            "prior_netrtg": "away_prior_netrtg",
            "prior_opponent_netrtg": "away_prior_opponent_netrtg",
            "adjusted_netrtg_strength": "away_adjusted_netrtg_strength",
            "prior_pts": "away_prior_pts",
            "prior_d_pts": "away_prior_d_pts",
        }
    )
    games = home_history.merge(
        away_history,
        on="contest_id",
        how="inner",
        validate="one_to_one",
    )
    games["netrtg_strength_edge"] = (
        games["home_adjusted_netrtg_strength"]
        - games["away_adjusted_netrtg_strength"]
    )
    games["home_margin"] = games["pts"] - games["d_pts"]
    games["total_points"] = games["pts"] + games["d_pts"]

    home_perspective = games[
        [
            "contest_id",
            "game_date",
            "season",
            "home",
            "away",
            "netrtg_strength_edge",
            "win",
        ]
    ].copy()
    home_perspective["is_home"] = 1
    away_perspective = home_perspective.copy()
    away_perspective["netrtg_strength_edge"] *= -1
    away_perspective["win"] = 1 - away_perspective["win"]
    away_perspective["is_home"] = 0
    model_data = pd.concat(
        [home_perspective, away_perspective], ignore_index=True
    )

    latest = (
        history.sort_values(["team_key", "game_date", "contest_id"])
        .groupby("team_key")
        .tail(1)[
            [
                "team_key",
                "adjusted_netrtg_strength",
                "prior_netrtg",
                "prior_pts",
                "prior_d_pts",
            ]
        ]
    )
    latest["latest_adjusted_netrtg_strength"] = latest[
        "adjusted_netrtg_strength"
    ].fillna(latest["prior_netrtg"])
    return games, model_data, latest.rename(columns={"team_key": "lookup_team"})


def fit_models(
    games: pd.DataFrame, model_data: pd.DataFrame
) -> tuple[Pipeline, Pipeline, Pipeline]:
    completed = games["season"].le(2027)
    win_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=2_000, random_state=42),
            ),
        ]
    ).fit(
        model_data[model_data["season"].le(2027)][CORE_FEATURES],
        model_data[model_data["season"].le(2027)]["win"],
    )
    margin_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=10.0)),
        ]
    ).fit(
        games.loc[completed, ["netrtg_strength_edge"]],
        games.loc[completed, "home_margin"],
    )
    total_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=10.0)),
        ]
    ).fit(
        games.loc[completed, TOTAL_FEATURES],
        games.loc[completed, "total_points"],
    )
    return win_model, margin_model, total_model


def build_upcoming_predictions(
    schedule_path: Path,
    latest: pd.DataFrame,
    win_model: Pipeline,
    margin_model: Pipeline,
    total_model: Pipeline,
) -> pd.DataFrame:
    schedule = pd.read_csv(schedule_path)
    schedule["game_date"] = pd.to_datetime(schedule["game_date"])
    as_of = pd.Timestamp.today().normalize()
    schedule = schedule[schedule["game_date"].ge(as_of)].copy()
    lookup = latest[
        [
            "lookup_team",
            "latest_adjusted_netrtg_strength",
            "prior_pts",
            "prior_d_pts",
        ]
    ]
    schedule = schedule.merge(
        lookup.rename(
            columns={
                "lookup_team": "team_1",
                "latest_adjusted_netrtg_strength": "team_1_strength",
                "prior_pts": "team_1_prior_pts",
                "prior_d_pts": "team_1_prior_d_pts",
            }
        ),
        on="team_1",
        how="left",
    ).merge(
        lookup.rename(
            columns={
                "lookup_team": "team_2",
                "latest_adjusted_netrtg_strength": "team_2_strength",
                "prior_pts": "team_2_prior_pts",
                "prior_d_pts": "team_2_prior_d_pts",
            }
        ),
        on="team_2",
        how="left",
    )
    inputs = [
        "team_1_strength",
        "team_2_strength",
        "team_1_prior_pts",
        "team_2_prior_pts",
        "team_1_prior_d_pts",
        "team_2_prior_d_pts",
    ]
    schedule = schedule[
        schedule[inputs].notna().all(axis=1) & ~schedule["needs_name_review"]
    ].copy()

    neutral = schedule["neutral_site"]
    strength_edge = schedule["team_1_strength"] - schedule["team_2_strength"]
    schedule["model_strength_edge"] = np.where(
        neutral, strength_edge, -strength_edge
    )
    schedule["model_is_home"] = np.where(neutral, 0, 1)
    matrix = schedule[["model_strength_edge", "model_is_home"]].rename(
        columns={
            "model_strength_edge": "netrtg_strength_edge",
            "model_is_home": "is_home",
        }
    )
    probabilities = win_model.predict_proba(matrix)[:, 1]
    margin_input = matrix[["netrtg_strength_edge"]]
    margin_home_predictions = margin_model.predict(margin_input)
    margin_transformed = margin_model.named_steps["scaler"].transform(
        margin_model.named_steps["imputer"].transform(margin_input)
    )
    margin_without_intercept = (
        margin_transformed @ margin_model.named_steps["regressor"].coef_
    )
    margins = np.where(
        neutral, margin_without_intercept, margin_home_predictions
    )
    schedule["predicted_team"] = np.where(
        probabilities >= 0.5,
        np.where(neutral, schedule["team_1"], schedule["home"]),
        np.where(neutral, schedule["team_2"], schedule["away"]),
    )
    schedule["win_pct"] = (
        100 * np.maximum(probabilities, 1 - probabilities)
    ).round(1)
    schedule["point_spread"] = np.abs(margins).round(1)

    total_input = pd.DataFrame(
        {
            "home_prior_pts": schedule["team_2_prior_pts"],
            "away_prior_pts": schedule["team_1_prior_pts"],
            "home_prior_d_pts": schedule["team_2_prior_d_pts"],
            "away_prior_d_pts": schedule["team_1_prior_d_pts"],
        }
    )
    home_total_predictions = total_model.predict(
        total_input[TOTAL_FEATURES]
    )
    reversed_total_input = total_input.rename(
        columns={
            "home_prior_pts": "away_prior_pts",
            "away_prior_pts": "home_prior_pts",
            "home_prior_d_pts": "away_prior_d_pts",
            "away_prior_d_pts": "home_prior_d_pts",
        }
    )[TOTAL_FEATURES]
    away_total_predictions = total_model.predict(reversed_total_input)
    schedule["projected_total"] = np.where(
        neutral,
        (home_total_predictions + away_total_predictions) / 2,
        home_total_predictions,
    ).round(1)
    return schedule[
        [
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
    ].sort_values(["game_date", "game_time"], na_position="last")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh SportsDataverse training data and regenerate projections."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-download", action="store_true", help="Use existing parquet files."
    )
    args = parser.parse_args()

    if not args.skip_download:
        download_training_data(args.data_root)
    team_box, schedule = read_frames(args.data_root)
    games, model_data, latest = build_model_games(team_box, schedule)
    win_model, margin_model, total_model = fit_models(games, model_data)
    predictions = build_upcoming_predictions(
        args.schedule, latest, win_model, margin_model, total_model
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(
        f"Generated {len(predictions):,} projections from "
        f"{len(games):,} completed games."
    )


if __name__ == "__main__":
    main()
