from datetime import date

import pandas as pd
import pytest

from prediction_data import (
    available_teams,
    filter_predictions,
    load_predictions,
    split_matchup,
)


def sample_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_date": [date(2026, 11, 23), date(2026, 11, 24)],
            "matchup": ["Alpha @ Beta", "Gamma vs. Delta"],
            "team_1": ["Alpha", "Gamma"],
            "team_2": ["Beta", "Delta"],
            "neutral_site": [False, True],
            "predicted_team": ["Beta", "Gamma"],
            "win_pct": [72.5, 54.0],
            "point_spread": [6.5, 1.0],
        }
    )


def test_split_matchup_handles_away_and_neutral_formats() -> None:
    assert split_matchup("Duke @ North Carolina") == ("Duke", "North Carolina")
    assert split_matchup("Duke vs. Kansas") == ("Duke", "Kansas")


def test_split_matchup_rejects_unknown_format() -> None:
    with pytest.raises(ValueError, match="Unsupported matchup format"):
        split_matchup("Duke-Kansas")


def test_filter_predictions_matches_either_team_and_canonical_prediction(
) -> None:
    predictions = sample_predictions()

    assert len(filter_predictions(predictions, selected_teams=["alpha"])) == 1
    assert len(filter_predictions(predictions, selected_teams=["GAMMA"])) == 1


def test_filter_predictions_combines_confidence_date_and_venue() -> None:
    predictions = sample_predictions()

    result = filter_predictions(
        predictions,
        start_date=date(2026, 11, 24),
        min_confidence=55.0,
        neutral_filter="Non-neutral only",
    )

    assert result.empty


def test_available_teams_is_unique_and_alphabetical() -> None:
    assert available_teams(sample_predictions()) == [
        "Alpha",
        "Beta",
        "Delta",
        "Gamma",
    ]


def test_load_predictions_validates_schema(tmp_path) -> None:
    invalid_file = tmp_path / "predictions.csv"
    pd.DataFrame({"matchup": ["Alpha @ Beta"]}).to_csv(invalid_file, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_predictions(invalid_file)


def test_load_predictions_uses_canonical_schedule_names(tmp_path) -> None:
    prediction_file = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "game_date": ["2026-11-23"],
            "game_time": ["7:00 PM"],
            "matchup": ["E Washington @ UNLV"],
            "venue": ["Arena"],
            "neutral_site": [False],
            "predicted_team": ["UNLV"],
            "win_pct": [60.0],
            "point_spread": [2.5],
        }
    ).to_csv(prediction_file, index=False)
    schedule_file = tmp_path / "schedule.csv"
    pd.DataFrame(
        {
            "game_date": ["2026-11-23"],
            "game_time": ["7:00 PM"],
            "matchup": ["E Washington @ UNLV"],
            "venue": ["Arena"],
            "team_1": ["Eastern Wash."],
            "team_2": ["UNLV"],
        }
    ).to_csv(schedule_file, index=False)

    predictions = load_predictions(prediction_file, schedule_file)

    assert predictions.loc[0, "team_1"] == "Eastern Wash."
    assert predictions.loc[0, "team_2"] == "UNLV"
