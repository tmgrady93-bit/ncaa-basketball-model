from datetime import date

import pandas as pd

from odds_data import (
    format_snapshot_timestamp,
    local_today,
    match_predictions_to_odds,
    score_yesterday,
    summarize_odds_snapshot,
    split_model_matchup,
)


def sample_payload() -> dict:
    return {
        "timestamp": "2026-11-23T12:00:00Z",
        "data": [
            {
                "id": "event-1",
                "commence_time": "2026-11-24T02:00:00Z",
                "sport_key": "basketball_ncaab",
                "home_team": "Beta Wildcats",
                "away_team": "Alpha",
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "markets": [
                            {
                                "key": "h2h",
                                "last_update": "2026-11-23T12:00:00Z",
                                "outcomes": [
                                    {"name": "Beta Wildcats", "price": -200},
                                    {"name": "Alpha", "price": 170},
                                ],
                            },
                            {
                                "key": "spreads",
                                "last_update": "2026-11-23T12:00:00Z",
                                "outcomes": [
                                    {
                                        "name": "Beta Wildcats",
                                        "price": -110,
                                        "point": -4.5,
                                    },
                                    {
                                        "name": "Alpha",
                                        "price": -110,
                                        "point": 4.5,
                                    },
                                ],
                            },
                            {
                                "key": "totals",
                                "last_update": "2026-11-23T12:00:00Z",
                                "outcomes": [
                                    {
                                        "name": "Over",
                                        "price": -115,
                                        "point": 145.5,
                                    },
                                    {
                                        "name": "Under",
                                        "price": -105,
                                        "point": 145.5,
                                    },
                                ],
                            },
                        ],
                    }
                ],
            }
        ],
    }


def test_format_snapshot_timestamp_uses_midnight_utc() -> None:
    assert format_snapshot_timestamp(date(2026, 11, 23)) == "2026-11-23T00:00:00Z"
    assert isinstance(local_today(), date)
    assert split_model_matchup("Alpha vs. Beta") == ("Alpha", "Beta")
    assert split_model_matchup("Alpha @ Beta") == ("Alpha", "Beta")


def test_summarize_odds_snapshot_builds_game_benchmark() -> None:
    summary = summarize_odds_snapshot(sample_payload())

    assert summary.loc[0, "market_favorite"] == "Beta Wildcats"
    assert summary.loc[0, "market_spread"] == 4.5
    assert summary.loc[0, "market_total"] == 145.5
    assert summary.loc[0, "market_win_pct"] > 60


def test_match_predictions_uses_eastern_game_date() -> None:
    predictions = pd.DataFrame(
        {
            "game_date": ["2026-11-23"],
            "matchup": ["Alpha @ Beta"],
            "team_1": ["Alpha"],
            "team_2": ["Beta"],
            "predicted_team": ["Beta"],
            "win_pct": [70.0],
            "projected_total": [150.0],
        }
    )
    odds = summarize_odds_snapshot(sample_payload())

    matched = match_predictions_to_odds(predictions, odds)

    assert matched.loc[0, "event_id"] == "event-1"
    assert matched.loc[0, "market_win_pct_for_prediction"] > 60
    assert matched.loc[0, "model_total_edge"] == 4.5


def test_match_predictions_recognizes_state_abbreviation() -> None:
    predictions = pd.DataFrame(
        {
            "game_date": ["2026-11-23"],
            "matchup": ["Alpha @ Weber St"],
            "team_1": ["Alpha"],
            "team_2": ["Weber St"],
            "predicted_team": ["Weber St."],
            "win_pct": [60.0],
            "projected_total": [150.0],
        }
    )
    odds = pd.DataFrame(
        {
            "commence_time": ["2026-11-24T02:00:00Z"],
            "home": ["Weber State Wildcats"],
            "away": ["Alpha"],
            "market_favorite": ["Weber State Wildcats"],
            "market_win_pct": [70.0],
            "market_spread": [4.5],
            "market_total": [145.5],
        }
    )

    matched = match_predictions_to_odds(predictions, odds)

    assert matched.loc[0, "market_win_pct_for_prediction"] == 70.0
    assert matched.loc[0, "model_win_probability_edge"] == -10.0


def test_match_predictions_repairs_blank_team_columns() -> None:
    predictions = pd.DataFrame(
        {
            "game_date": ["2026-11-23"],
            "matchup": ["Alpha @ Beta"],
            "team_1": [None],
            "team_2": [None],
            "predicted_team": ["Beta"],
            "win_pct": [70.0],
            "projected_total": [150.0],
        }
    )

    matched = match_predictions_to_odds(
        predictions, summarize_odds_snapshot(sample_payload())
    )

    assert matched.loc[0, "team_1"] == "Alpha"
    assert matched.loc[0, "team_2"] == "Beta"
    assert matched.loc[0, "event_id"] == "event-1"


def test_score_yesterday_compares_model_and_market() -> None:
    model_results = pd.DataFrame(
        {
            "game_date": ["2026-11-23"],
            "matchup": ["Alpha @ Beta"],
            "team_1": ["Alpha"],
            "team_2": ["Beta"],
            "home": ["Beta"],
            "away": ["Alpha"],
            "home_score": [80],
            "away_score": [70],
            "predicted_team": ["Beta"],
            "win_pct": [70.0],
            "projected_total": [148.0],
        }
    )
    odds = summarize_odds_snapshot(sample_payload())

    scorecard = score_yesterday(model_results, odds)

    assert scorecard.loc[0, "model_correct"]
    assert scorecard.loc[0, "market_correct"]
    assert scorecard.loc[0, "model_total_error"] == 2.0
    assert scorecard.loc[0, "market_total_error"] == 4.5


def test_score_model_history_matches_abbreviated_mascot_names() -> None:
    predictions = pd.DataFrame(
        {
            "game_date": ["2026-11-23"],
            "matchup": ["E Carolina @ App State"],
            "team_1": ["E Carolina"],
            "team_2": ["App State"],
            "predicted_team": ["App State"],
            "win_pct": [60.0],
            "point_spread": [3.0],
            "projected_total": [145.0],
        }
    )
    odds = pd.DataFrame(
        {
            "commence_time": ["2026-11-23T23:00:00Z"],
            "home": ["App State Mountaineers"],
            "away": ["East Carolina Pirates"],
            "market_favorite": ["App State Mountaineers"],
            "market_win_pct": [65.0],
            "market_spread": [4.0],
            "market_total": [148.0],
        }
    )
    scores = pd.DataFrame(
        {
            "commence_time": pd.to_datetime(["2026-11-23T23:00:00Z"], utc=True),
            "home": ["App State Mountaineers"],
            "away": ["East Carolina Pirates"],
            "home_score": [75],
            "away_score": [70],
        }
    )
    from odds_data import score_model_history

    scorecard = score_model_history(
        predictions, odds, scores, score_date=date(2026, 11, 23)
    )

    assert len(scorecard) == 1
    assert scorecard.loc[0, "model_correct"]
