from __future__ import annotations

import argparse
import os
import re
from datetime import date, datetime, time, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
DEFAULT_SPORT = "basketball_ncaab"
DEFAULT_BOOKMAKER = "draftkings"


def require_api_key(value: str | None = None) -> str:
    api_key = value or os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set ODDS_API_KEY before refreshing DraftKings benchmark data."
        )
    return api_key


def fetch_odds_snapshot(
    api_key: str,
    *,
    snapshot_date: date | datetime | str | None = None,
    sport: str = DEFAULT_SPORT,
    bookmaker: str = DEFAULT_BOOKMAKER,
) -> dict[str, Any]:
    if snapshot_date is None:
        url = f"{ODDS_API_BASE_URL}/sports/{sport}/odds"
    else:
        timestamp = format_snapshot_timestamp(snapshot_date)
        url = f"{ODDS_API_BASE_URL}/historical/sports/{sport}/odds"

    params: dict[str, str] = {
        "apiKey": require_api_key(api_key),
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "bookmakers": bookmaker,
    }
    if snapshot_date is not None:
        params["date"] = timestamp

    response = requests.get(url, params=params, timeout=30)
    if not response.ok:
        raise RuntimeError(
            f"Odds API request failed with {response.status_code}: {response.text}"
        )
    payload = response.json()
    if snapshot_date is not None and not isinstance(payload, dict):
        raise ValueError("Historical odds response must be a JSON object")
    return payload


def format_snapshot_timestamp(value: date | datetime | str) -> str:
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    elif isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.combine(value, time.min)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def american_implied_probability(price: Any) -> float | None:
    try:
        american = float(price)
    except (TypeError, ValueError):
        return None
    if american > 0:
        return 100 / (american + 100)
    if american < 0:
        return abs(american) / (abs(american) + 100)
    return None


def no_vig_probability(
    first_probability: float | None,
    second_probability: float | None,
) -> tuple[float | None, float | None]:
    if first_probability is None or second_probability is None:
        return None, None
    overround = first_probability + second_probability
    if overround <= 0:
        return None, None
    return first_probability / overround, second_probability / overround


def flatten_historical_odds(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    snapshot = payload.get("timestamp")
    for game in payload.get("data", []):
        for bookmaker in game.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                for outcome in market.get("outcomes", []):
                    rows.append(
                        {
                            "snapshot_timestamp": snapshot,
                            "event_id": game["id"],
                            "commence_time": game["commence_time"],
                            "sport_key": game["sport_key"],
                            "away_team": game["away_team"],
                            "home_team": game["home_team"],
                            "bookmaker": bookmaker["key"],
                            "market": market["key"],
                            "last_update": market["last_update"],
                            "outcome": outcome["name"],
                            "price": outcome["price"],
                            "point": outcome.get("point"),
                        }
                    )
    flattened = pd.DataFrame(rows)
    if flattened.empty:
        return flattened
    for column in ("snapshot_timestamp", "commence_time", "last_update"):
        flattened[column] = pd.to_datetime(flattened[column], utc=True)
    return flattened


def _outcome_for_team(
    outcomes: list[dict[str, Any]], team: str
) -> dict[str, Any]:
    target = normalize_team_name(team)
    for outcome in outcomes:
        if normalize_team_name(str(outcome.get("name", ""))) == target:
            return outcome
    return {}


def _first_total(
    outcomes: list[dict[str, Any]], direction: str
) -> dict[str, Any]:
    for outcome in outcomes:
        if direction in str(outcome.get("name", "")).casefold():
            return outcome
    return {}


def summarize_odds_snapshot(payload: dict[str, Any] | list[dict[str, Any]]) -> pd.DataFrame:
    events = payload.get("data", []) if isinstance(payload, dict) else payload
    snapshot = (
        payload.get("timestamp")
        if isinstance(payload, dict)
        else datetime.now(timezone.utc).isoformat()
    )
    rows: list[dict[str, Any]] = []

    for game in events:
        for bookmaker in game.get("bookmakers", []):
            markets = {
                market["key"]: market.get("outcomes", [])
                for market in bookmaker.get("markets", [])
            }
            home_moneyline = _outcome_for_team(
                markets.get("h2h", []), game["home_team"]
            )
            away_moneyline = _outcome_for_team(
                markets.get("h2h", []), game["away_team"]
            )
            home_probability, away_probability = no_vig_probability(
                american_implied_probability(home_moneyline.get("price")),
                american_implied_probability(away_moneyline.get("price")),
            )
            home_spread = _outcome_for_team(
                markets.get("spreads", []), game["home_team"]
            )
            away_spread = _outcome_for_team(
                markets.get("spreads", []), game["away_team"]
            )
            over = _first_total(markets.get("totals", []), "over")
            under = _first_total(markets.get("totals", []), "under")

            if home_probability is not None and away_probability is not None:
                market_favorite = (
                    game["home_team"]
                    if home_probability >= away_probability
                    else game["away_team"]
                )
                market_win_probability = max(home_probability, away_probability) * 100
            else:
                market_favorite = None
                market_win_probability = None

            favorite_spread = (
                home_spread.get("point")
                if market_favorite == game["home_team"]
                else away_spread.get("point")
                if market_favorite == game["away_team"]
                else None
            )
            rows.append(
                {
                    "snapshot_timestamp": snapshot,
                    "event_id": game["id"],
                    "commence_time": game["commence_time"],
                    "home": game["home_team"],
                    "away": game["away_team"],
                    "bookmaker": bookmaker["key"],
                    "market_favorite": market_favorite,
                    "market_win_pct": market_win_probability,
                    "home_market_win_pct": (
                        home_probability * 100 if home_probability is not None else None
                    ),
                    "away_market_win_pct": (
                        away_probability * 100 if away_probability is not None else None
                    ),
                    "market_spread": (
                        abs(float(favorite_spread))
                        if favorite_spread is not None
                        else None
                    ),
                    "home_spread": home_spread.get("point"),
                    "away_spread": away_spread.get("point"),
                    "market_total": over.get("point", under.get("point")),
                    "over_price": over.get("price"),
                    "under_price": under.get("price"),
                }
            )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["commence_time"] = pd.to_datetime(summary["commence_time"], utc=True)
    if "snapshot_timestamp" in summary:
        summary["snapshot_timestamp"] = pd.to_datetime(
            summary["snapshot_timestamp"], utc=True
        )
    return summary.sort_values("commence_time").reset_index(drop=True)


def normalize_team_name(value: Any) -> str:
    replacements = {
        "north carolina": "unc",
        "st johns": "st john",
        "st john s": "st john",
        "saint johns": "st john",
        "mississippi": "ole miss",
    }
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()
    normalized = " ".join(
        token
        for token in normalized.split()
        if token
        not in {
            "aztecs",
            "bears",
            "bluejays",
            "broncos",
            "bulldogs",
            "cardinals",
            "cougars",
            "eagles",
            "falcons",
            "gators",
            "hawks",
            "hornets",
            "huskies",
            "lions",
            "mountaineers",
            "owls",
            "panthers",
            "pirates",
            "rebels",
            "royals",
            "tigers",
            "wildcats",
            "wolverines",
            "spartan",
            "spartans",
        }
    )
    normalized = normalized.strip()
    state_abbreviations = {
        "boise st": "boise state",
        "cleveland st": "cleveland state",
        "colorado st": "colorado state",
        "florida st": "florida state",
        "fresno st": "fresno state",
        "iowa st": "iowa state",
        "kansas st": "kansas state",
        "kent st": "kent state",
        "michigan st": "michigan state",
        "mississippi st": "mississippi state",
        "ohio st": "ohio state",
        "oregon st": "oregon state",
        "penn st": "penn state",
        "san diego st": "san diego state",
        "weber st": "weber state",
    }
    normalized = state_abbreviations.get(normalized, normalized)
    return replacements.get(normalized, normalized.replace(" ", ""))


def _team_similarity(first: Any, second: Any) -> float:
    first_normalized = normalize_team_name(first)
    second_normalized = normalize_team_name(second)
    if not first_normalized or not second_normalized:
        return 0.0
    if first_normalized == second_normalized:
        return 1.0
    if (
        first_normalized in second_normalized
        or second_normalized in first_normalized
    ):
        return 0.9
    return SequenceMatcher(None, first_normalized, second_normalized).ratio()


def match_predictions_to_odds(
    predictions: pd.DataFrame,
    odds: pd.DataFrame,
    *,
    minimum_similarity: float = 0.72,
) -> pd.DataFrame:
    model = predictions.copy()
    benchmark = odds.copy()
    if model.empty or benchmark.empty:
        return model.assign(_odds_match_index=pd.Series(dtype="Int64"))

    model["game_date"] = pd.to_datetime(model["game_date"]).dt.date
    benchmark["game_date"] = (
        pd.to_datetime(benchmark["commence_time"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.date
    )

    if not {"team_1", "team_2"}.issubset(model.columns):
        split = model["matchup"].str.extract(r"^(.*?) (?:@|vs\.) (.*?)$")
        model["team_1"] = split[0]
        model["team_2"] = split[1]

    benchmark_available = set(benchmark.index)
    matches: list[int | None] = []
    for _, prediction in model.iterrows():
        candidates = benchmark[
            benchmark["game_date"].eq(prediction["game_date"])
            & benchmark.index.to_series().isin(benchmark_available)
        ]
        best_index: int | None = None
        best_score = minimum_similarity
        for odds_index, market_game in candidates.iterrows():
            direct = (
                _team_similarity(prediction["team_1"], market_game["home"])
                + _team_similarity(prediction["team_2"], market_game["away"])
            ) / 2
            reversed_score = (
                _team_similarity(prediction["team_1"], market_game["away"])
                + _team_similarity(prediction["team_2"], market_game["home"])
            ) / 2
            score = max(direct, reversed_score)
            if score > best_score:
                best_score = score
                best_index = odds_index
        matches.append(best_index)
        if best_index is not None:
            benchmark_available.discard(best_index)

    benchmark_columns = [
        "event_id",
        "commence_time",
        "bookmaker",
        "market_favorite",
        "market_win_pct",
        "market_spread",
        "market_total",
        "over_price",
        "under_price",
    ]
    available_columns = [column for column in benchmark_columns if column in benchmark]
    matched_rows = []
    for index in matches:
        if index is None:
            matched_rows.append({column: None for column in available_columns})
        else:
            matched_rows.append(benchmark.loc[index, available_columns].to_dict())
    market_values = pd.DataFrame(matched_rows, index=model.index)
    matched = pd.concat([model, market_values], axis=1)
    matched["market_win_pct_for_prediction"] = matched.apply(
        lambda row: (
            row["market_win_pct"]
            if _team_similarity(
                row.get("predicted_team"), row.get("market_favorite")
            )
            >= 0.8
            else 100 - row["market_win_pct"]
            if pd.notna(row.get("market_win_pct"))
            else None
        ),
        axis=1,
    )
    matched["model_win_probability_edge"] = (
        matched["win_pct"] - matched["market_win_pct_for_prediction"]
    )
    matched["model_total_edge"] = (
        matched["projected_total"]
        - pd.to_numeric(matched["market_total"], errors="coerce")
    )
    return matched


def score_yesterday(
    model_results: pd.DataFrame,
    odds: pd.DataFrame,
) -> pd.DataFrame:
    scored = match_predictions_to_odds(model_results, odds)
    scored["actual_winner"] = scored.apply(
        lambda row: row["home"] if row["home_score"] > row["away_score"] else row["away"],
        axis=1,
    )
    scored["model_correct"] = (
        scored["predicted_team"].map(normalize_team_name)
        == scored["actual_winner"].map(normalize_team_name)
    )
    scored["market_correct"] = (
        scored["market_favorite"].map(normalize_team_name)
        == scored["actual_winner"].map(normalize_team_name)
    )
    scored["actual_total"] = scored["home_score"] + scored["away_score"]
    scored["model_total_error"] = (
        scored["projected_total"] - scored["actual_total"]
    ).abs()
    scored["market_total_error"] = (
        pd.to_numeric(scored["market_total"], errors="coerce")
        - scored["actual_total"]
    ).abs()
    return scored


def refresh_current_benchmarks(
    upcoming_predictions: pd.DataFrame,
    api_key: str,
    output_dir: Path,
) -> pd.DataFrame:
    live_payload = fetch_odds_snapshot(api_key)
    live_odds = summarize_odds_snapshot(live_payload)
    matched_upcoming = match_predictions_to_odds(upcoming_predictions, live_odds)

    output_dir.mkdir(parents=True, exist_ok=True)
    live_odds.to_csv(output_dir / "draftkings_ncaab_odds.csv", index=False)
    matched_upcoming.to_csv(
        output_dir / "2027_upcoming_with_benchmarks.csv", index=False
    )
    return matched_upcoming


def write_market_files(
    upcoming_predictions: pd.DataFrame,
    yesterday_predictions: pd.DataFrame,
    api_key: str,
    output_dir: Path,
    historical_date: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matched_upcoming = refresh_current_benchmarks(
        upcoming_predictions, api_key, output_dir
    )

    previous_date = historical_date or (date.today() - timedelta(days=1))
    historical_payload = fetch_odds_snapshot(api_key, snapshot_date=previous_date)
    historical_odds = summarize_odds_snapshot(historical_payload)
    yesterday_scorecard = score_yesterday(
        yesterday_predictions, historical_odds
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    yesterday_scorecard.to_csv(
        output_dir / "yesterday_model_scorecard.csv", index=False
    )
    return matched_upcoming, yesterday_scorecard


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh DraftKings benchmarks and an optional scorecard."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--yesterday-predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--historical-date", type=date.fromisoformat)
    args = parser.parse_args()

    upcoming = pd.read_csv(args.predictions)
    api_key = require_api_key()
    if args.yesterday_predictions:
        yesterday = pd.read_csv(args.yesterday_predictions)
        matched, scorecard = write_market_files(
            upcoming,
            yesterday,
            api_key,
            args.output_dir,
            args.historical_date,
        )
        print(f"Yesterday scorecard games: {len(scorecard):,}")
    else:
        matched = refresh_current_benchmarks(
            upcoming, api_key, args.output_dir
        )
    print(f"Matched upcoming games: {matched['event_id'].notna().sum():,}")


if __name__ == "__main__":
    main()
