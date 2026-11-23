from __future__ import annotations

import argparse
import difflib
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

DATE_PATTERN = re.compile(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun) ([A-Z][a-z]{2}) (\d{1,2})$")
MATCHUP_PATTERN = re.compile(r"^(.+?)\s+(@|vs\.?)\s+(.+)$")
MONTH_NUMBERS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "Nov": 11,
    "Dec": 12,
}

TEAM_ALIASES = {
    "army": "Army West Point",
    "bethune": "Bethune-Cookman",
    "alcorn st": "Alcorn",
    "abl christian": "Abilene Christian",
    "queens": "Queens (NC)",
    "cs fullerton": "Cal St. Fullerton",
    "cs northridge": "CSUN",
    "hou christian": "Houston Christian",
    "purdue fw": "Purdue Fort Wayne",
    "tenn tech": "Tennessee Tech",
    "se louisiana": "Southeastern La.",
    "florida atlantic": "Fla. Atlantic",
    "miss valley st": "Mississippi Val.",
    "cal baptist": "California Baptist",
    "lamar": "Lamar University",
    "n iowa": "UNI",
    "n alabama": "North Ala.",
    "n arizona": "Northern Ariz.",
    "nw state": "Northwestern St.",
    "loyola mymt": "LMU (CA)",
    "loyola md": "Loyola Maryland",
    "loyola chi": "Loyola Chicago",
    "usc": "Southern California",
    "s utah": "Southern Utah",
    "illinois chicago": "UIC",
    "ut rio grande": "UTRGV",
    "ucsb": "UC Santa Barbara",
    "n texas": "North Texas",
    "w carolina": "Western Caro.",
    "umass": "Massachusetts",
    "maryland es": "UMES",
    "coastal car": "Coastal Carolina",
    "j madison": "James Madison",
    "siu edward": "SIUE",
    "saint mary s": "Saint Mary's (CA)",
    "saint marys": "Saint Mary's (CA)",
    "st mary s": "Saint Mary's (CA)",
    "ul monroe": "ULM",
    "sf austin": "SFA",
    "ucsd": "UC San Diego",
    "florida intl": "FIU",
    "miami": "Miami (FL)",
    "c connecticut": "Central Conn. St.",
    "st john s": "St. John's (NY)",
    "st thomas": "St. Thomas (MN)",
    "nc wilmington": "UNCW",
    "n florida": "North Florida",
    "n kentucky": "Northern Ky.",
    "c michigan": "Central Mich.",
    "e kentucky": "Eastern Ky.",
    "e michigan": "Eastern Mich.",
    "e illinois": "Eastern Ill.",
    "n illinois": "NIU",
    "s alabama": "South Alabama",
    "s carolina st": "South Carolina St.",
    "s florida": "South Fla.",
    "s indiana": "Southern Ind.",
    "w michigan": "Western Mich.",
    "w illinois": "Western Ill.",
    "w kentucky": "Western Ky.",
    "w georgia": "West Ga.",
    "w florida": "West Florida",
    "charleston": "Col. of Charleston",
    "e carolina": "East Carolina",
    "seattle": "Seattle U",
    "mt st mary s": "Mount St. Mary's",
    "southern": "Southern U.",
    "albany": "UAlbany",
    "hawai i": "Hawaii",
    "nc central": "N.C. Central",
    "sc upstate": "USC Upstate",
    "nc asheville": "UNC Asheville",
    "ar pine bluff": "Ark.-Pine Bluff",
    "cs bakersfield": "CSU Bakersfield",
    "mississippi": "Ole Miss",
    "se missouri st": "Southeast Mo. St.",
    "s dakota st": "South Dakota St.",
    "texas so": "Texas Southern",
    "texas a and m cc": "A&M-Corpus Christi",
    "g washington": "George Washington",
    "e texas a and m": "East Texas A&M",
    "n colorado": "Northern Colo.",
    "n dakota st": "North Dakota St.",
    "s illinois": "Southern Ill.",
    "e washington": "Eastern Wash.",
    "e tennessee st": "ETSU",
    "georgia so": "Ga. Southern",
    "c arkansas": "Central Ark.",
    "nc greensboro": "UNC Greensboro",
    "nc a and t": "N.C. A&T",
    "f dickinson": "FDU",
    "incarnate word": "UIW",
}


def normalize_team_name(name: str) -> str:
    normalized = name.strip().lower().replace("&", " and ")
    normalized = re.sub(r"\bst\.?\b", "st", normalized)
    normalized = re.sub(r"\bsaint\b", "st", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def canonical_team_names() -> list[str]:
    try:
        from . import read_data
    except ImportError:
        import read_data

    team_box = read_data.datasets["team_box"].filter(
        "season IN (2026, 2027)"
    )
    team_ids = read_data.datasets["team_ids"].filter(
        "season IN (2026, 2027)"
    )
    names = pd.concat(
        [
            team_box.project("team").df()["team"],
            team_ids.project("team").df()["team"],
        ],
        ignore_index=True,
    )
    return sorted(names.dropna().astype(str).unique())


def build_name_lookup(canonical_names: list[str]) -> tuple[dict[str, str], list[str]]:
    normalized_to_names: dict[str, list[str]] = {}
    for name in canonical_names:
        normalized_to_names.setdefault(normalize_team_name(name), []).append(name)
    return (
        {key: values[0] for key, values in normalized_to_names.items() if len(values) == 1},
        list(normalized_to_names),
    )


def match_team_name(
    raw_name: str,
    exact_lookup: dict[str, str],
    normalized_names: list[str],
    cutoff: float = 0.86,
) -> tuple[str, float, bool]:
    normalized = normalize_team_name(raw_name)
    alias = TEAM_ALIASES.get(normalized)
    if alias:
        canonical = exact_lookup.get(normalize_team_name(alias))
        if canonical:
            return canonical, 1.0, True
        return raw_name.strip(), 0.0, False

    exact = exact_lookup.get(normalized)
    if exact:
        return exact, 1.0, True

    candidates = difflib.get_close_matches(
        normalized,
        normalized_names,
        n=1,
        cutoff=cutoff,
    )
    if candidates:
        matched_normalized = candidates[0]
        score = difflib.SequenceMatcher(
            None, normalized, matched_normalized
        ).ratio()
        return exact_lookup[matched_normalized], score, True

    return raw_name.strip(), 0.0, False


def infer_game_date(day_text: str) -> datetime:
    match = DATE_PATTERN.fullmatch(day_text)
    if not match:
        raise ValueError(f"Invalid date row: {day_text!r}")
    month = MONTH_NUMBERS[match.group(2)]
    year = 2026 if month >= 10 else 2027
    return datetime(year, month, int(match.group(3)))


def parse_matchup_file(input_path: Path) -> pd.DataFrame:
    canonical_names = canonical_team_names()
    exact_lookup, normalized_names = build_name_lookup(canonical_names)
    rows = []
    current_date: datetime | None = None
    source_line_number = 0

    for source_line_number, raw_line in enumerate(
        input_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue

        fields = [field.strip() for field in raw_line.split("\t")]
        date_text = fields[0]
        if DATE_PATTERN.fullmatch(date_text):
            current_date = infer_game_date(date_text)
            continue
        if current_date is None:
            continue
        if len(fields) < 3:
            continue

        matchup_match = MATCHUP_PATTERN.fullmatch(fields[0])
        if not matchup_match:
            continue

        first_team_raw, separator, second_team_raw = matchup_match.groups()
        first_team, first_score, first_matched = match_team_name(
            first_team_raw, exact_lookup, normalized_names
        )
        second_team, second_score, second_matched = match_team_name(
            second_team_raw, exact_lookup, normalized_names
        )
        neutral_site = separator.lower().startswith("vs")

        if neutral_site:
            home, away = None, None
        else:
            away, home = first_team, second_team

        rows.append(
            {
                "season": 2027,
                "game_date": current_date.date().isoformat(),
                "game_time": fields[1] or None,
                "venue": fields[2] or None,
                "matchup": fields[0],
                "team_1_raw": first_team_raw.strip(),
                "team_2_raw": second_team_raw.strip(),
                "team_1": first_team,
                "team_2": second_team,
                "away": away,
                "home": home,
                "neutral_site": neutral_site,
                "team_1_match_score": first_score,
                "team_2_match_score": second_score,
                "needs_name_review": not (first_matched and second_matched),
                "source_line": source_line_number,
            }
        )

    schedule = pd.DataFrame(rows)
    if schedule.empty:
        raise ValueError(f"No matchups were parsed from {input_path}")
    return schedule.sort_values(
        ["game_date", "game_time", "source_line"], ignore_index=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse the 2027 NCAA matchup text into a model-ready CSV."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parents[1] / "data" / "2027 matchups.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "schedule_2027_upcoming.csv",
    )
    args = parser.parse_args()

    schedule = parse_matchup_file(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(args.output, index=False)

    review = schedule["needs_name_review"].sum()
    neutral = schedule["neutral_site"].sum()
    print(f"Parsed {len(schedule):,} games from {args.input}")
    print(f"Neutral-site games: {neutral:,}")
    print(f"Games needing name review: {review:,}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
