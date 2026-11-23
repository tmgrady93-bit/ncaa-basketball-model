import pytest

from scripts.parse_2027_matchups import (
    build_name_lookup,
    match_team_name,
)


CANONICAL_TEAMS = [
    "Abilene Christian",
    "Alabama",
    "A&M-Corpus Christi",
    "Ark.-Pine Bluff",
    "Central Mich.",
    "Col. of Charleston",
    "CSUN",
    "CSU Bakersfield",
    "East Texas A&M",
    "Eastern Ky.",
    "Eastern Mich.",
    "East Carolina",
    "Florida",
    "George Washington",
    "Hawaii",
    "LMU (CA)",
    "North Ala.",
    "Northern Ariz.",
    "Northern Ill.",
    "NIU",
    "Northwestern St.",
    "Ole Miss",
    "Mount St. Mary's",
    "N.C. Central",
    "South Alabama",
    "South Carolina St.",
    "South Fla.",
    "Seattle U",
    "Southeast Mo. St.",
    "Southern Ind.",
    "South Dakota St.",
    "Texas",
    "Texas A&M",
    "Texas Southern",
    "Texas St.",
    "Southern U.",
    "UAlbany",
    "UNI",
    "UNCW",
    "Washington",
    "West Ga.",
    "Western Ill.",
    "Western Ky.",
    "Western Caro.",
    "UNC Asheville",
    "USC Upstate",
]


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("Abl Christian", "Abilene Christian"),
        ("N Alabama", "North Ala."),
        ("N Arizona", "Northern Ariz."),
        ("N Illinois", "NIU"),
        ("NW State", "Northwestern St."),
        ("C Michigan", "Central Mich."),
        ("E Kentucky", "Eastern Ky."),
        ("E Michigan", "Eastern Mich."),
        ("S Alabama", "South Alabama"),
        ("S Florida", "South Fla."),
        ("S Indiana", "Southern Ind."),
        ("W Illinois", "Western Ill."),
        ("W Kentucky", "Western Ky."),
        ("W Georgia", "West Ga."),
        ("Charleston", "Col. of Charleston"),
        ("Mississippi", "Ole Miss"),
        ("SE Missouri St", "Southeast Mo. St."),
        ("S Dakota St", "South Dakota St."),
        ("Texas So", "Texas Southern"),
        ("Texas A&M-CC", "A&M-Corpus Christi"),
        ("G Washington", "George Washington"),
        ("E Texas A&M", "East Texas A&M"),
        ("N Iowa", "UNI"),
        ("CS Northridge", "CSUN"),
        ("Loyola Mymt", "LMU (CA)"),
        ("NC Wilmington", "UNCW"),
        ("W Carolina", "Western Caro."),
        ("S Carolina St", "South Carolina St."),
        ("E Carolina", "East Carolina"),
        ("Seattle", "Seattle U"),
        ("Mt St Mary's", "Mount St. Mary's"),
        ("Southern", "Southern U."),
        ("Albany", "UAlbany"),
        ("Hawai'i", "Hawaii"),
        ("NC Central", "N.C. Central"),
        ("SC Upstate", "USC Upstate"),
        ("NC Asheville", "UNC Asheville"),
        ("AR-Pine Bluff", "Ark.-Pine Bluff"),
        ("CS Bakersfield", "CSU Bakersfield"),
    ],
)
def test_match_team_name_uses_explicit_aliases(
    raw_name: str, expected: str
) -> None:
    exact_lookup, normalized_names = build_name_lookup(CANONICAL_TEAMS)

    matched, score, is_matched = match_team_name(
        raw_name, exact_lookup, normalized_names
    )

    assert matched == expected
    assert score == 1.0
    assert is_matched


def test_match_team_name_rejects_alias_without_canonical_target() -> None:
    exact_lookup, normalized_names = build_name_lookup(CANONICAL_TEAMS)

    matched, score, is_matched = match_team_name(
        "W Florida", exact_lookup, normalized_names
    )

    assert matched == "W Florida"
    assert score == 0.0
    assert not is_matched
