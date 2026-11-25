# How these predictions work

The model estimates which team is more likely to win, then gives a projected
winning margin. It is designed to help you compare matchups, not to guarantee a
winner.

## The short version

For every team, the model looks at recent game performance and asks:

1. How well has this team played recently?
2. How strong were its recent opponents?
3. Is the team playing at home, away, or at a neutral site?

Those inputs produce three results:

- **Win probability:** the estimated chance that the displayed team wins.
- **Projected margin:** the estimated number of points by which the displayed
  winner may finish ahead.
- **Projected total:** the estimated combined score for both teams.

### Example

Suppose a projection says:

> Ole Miss — 58.4% win chance — projected margin 5.3 points

Read this as:

- The model picks Ole Miss to win.
- Ole Miss would win in about 58 out of 100 similar matchups, according to the
  model.
- Ole Miss's expected winning margin is about 5 points, with substantial
  uncertainty.

It does **not** mean the game is 58.4% certain to finish by exactly 5.3 points.

## What information the model uses

### Recent net rating

Net rating measures how many more points a team scored than it allowed after
adjusting for possessions. It is a compact way to describe overall performance.

The model averages the team's previous 10 games. The current game is excluded,
so its result cannot leak into its own prediction.

### Strength of recent opponents

Beating strong opponents means more than beating weak opponents. The model adds
context by tracking the recent net rating of each team's opponents.

The final strength edge is:

> your recent team strength plus your recent opponent strength, compared with
> the other team's equivalent measure

If a team has a positive strength edge, the model views it as the stronger side
heading into the matchup.

### Home-court advantage

The model also learns the usual advantage of playing at home. For neutral-site
games, that home-court input is removed.

## How the numbers are produced

One statistical model estimates win probability. A second, separate model
estimates the expected point margin. A third model estimates total points from
each team's recent scoring and points allowed. All are trained on completed
SportsDataverse NCAA games without using the current game's result as an input.

The model is trained again as newer completed games become available. It does
not make a prediction by simply matching team names or copying betting odds.

## DraftKings benchmark

The dashboard places the model's prediction beside the current DraftKings
moneyline favorite, spread, and total when a market match is available.
Moneyline prices are converted to no-vig implied probabilities before the
model edge is shown. Missing markets are labeled rather than treated as zero.

The scorecard at the top compares winner picks and total-point error for the
model and DraftKings on the latest date represented in the completed game
data. When the data feed is behind, the displayed scorecard date makes that
lag explicit.

## How successful have the predictions been?

The results below come from testing five completed seasons, 2022 through 2026.
For each season, the model was trained only on earlier seasons and then tested
on the later season. This is called **walk-forward testing**.

### About 72% of winners were picked correctly

**Winner accuracy: 71.9% on average.**

If the model predicted 100 games, it picked the actual winner correctly in
about 72 of them on the historical tests.

This is useful, but it is not a guarantee for the next 100 games. Close games
and unexpected performances will still produce misses.

### The model correctly ranked winners from losers

**AUC: 0.753 on average.**

AUC is a ranking score between 0.5 and 1.0:

- **0.5** means the ranking is no better than a coin flip.
- **1.0** means the model always ranks the winner above the loser.
- **0.753** means the model usually gives stronger win probabilities to teams
  that actually win.

You do not need to use this number day to day; it is a quality check on the
model.

### The confidence percentages were reasonably calibrated

**Brier score: 0.184, compared with 0.222 for a simple home-team baseline.**

The Brier score measures the quality of probabilities, not just whether the
winner was guessed correctly. Lower is better.

For example, a game predicted at 70% deserves a better probability score when
the favorite wins than when a team is predicted at 50%. The model's lower score
than the baseline means its percentages carried more useful information than
simply relying on the historical home-win rate.

The model's **log loss**, another probability-quality score where lower is
better, was **0.541** versus **about 0.638** for the simple home-team baseline.

### The projected margin was typically off by about 10.7 points

Across the historical tests, the average absolute margin error was **10.7
points**.

If the model projects an 8-point win, the final margin might be close, but a
normal miss of around 10 points is still possible in either direction. This
also means the displayed winner can still lose, especially in close matchups.

## Confidence labels

The app groups predictions this way:

- **Close matchup:** 50% to 54.9%
- **Lean:** 55% to 64.9%
- **Clear favorite:** 65% to 74.9%
- **Strong favorite:** 75% or higher

A 55% favorite is only a modest lean. It is much less certain than a 90%
favorite.

## Important limitations

- The model uses team-level recent performance, not injuries, suspensions,
  travel, rest, weather, player availability, or breaking news.
- A team with a new roster may not be represented perfectly by older results.
- Games against teams outside the available NCAA data population cannot be
  projected reliably and may be omitted.
- Basketball is unpredictable. Upsets do happen.
- These outputs are estimates for exploration, not guarantees or betting
  advice.

## Bottom line

Use the win probability to compare relative confidence, and use the projected
margin as a rough sense of expected separation. Treat games near 50% as close,
and remember that the model's historical winner-picking accuracy is about 72%,
not 100%.
