# NCAA Basketball Projection Explorer

An interactive Streamlit dashboard for searching and filtering the model's
upcoming college basketball win projections.

## Run locally

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app reads `data/2027_upcoming_predictions.csv` and uses
`data/schedule_2027_upcoming.csv` to normalize searchable team names.
Regenerate those exports from the model notebook before refreshing the
dashboard. Model training is not run while users interact with the app.

The app includes a separate **How It Works** page rendered from
`docs/model_guide.md`, with plain-language explanations of the model inputs,
predictions, and backtest metrics.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, choose **Create app**.
3. Select this repository and branch, then use `app.py` as the entry point.
4. For a private repository, ensure your Streamlit account and plan have access
   to deploy private GitHub repositories.

## Test

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
```
