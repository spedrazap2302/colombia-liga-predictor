# Liga BetPlay Colombia — Predictor

A Streamlit app for predicting Liga BetPlay Colombia (Colombian football) match outcomes and simulating the rest of the season. It combines Elo ratings, a Dixon-Coles-style attack/defense model, and ML-based features to estimate match probabilities, generate score predictions, and run Monte Carlo season simulations for the current standings.

## Stack

- Python
- Streamlit
- pandas / numpy
- scikit-learn

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Data

Match and coaching data live in `data/matches.csv.xlsx` and `data/coaches.xlsx`.
