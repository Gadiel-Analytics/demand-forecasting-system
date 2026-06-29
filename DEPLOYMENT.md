# Deployment & Data

## Running on synthetic data (default)

No setup needed. The pipeline generates synthetic M5-like data if the real
files are absent:

```bash
pip install -r requirements.txt
python run.py --synthetic
```

## Running on the real M5 dataset

1. Obtain the three M5 files from the Kaggle M5 Forecasting — Accuracy
   competition: `sales_train_validation.csv`, `calendar.csv`,
   `sell_prices.csv`.
2. Place them in `data/`.
3. Run:

```bash
python run.py                  # full dataset (30,490 series)
python run.py --sample-frac 0.1   # 10% subsample for fast iteration
```

The real files are git-ignored (they are large and gated); only synthetic
data is regenerated in-repo.

## Continuous integration

`.github/workflows/ci.yml` runs the full pipeline on synthetic data and the
pytest suite on every push, so a broken pipeline never reaches `main`.

## Publishing the dashboard

The dashboard is static (`dashboard/index.html` + `dashboard/data.json`).
Enable GitHub Pages on the repository (Settings → Pages → deploy from
`main`, `/dashboard` folder) to publish it. Re-running the pipeline updates
`data.json`; commit it to refresh the live dashboard.
