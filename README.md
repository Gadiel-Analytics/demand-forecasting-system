# Hierarchical Demand Forecasting System

> A demand-planning system that produces **coherent** forecasts across every level of a retail hierarchy — and translates them into the revenue and forecast-error cost a planner actually decides on. Built on the M5 benchmark.

**Author:** [Gadiel Analytics](https://github.com/Gadiel-Analytics) · **Website:** <https://gadielanalytics.com> · **Book:** [*The Analytics System*](https://theanalyticssystem.com) · **Contact:** <hello@gadielanalytics.com>

**▶ [Live dashboard](https://gadiel-analytics.github.io/demand-forecasting-system/)** · runs end-to-end on synthetic data with one command, no dataset download required.

---

## The problem this solves

A retailer forecasts demand to answer a money question: how much to stock, where, and when. Get it wrong low and you stock out — you lose the full margin on a sale you could have made. Get it wrong high and you tie up working capital and risk waste. The forecast is not the deliverable; the *decision* it informs is.

Most retail-forecasting projects miss two things that make the difference between a model and a system:

**Coherence.** Forecasts are consumed at many levels at once — a store manager plans at the store level, a category buyer at the category level, finance at the total level. If the store forecasts do not sum to the category forecast, and the category forecasts do not sum to the total, the business plans against numbers that contradict each other. This system enforces coherence by construction and *asserts* it before publishing: if the levels do not reconcile, nothing ships.

**Business translation.** Accuracy is necessary but not the point. This system prices the forecast's errors asymmetrically — a stockout costs more per unit than an overstock — and reports expected revenue and error cost, not just RMSSE. That is the number a planner drives down.

## What it does

Every run executes the full pipeline: ingest and validate the data, engineer causal (leakage-free) features, split temporally, train a gradient-boosted model tuned for intermittent demand, forecast the next 28 days, reconcile the forecasts across the hierarchy, assert coherence, evaluate accuracy with the competition-standard RMSSE, translate units into money, and publish an executive brief and a live dashboard.

```text
  INGEST ──▶ FEATURES ──▶ SPLIT ──▶ TRAIN ──▶ FORECAST ──▶ RECONCILE ──▶ EVALUATE ──▶ PUBLISH
  validate   causal lags   temporal  LightGBM  28-day       bottom-up     RMSSE +      brief +
  schema     calendar      holdout   tweedie   horizon      + coherence   financial    dashboard
             prices, SNAP                                    assertion     translation
             target-encode
```

## The data

Built on **M5** — the Walmart hierarchical retail dataset that is the industry-standard forecasting benchmark: 30,490 bottom-level series across 3,049 products, 10 stores, and 3 US states, with calendar events, SNAP benefit days, and sell prices, spanning 2011–2016.

Because the real M5 files are large and gated behind a Kaggle account, the repository ships a **synthetic M5-like generator** with the same schema, hierarchy, seasonality, and intermittency. This means anyone can clone the repo and run the entire pipeline in seconds, and CI can test it deterministically — the same "runs for any visitor" principle as the [flagship price-intelligence pipeline](https://github.com/Gadiel-Analytics/fmcg-price-intelligence). Point it at the real M5 files for production runs (see [DEPLOYMENT.md](DEPLOYMENT.md)).

## Run it

```bash
pip install -r requirements.txt
python run.py --synthetic          # full pipeline on synthetic data, ~10s
```

This generates `reports/executive_brief.md`, `reports/accuracy.csv`, and `dashboard/data.json`. Open `dashboard/index.html` to see the dashboard. To run on real M5 data, drop the three CSV files into `data/` and run `python run.py` (add `--sample-frac 0.1` to subsample for speed).

## Why this is a system, not a notebook

The engineering decisions are where the seniority shows:

| Decision | Why |
|---|---|
| **Tweedie objective** in LightGBM | Store-level retail demand is intermittent — many zeros. Tweedie models this directly; a plain regression does not. |
| **Causal feature engineering** | Every lag and rolling window is shifted so day *t* never sees its own target. One leaked future value inflates validation accuracy and breaks in production. |
| **Time-ordered target encoding** | The product×store interaction is the most predictive signal in M5, but mean-encoding leaks. The expanding, shifted encoding is leakage-safe. |
| **Bottom-up reconciliation + coherence assertion** | Makes the forecasts safe to plan against at every level. The pipeline refuses to publish incoherent forecasts. |
| **Asymmetric financial translation** | Encodes the real planning trade-off: stockouts cost more than overstocks. Turns accuracy into a business number. |
| **Synthetic data fallback** | The repo runs for anyone, and CI tests it deterministically. |

## Project structure

```text
demand-forecasting-system/
├── run.py                      end-to-end pipeline entrypoint
├── src/
│   ├── data.py                 load + validate M5 (schema, integrity checks)
│   ├── features.py             causal lags, calendar, price, target encoding
│   ├── model.py                LightGBM (tweedie) + temporal split
│   ├── reconciliation.py       hierarchical coherence (the differentiator)
│   ├── evaluate.py             RMSSE + asymmetric financial translation
│   └── synthetic.py            M5-like data generator (runs without download)
├── tests/                      pytest suite, incl. coherence + leakage checks
├── dashboard/                  D3 dashboard (GitHub Pages)
├── reports/                    auto-generated brief + accuracy
└── .github/workflows/ci.yml    runs the pipeline + tests on every push
```

## Tests

```bash
pytest tests/ -q
```

The suite covers data validation (negative sales rejected), feature causality, temporal-split integrity, the **coherence guarantee** (aggregated levels sum to the grand total), RMSSE correctness, and the financial-translation asymmetry. CI runs the whole pipeline on synthetic data on every push.

---

This is applied business data science: a forecasting model wrapped in the engineering — validation, coherence, evaluation, and financial translation — that makes it a planning system rather than an accuracy exercise. The instruments for turning these forecasts into pricing and assortment decisions are the subject of [*The Analytics System*](https://theanalyticssystem.com).

<div align="center">

**Gadiel Guadarrama, M.Sc.** — Decision-System Architect · Author of *The Analytics System*

[gadielanalytics.com](https://gadielanalytics.com) · [theanalyticssystem.com](https://theanalyticssystem.com) · hello@gadielanalytics.com

</div>
