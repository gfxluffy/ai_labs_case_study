# Challenge 1 — Solar Output Prediction Pipeline

## Repo layout

| Dir | Contents |
|---|---|
| `src/` | pipeline + supporting scripts, `conf.cfg` (this README) |
| `notebooks/` | exploratory notebooks behind the pipeline scripts (see below) |
| `data/` | raw inputs (Kaggle weather CSV, Global Solar Atlas GeoTIFF) |
| `output/` | generated artifacts (`city_solar_output.csv`, `modeling_data_aggregated.csv`, `solar_model.joblib`, etc.) |
| `docs/` | write-ups / presentation material |

## Pipeline stages

| Order | Script | Input | Output |
|---|---|---|---|
| 1 | `solar_extraction.py` | `PVOUT.tif` (Global Solar Atlas), city name list | `city_coords.csv`, `city_solar_output.csv` |
| 2 | `weather_prep.py` | Kaggle Australia weather CSV, `city_solar_output.csv` | `modeling_data_aggregated.csv` |
| 3 | `model_training.py` | `modeling_data_aggregated.csv` | `solar_model.joblib`, `feature_ranges.json`, `city_features_lookup.csv` |
| 4 | `agent.py` | trained model artifacts + a local LLM | natural-language answers to the demo questions |

Run each stage's script individually, in order, from inside `src/` (all paths are relative, e.g. `../output/...`, `../data/...`):

```bash
python solar_extraction.py
python weather_prep.py
python model_training.py
python agent.py
```

There's no `run_pipeline.py` wrapper — each script is meant to be run and its intermediate output inspected before moving to the next stage.

## Supporting scripts

These aren't part of the linear run order above — they validate the choices made in stages 3 and 4 rather than producing pipeline artifacts.

| Script | Validates | Input | Output |
|---|---|---|---|
| `model_comparison.py` | the choice of Ridge in `model_training.py` | `modeling_data_aggregated.csv` | printed LOOCV comparison table (Ridge, LinearRegression, Lasso, DecisionTree, RandomForest, GradientBoosting) |
| `eval_harness.py` | `agent.py`'s parsing + execution correctness | trained model artifacts (+ optionally the raw weather CSV, + optionally a running Ollama server) | printed pass/fail report, `../output/eval_results.csv` |

**`model_comparison.py`** runs the same LOOCV protocol as `model_training.py` across several regressor types and reports RMSE/MAE/R² for each, so "Ridge" is a documented comparison rather than an assumed choice — with only ~49 cities, tree-based models are expected to lose to the regularized linear models, and this makes that concrete. Run with:
```bash
python model_comparison.py
```

**`eval_harness.py`** covers three evaluation layers for the agent:
1. **White-box** — unit tests on individual tool functions (`get_city_row`, `check_extrapolation`, `predict_base_output`, etc.) against synthetic fixtures. No trained model or LLM needed; a failure here means a real logic bug.
2. **Trajectory (glass-box)** — checks whether `parse_question()` extracted the correct intent + arguments for each benchmark question, against a hand-labeled expected intent.
3. **Final response (black-box)** — checks whether the numeric value in the agent's final answer matches calling the underlying deterministic function directly, i.e. confirms the LLM-parsed path didn't corrupt the numbers on the way in (not a real-world-accuracy check — that's `model_training.py`'s LOOCV metrics).

Ollama doesn't need to be running: `parse_question()` falls back to rule-based parsing when the LLM is unreachable, so the harness still runs end-to-end (with a weaker trajectory-accuracy result, which is itself worth reporting — LLM-parsed vs. rule-based-fallback accuracy). Run with:
```bash
python eval_harness.py
```

## Notebooks

`../notebooks/` holds exploratory work that fed into the pipeline scripts above — not part of the run order, and not required to reproduce the pipeline outputs.

| Notebook | Purpose |
|---|---|
| `solar_data_check.ipynb` | Inspects `PVOUT.tif`'s metadata (tags, descriptions) to confirm it has no per-date info — the basis for assumption #1 below (long-term yearly average, not date-specific) |
| `weather_data_eda.ipynb` | EDA on the Kaggle weather CSV: load, preprocess, analyze, aggregate/join to solar output, and check feature correlations — the exploration behind `weather_prep.py` and the feature set used in `model_training.py` |

## Setup

```bash
pip install -r requirements.txt --break-system-packages   # or use a venv/conda env
```

For the agent's LLM backend (no paid API, runs fully locally):
```bash
# https://ollama.com
ollama pull llama3.1
ollama serve
```
`agent.py` talks to Ollama's OpenAI-compatible endpoint at `http://localhost:11434/v1`. Swap `llm_base_url` / `llm_model_name` in `conf.cfg`'s `[agent]` section if you're using a different local model.

## Before running, fill in these config values

All config for every script lives in **`conf.cfg`**, one section per script — edit that file to match your data/setup rather than editing the scripts themselves:

| Section | Key settings |
|---|---|
| `[solar_extraction]` | `tif_path`, `buffer_radius_m`, `city_names` (the full city/station list from your weather CSV) |
| `[weather_prep]` | `weather_csv_path`, `city_column`, `weather_feature_columns` (match your actual CSV's column names) |
| `[model_training]` | `features` (the domain-informed feature set), `ridge_alpha` |
| `[agent]` | `llm_base_url`, `llm_model_name` (swap for whatever local model you're running), `known_weather_features` |
| `[model_comparison]` | `random_state` |
| `[eval_harness]` | `eval_results_output_path` |

All scripts read `conf.cfg` relative to the current working directory, so run them from inside `src/` (as shown above).

## Key assumptions this pipeline bakes in


1. **Solar variable & temporal resolution**: uses PVOUT's long-term **yearly** average (not monthly), matched against weather statistics **aggregated across all available observations per city** — not date-specific. This is necessary, not just simpler: the weather CSV has no date column, so there's no way to build matching monthly aggregates, and PVOUT itself only offers long-term averages (no per-date values) regardless.
2. **City buffer radius**: 15km fixed radius around each geocoded city point (`buffer_radius_m` in `conf.cfg`'s `[solar_extraction]` section) — a documented simplification, not a researched constant.
3. **Small-n modeling**: with ~49 cities, `model_training.py` uses a narrow, domain-informed feature set, a regularized linear model (Ridge), and Leave-One-Out Cross-Validation instead of a train/test split — and reports an honest baseline ("predict the mean") comparison alongside the model's own error. `model_comparison.py` backs the Ridge choice with a side-by-side LOOCV comparison against LinearRegression, Lasso, and tree-based models, rather than leaving it an assumed pick.
4. **Farm-size queries** are deterministic scaling of the model's per-unit-area prediction (rate × area), not a separate model input — farm size was never a trained feature.
5. **Rainfall "what-if" queries** are counterfactual model calls (predict twice, compare) and are checked against the feature's observed training range — flagged as an extrapolation warning if the modified value falls outside it. These are learned partial correlations from a linear model, not a physically simulated causal effect — worth stating plainly if asked.
6. **Cities outside Global Solar Atlas's coverage** (e.g. Norfolk Island) will fail extraction with a `no_overlap` status in `city_solar_output.csv` and are excluded from the modeling/agent data — pending confirmation from the panel per the clarification email, but the pipeline handles this gracefully either way (no crash, clear logging).

## Note on the LLM's role in the agent

`agent.py` deliberately limits the LLM to two jobs: **parsing** the question into structured intent, and **phrasing** the final answer. All actual numbers come from the trained model / deterministic arithmetic (`execute_intent()`) — the LLM is explicitly instructed not to invent or alter figures when formatting the answer.
