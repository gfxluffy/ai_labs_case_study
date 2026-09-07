"""
agent_v2.py

Pipeline for Challenge 1 - LLM agent layer.

Wraps the trained regression model (solar_model.joblib) with an LLM-based
orchestration layer: natural-language question -> structured intent ->
call the right function against the model -> natural-language answer.

Five query patterns are supported, matching what the trained model can
actually do (see the earlier discussion this pipeline is based on):

  1. BASE / FARM_SIZE queries - "what's my expected output in [city]" or
     "for a 5-hectare farm in [city]". The model predicts a long-term
     average daily SPECIFIC YIELD in kWh/kWp (Global Solar Atlas's PVOUT
     layer). Farm-size scaling is deterministic arithmetic, not a second
     model call: area (m^2) * panel capacity density (kWp/m^2) gives system
     capacity (kWp), which * specific yield (kWh/kWp) gives total output
     in kWh.

  2. WEATHER-CONDITIONED queries - "what's my expected yield tomorrow in
     [city] given Sunshine of 9 hours and Cloud9am of 1?" (the brief's own
     example question). Any subset of the model's weather features can be
     supplied; anything not supplied falls back to that city's stored
     average, and the response says explicitly which features were
     user-given vs. defaulted -- this is the "missing requirements"
     handling the brief asks candidates to state explicitly. Each
     supplied value is checked against its observed training range and
     flagged if it extrapolates beyond what the model has actually seen.

  3. SENSITIVITY queries - "how does yield change if rainfall doubles".
     Runs the model twice: once on the city's real feature vector, once on
     a modified copy, and reports the difference. Each modified feature is
     checked against its observed training range (feature_ranges.json) and
     flagged if it extrapolates beyond what the model has actually seen.

  4. COMPARE queries - ranks a list of cities by predicted output.

  5. DESCRIPTIVE STATS queries - computes raw-data statistics (mean,
     percentile, variability) directly from daily records (no model involved).

LLM BACKEND: this script is written against a local, OpenAI-compatible
endpoint (e.g. Ollama's `ollama serve`, which exposes /v1/chat/completions
on http://localhost:11434/v1) so no paid API is required.

FALLBACK BEHAVIOR: if the LLM server is unreachable or returns something
unparseable, call_llm() returns None instead of raising. parse_question()
then falls back to a keyword/regex-based intent parser (_rule_based_parse),
and format_answer() falls back to a templated (non-LLM) phrasing of the
result (_template_answer). This means the pipeline stays runnable end to
end without Ollama -- with reduced trajectory accuracy, which is itself a
useful thing to report when evaluating LLM-parsed vs. rule-based accuracy.

Usage:
    python agent.py
"""

import configparser
import json
import re
import sys
import time
from datetime import datetime

import joblib
import pandas as pd
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

cfg = configparser.ConfigParser()
cfg.read('conf.cfg')
_agent_cfg = cfg["agent"]

MODEL_PATH = _agent_cfg["model_path"]
FEATURE_RANGES_PATH = _agent_cfg["feature_ranges_path"]
CITY_FEATURES_PATH = _agent_cfg["city_features_path"]
CITY_COLUMN = _agent_cfg["city_column"]

# Raw (daily-granularity) weather records -- used ONLY for descriptive_stats
# (e.g. "how variable is rainfall in Cairns?").
WEATHER_RAW_PATH = _agent_cfg["weather_raw_path"]

VALID_STATS = {s.strip() for s in _agent_cfg["valid_stats"].split(",")}

# Point this at your local model server. Ollama example:
#   ollama pull llama3.1
#   ollama serve
LLM_BASE_URL = _agent_cfg["llm_base_url"]
LLM_MODEL_NAME = _agent_cfg["llm_model_name"]
LLM_API_KEY = _agent_cfg["llm_api_key"]  # Ollama ignores this, but the client requires a value

HECTARE_TO_M2 = int(_agent_cfg["hectare_to_m2"])
PANEL_KWP_PER_M2 = float(_agent_cfg["panel_kwp_per_m2"])


# ---------------------------------------------------------------------------
# Load artifacts once
# ---------------------------------------------------------------------------

def load_artifacts():
    bundle = joblib.load(MODEL_PATH)
    model, features, target = bundle["model"], bundle["features"], bundle["target"]

    with open(FEATURE_RANGES_PATH) as f:
        feature_ranges = json.load(f)

    city_features = pd.read_csv(CITY_FEATURES_PATH)
    list_cities = city_features[CITY_COLUMN].tolist()
    try:
        weather_raw = pd.read_csv(WEATHER_RAW_PATH)

    except FileNotFoundError:
        weather_raw = None  # descriptive_stats will report this honestly rather than guess

    return model, features, target, feature_ranges, city_features, weather_raw, list_cities


# ---------------------------------------------------------------------------
# Core model-calling functions (the agent's "tools")
# ---------------------------------------------------------------------------

def get_city_row(city, city_features, city_col=CITY_COLUMN):
    """
    Resolve a user-typed city name to a row in city_features.

    Exact (case-insensitive) match wins outright. Only if that fails do we
    fall back to substring matching -- and if the substring fallback is
    itself ambiguous (e.g. "Perth" contained in both "Perth" and
    "PerthAirport" -- though "Perth" would already hit the exact-match
    branch above -- or "Newcastle" vs a hypothetical "NewcastleAirport"),
    we don't silently guess: we pick the shortest matching name (closest
    to what was typed) and surface a `note` explaining the substitution,
    so it shows up in the final answer rather than failing silently.

    Returns (row_or_None, note_or_None).
    """
    query = city.strip().lower()

    exact = city_features[city_features[city_col].str.lower() == query]
    if not exact.empty:
        return exact.iloc[0], None

    candidates = city_features[city_features[city_col].str.lower().str.contains(query, regex=False)]
    if candidates.empty:
        return None, None
    if len(candidates) == 1:
        return candidates.iloc[0], None

    # Ambiguous: multiple locations contain the query string. Prefer the
    # shortest name as the closest match, but say so explicitly.
    candidate_names = candidates[city_col].tolist()
    chosen = candidates.loc[candidates[city_col].str.len().idxmin()]
    note = (
        f"'{city}' matched multiple locations ({', '.join(candidate_names)}); "
        f"used '{chosen[city_col]}' as the closest match."
    )
    return chosen, note


def check_extrapolation(feature_name, value, feature_ranges):
    """
    Shared helper: flag if `value` for `feature_name` falls outside the
    range the model was actually trained on. Used by both the
    weather-conditioned and sensitivity intents so extrapolation gets
    caught consistently regardless of which feature(s) a question touches.
    Returns a warning string, or None if within range / range unknown.
    """
    rng = feature_ranges.get(feature_name)
    if not rng:
        return None
    if value > rng["max"] or value < rng["min"]:
        return (
            f"Note: the given {feature_name} value ({value:.2f}) falls outside the range "
            f"the model was trained on ({rng['min']:.2f} to {rng['max']:.2f}). This prediction "
            f"is an extrapolation and should be treated as less reliable."
        )
    return None


def compute_descriptive_stat(city, feature, stat, weather_raw, threshold=None, city_col=CITY_COLUMN):
    """
    Basic-statistics tool: answers questions the trained regression model
    was never meant to answer (e.g. "how variable is rainfall in Cairns?",
    "what fraction of days in Brisbane have Cloud9am below 3?") directly
    from the raw daily weather records using plain pandas aggregation --
    no model, no LLM-computed numbers.

    This deliberately operates on WEATHER_RAW, not the city-level
    aggregate, since only the raw daily records carry real variability;
    the aggregate table has already thrown that information away.
    """
    if weather_raw is None:
        return None, ("I don't have the raw daily weather records loaded, so I can't compute "
                       "that statistic accurately -- I won't guess a number for this."), None

    if stat not in VALID_STATS:
        return None, f"'{stat}' isn't a statistic I know how to compute. Valid options: {sorted(VALID_STATS)}.", None

    if feature not in weather_raw.columns:
        return None, f"'{feature}' isn't a column in the weather data.", None

    row_sample, note = get_city_row(city, weather_raw.drop_duplicates(subset=[city_col]))
    if row_sample is None:
        return None, f"No data available for '{city}'.", None
    resolved_city = row_sample[city_col]

    city_rows = weather_raw[weather_raw[city_col] == resolved_city][feature].dropna()
    if city_rows.empty:
        return None, f"No '{feature}' records found for '{resolved_city}'.", note

    if stat == "mean":
        value = city_rows.mean()
    elif stat == "median":
        value = city_rows.median()
    elif stat == "std":
        value = city_rows.std()
    elif stat == "min":
        value = city_rows.min()
    elif stat == "max":
        value = city_rows.max()
    elif stat in ("percent_above", "percent_below"):
        if threshold is None:
            return None, f"'{stat}' needs a threshold value, none was given.", note
        value = (city_rows > threshold).mean() * 100 if stat == "percent_above" else (city_rows < threshold).mean() * 100
    elif stat.startswith("p") and stat[1:].isdigit():
        value = city_rows.quantile(int(stat[1:]) / 100)
    else:
        return None, f"Unhandled stat '{stat}'.", note

    result = {
        "city": resolved_city,
        "feature": feature,
        "stat": stat,
        "value": float(value),
        "n_observations": int(city_rows.shape[0]),
    }
    if threshold is not None:
        result["threshold"] = threshold

    return result, None, note



def predict_base_output(city, model, features, city_features):
    """Predicted long-term average daily specific yield (kWh/kWp) for a city."""
    row, note = get_city_row(city, city_features)
    if row is None:
        return None, f"No data available for '{city}' - it may be outside Global Solar Atlas's coverage or missing from the weather dataset.", None
    X = row[features].values.reshape(1, -1)
    pred = model.predict(X)[0]
    return pred, None, note


def predict_farm_output(city, hectares, model, features, city_features):
    """Total expected daily output (kWh) for a farm of the given size. This is
    deterministic scaling of the base specific-yield prediction - NOT a
    separate model call, since farm size isn't one of the trained features.
    area (m^2) * PANEL_KWP_PER_M2 gives system capacity (kWp); capacity (kWp)
    * specific yield (kWh/kWp) gives total output (kWh)."""
    specific_yield, error, note = predict_base_output(city, model, features, city_features)
    if error:
        return None, error, note
    area_m2 = hectares * HECTARE_TO_M2
    capacity_kwp = area_m2 * PANEL_KWP_PER_M2
    total_kwh = specific_yield * capacity_kwp
    result = {
        "specific_yield_kwh_per_kwp": float(specific_yield),
        "area_m2": area_m2,
        "capacity_kwp": capacity_kwp,
        "total_output_kwh": float(total_kwh),
    }
    return result, None, note


def predict_weather_conditioned_output(city, weather_overrides, model, features, city_features, feature_ranges, hectares=None):
    """
    Predict output for a city under user-specified weather conditions --
    this is the brief's headline example question ("what is my expected
    yield tomorrow given the following probable weather conditions...").

    weather_overrides: dict of {feature_name: value}, any subset of
    `features`. Anything NOT supplied falls back to that city's stored
    average -- this is the "missing requirements" handling the brief
    explicitly asks candidates to state in their presentation. We report
    which features were user-given vs. defaulted, rather than blending
    them invisibly, so the accuracy caveat is visible in the answer.

    Each user-supplied value is checked against the model's training
    range and flagged individually if it extrapolates.
    """
    row, note = get_city_row(city, city_features)
    if row is None:
        return None, f"No data available for '{city}'.", note

    feature_vector = row[features].copy()
    given_features, defaulted_features, warnings = [], [], []

    for feat in features:
        if feat in weather_overrides and weather_overrides[feat] is not None:
            value = weather_overrides[feat]
            feature_vector[feat] = value
            given_features.append(feat)
            warning = check_extrapolation(feat, value, feature_ranges)
            if warning:
                warnings.append(warning)
        else:
            defaulted_features.append(feat)

    pred = model.predict(feature_vector.values.reshape(1, -1))[0]

    result = {
        "specific_yield_kwh_per_kwp": float(pred),
        "given_features": {f: weather_overrides[f] for f in given_features},
        "defaulted_features": defaulted_features,
    }
    if hectares:
        area_m2 = hectares * HECTARE_TO_M2
        capacity_kwp = area_m2 * PANEL_KWP_PER_M2
        result["hectares"] = hectares
        result["capacity_kwp"] = capacity_kwp
        result["total_output_kwh"] = float(pred) * capacity_kwp

    combined_note = note
    if defaulted_features:
        defaults_note = (
            f"No value given for {', '.join(defaulted_features)} -- used {city}'s historical "
            f"average for {'these' if len(defaulted_features) > 1 else 'this'} feature instead."
        )
        combined_note = f"{combined_note} {defaults_note}" if combined_note else defaults_note
    if warnings:
        combined_note = f"{combined_note} {' '.join(warnings)}" if combined_note else " ".join(warnings)

    return result, None, combined_note


def predict_rainfall_sensitivity(city, multiplier, model, features, city_features, feature_ranges):
    """Counterfactual: how does predicted output change if Rainfall_mean is
    scaled by `multiplier`, holding other features constant. Flags
    extrapolation beyond the training range - see module docstring."""
    row, note = get_city_row(city, city_features)
    if row is None:
        return None, f"No data available for '{city}'.", None

    rainfall_col = "Rainfall_mean"
    if rainfall_col not in features:
        return None, "Rainfall is not one of the model's trained features, so this can't be simulated.", None

    baseline_pred = model.predict(row[features].values.reshape(1, -1))[0]

    modified_row = row[features].copy()
    modified_row[rainfall_col] = modified_row[rainfall_col] * multiplier
    modified_pred = model.predict(modified_row.values.reshape(1, -1))[0]

    extrapolation_warning = check_extrapolation(rainfall_col, modified_row[rainfall_col], feature_ranges)
    combined_note = f"{note} {extrapolation_warning}".strip() if note else extrapolation_warning

    delta = modified_pred - baseline_pred
    pct_change = (delta / baseline_pred * 100) if baseline_pred != 0 else float("nan")

    result = {
        "baseline_prediction": float(baseline_pred),
        "modified_prediction": float(modified_pred),
        "delta": float(delta),
        "pct_change": float(pct_change),
        "unit": "kWh/kWp/day",
    }
    return result, None, combined_note


def compare_cities(cities, model, features, city_features):
    """Rank a list of cities by predicted output - used for
    'which of these cities would yield more' style questions."""
    results = []
    for city in cities:
        pred, error, note = predict_base_output(city, model, features, city_features)
        if error:
            results.append({"city": city, "prediction": None, "error": error, "note": note})
        else:
            results.append({"city": city, "prediction": float(pred), "unit": "kWh/kWp/day", "error": None, "note": note})
    results.sort(key=lambda r: (r["prediction"] is None, -(r["prediction"] or 0)))
    return results


# ---------------------------------------------------------------------------
# LLM calling
# ---------------------------------------------------------------------------

_client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

KNOWN_WEATHER_FEATURES = [f.strip() for f in _agent_cfg["known_weather_features"].split(",")]

def call_llm(prompt, system=None):
    """Returns the LLM's text response, or None if the server is unreachable
    or the call otherwise fails -- callers (parse_question, format_answer)
    are responsible for falling back gracefully rather than crashing."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        response = _client.chat.completions.create(model=LLM_MODEL_NAME, messages=messages, temperature=0)
        return response.choices[0].message.content
    except Exception as exc:  # noqa: BLE001 -- any failure here should degrade, not crash the agent
        print(f"[warn] LLM call failed ({exc}) -- falling back to non-LLM logic.", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Step 1: Parse the question into structured intent
# ---------------------------------------------------------------------------

def build_parse_system_prompt(valid_cities: list[str]) -> str:
    """
    Builds the intent-parsing system prompt with the ACTUAL cities present
    in the trained dataset (city_features[CITY_COLUMN].tolist(), passed in
    as list_cities). Telling the parser only about cities the model can
    really answer for avoids it confidently normalizing a question toward
    a city that will just fail downstream in get_city_row().
    """
    city_list_str = ", ".join(sorted(valid_cities)) if valid_cities else "(none loaded)"

    return f"""You are an intent parser for a solar-output prediction assistant.
Given a user's question, respond with ONLY a JSON object (no other text) matching one of these shapes:

Valid city names: {city_list_str}.
If the user's phrasing clearly refers to one of these (e.g. "Perth Airport", "perth airport",
a misspelling, or a nearby/common alias), normalize it to the exact spelling from this list. If
it doesn't match any of these, use the city name exactly as the user stated it -- do not invent
or substitute a different valid city, and do not refuse to extract it.

1. Base/farm output query (no specific weather conditions mentioned):
   {{"intent": "base_or_farm_output", "city": "<city name>", "hectares": <number or null>}}

2. Weather-conditioned query (the user gives specific expected weather, e.g. "given Sunshine
   of 9 hours", "if Cloud9am is 1", "sunny with low humidity"):
   {{"intent": "weather_conditioned_output", "city": "<city name>", "hectares": <number or null>,
    "weather": {{"<feature name>": <value>, ...}}}}
   Valid weather feature names: {", ".join(KNOWN_WEATHER_FEATURES)}.
   Only include features the user actually specified values for.

3. Rainfall sensitivity / what-if query (relative change, e.g. "doubles", "increases by 50%"):
   {{"intent": "rainfall_sensitivity", "city": "<city name>", "multiplier": <number, e.g. 2.0 for "doubles">}}

4. Compare multiple cities:
   {{"intent": "compare_cities", "cities": ["<city1>", "<city2>", ...]}}

5. Descriptive/basic-stat query about historical weather (NOT about predicted solar output --
   this is for questions about weather variability, typical values, or frequency, e.g. "how
   variable is rainfall in Perth", "what's the typical Sunshine in Cairns", "what fraction of
   days in Brisbane have Cloud9am below 3"):
   {{"intent": "descriptive_stats", "city": "<city name>", "feature": "<feature name>",
    "stat": "mean"|"median"|"std"|"min"|"max"|"p10"|"p25"|"p75"|"p90"|"percent_above"|"percent_below",
    "threshold": <number, only required for percent_above/percent_below>}}

6. Anything else / unclear:
   {{"intent": "unknown"}}

Respond with ONLY the JSON object, nothing else."""


def _rule_based_parse(question, known_cities, feature_cols):
    """
    Fallback intent parser used only when the LLM is unreachable or returns
    unparseable output. Keyword/regex-based -- deliberately simple and
    conservative: when in doubt it returns 'unknown' rather than guessing,
    since a wrong tool call is worse than an honest "can't parse this".
    """
    q_lower = question.lower()

    cities_found = [c for c in known_cities if c.lower() in q_lower]
    # prefer longer (more specific) names first, e.g. "SydneyAirport" over "Sydney"
    cities_found.sort(key=len, reverse=True)

    if len(cities_found) >= 2 and any(w in q_lower for w in ("compare", "which", "more", "versus", " vs")):
        return {"intent": "compare_cities", "cities": cities_found}

    city = cities_found[0] if cities_found else None

    multiplier = None
    if "doubl" in q_lower:
        multiplier = 2.0
    elif "tripl" in q_lower:
        multiplier = 3.0
    else:
        pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", q_lower)
        if pct_match and ("increase" in q_lower or "decrease" in q_lower or "change" in q_lower):
            pct = float(pct_match.group(1))
            multiplier = 1 + pct / 100 if "increase" in q_lower else 1 - pct / 100
    if multiplier is not None and "rain" in q_lower:
        return {"intent": "rainfall_sensitivity", "city": city, "multiplier": multiplier}

    stat_keywords = {
        "average": "mean", "typical": "mean", "mean": "mean", "median": "median",
        "variable": "std", "variability": "std", "minimum": "min", "lowest": "min",
        "maximum": "max", "highest": "max", "percentile": "p90",
    }
    feature = next((f for f in feature_cols if f.lower() in q_lower), None)
    for kw, stat in stat_keywords.items():
        if kw in q_lower and feature:
            return {"intent": "descriptive_stats", "city": city, "feature": feature, "stat": stat}

    frac_below = re.search(r"below\s*(\d+(?:\.\d+)?)", q_lower)
    if "fraction" in q_lower and frac_below and feature:
        return {"intent": "descriptive_stats", "city": city, "feature": feature,
                "stat": "percent_below", "threshold": float(frac_below.group(1))}
    frac_above = re.search(r"above\s*(\d+(?:\.\d+)?)", q_lower)
    if "fraction" in q_lower and frac_above and feature:
        return {"intent": "descriptive_stats", "city": city, "feature": feature,
                "stat": "percent_above", "threshold": float(frac_above.group(1))}

    weather = {}
    for feat in feature_cols:
        pattern = re.compile(re.escape(feat.lower()) + r".{0,12}?(\d+(?:\.\d+)?)")
        m = pattern.search(q_lower)
        if m:
            weather[feat] = float(m.group(1))

    hectare_match = re.search(r"(\d+(?:\.\d+)?)\s*[- ]?hectare", q_lower)
    hectares = float(hectare_match.group(1)) if hectare_match else None

    if weather:
        return {"intent": "weather_conditioned_output", "city": city, "hectares": hectares, "weather": weather}

    if city:
        return {"intent": "base_or_farm_output", "city": city, "hectares": hectares}

    return {"intent": "unknown"}


def parse_question(question, list_cities):
    raw = call_llm(question, system=build_parse_system_prompt(list_cities))
    if raw is not None:
        # Local models sometimes wrap JSON in prose or code fences despite instructions -
        # extract the first {...} block defensively rather than assuming a clean response.
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass  # fall through to rule-based parsing below

    # LLM unreachable or returned something unparseable -- degrade, don't crash.
    return _rule_based_parse(question, list_cities, KNOWN_WEATHER_FEATURES)


# ---------------------------------------------------------------------------
# Step 2: Execute the parsed intent against the model
# ---------------------------------------------------------------------------

def execute_intent(intent_data, model, features, city_features, feature_ranges, weather_raw=None):
    intent = intent_data.get("intent")

    if intent == "base_or_farm_output":
        city = (intent_data.get("city") or "").strip() or None
        hectares = intent_data.get("hectares")
        if hectares:
            result, error, note = predict_farm_output(city, hectares, model, features, city_features)
            return {"type": "farm_output", "city": city, "hectares": hectares, "result": result,
                    "error": error, "warning": note}
        else:
            value, error, note = predict_base_output(city, model, features, city_features)
            return {"type": "base_output", "city": city, "value": value, "unit": "kWh/kWp/day",
                    "error": error, "warning": note}

    elif intent == "weather_conditioned_output":
        city = (intent_data.get("city") or "").strip() or None
        hectares = intent_data.get("hectares")
        weather = intent_data.get("weather") or {}
        result, error, warning = predict_weather_conditioned_output(
            city, weather, model, features, city_features, feature_ranges, hectares=hectares
        )
        return {"type": "weather_conditioned_output", "city": city, "result": result,
                "error": error, "warning": warning}

    elif intent == "rainfall_sensitivity":
        city = (intent_data.get("city") or "").strip() or None
        multiplier = intent_data.get("multiplier", 1.0)
        result, error, warning = predict_rainfall_sensitivity(
            city, multiplier, model, features, city_features, feature_ranges
        )
        return {"type": "rainfall_sensitivity", "city": city, "multiplier": multiplier,
                "result": result, "error": error, "warning": warning}

    elif intent == "compare_cities":
        cities = [c.strip() for c in intent_data.get("cities", []) if c and c.strip()]
        results = compare_cities(cities, model, features, city_features)
        return {"type": "compare_cities", "results": results}

    elif intent == "descriptive_stats":
        city = (intent_data.get("city") or "").strip() or None
        feature = intent_data.get("feature")
        stat = intent_data.get("stat")
        threshold = intent_data.get("threshold")
        result, error, note = compute_descriptive_stat(city, feature, stat, weather_raw, threshold=threshold)
        return {"type": "descriptive_stats", "city": city, "result": result, "error": error, "warning": note}

    else:
        return {"type": "unknown"}


# ---------------------------------------------------------------------------
# Step 3: Format the result into a natural language answer
# ---------------------------------------------------------------------------

def format_answer(execution_result):
    """Use the LLM to phrase the numeric result naturally - but the numbers
    themselves come entirely from execution_result (the model's actual
    output), never invented by the LLM. This keeps the LLM's role limited
    to parsing and phrasing, not fabricating figures."""
    if execution_result["type"] == "unknown":
        return ("I couldn't map that question to something I can compute. I can answer questions "
                "about expected solar output for a city, farm-size scaling, output under specific "
                "weather conditions, rainfall sensitivity, comparisons across cities, or basic "
                "weather statistics (typical values, variability, frequency).")

    if execution_result.get("error"):
        return execution_result["error"]

    prompt = (
        "Phrase the following computed result as a clear, natural-language answer for a solar "
        "farm planning assistant. Do not invent or alter any numbers - use exactly what's given. "
        "Always state the unit that goes with each number exactly as given in the data (e.g. "
        "kWh/kWp/day for a per-capacity specific yield, kWh/day for a total farm output, kWp for "
        "a system capacity) - never omit units or invent different ones. "
        "Keep it to 2-3 sentences.\n\n"
        f"Data: {json.dumps(execution_result, default=str)}"
    )
    llm_answer = call_llm(prompt)
    return llm_answer if llm_answer is not None else _template_answer(execution_result)


def _template_answer(execution_result: dict) -> str:
    """Non-LLM fallback phrasing, used when call_llm() returns None. Purely
    templated string formatting over fields that are already fully computed
    -- no numbers are generated here, only formatted."""
    t = execution_result.get("type")
    city = execution_result.get("city")

    if t == "farm_output":
        r = execution_result.get("result") or {}
        return (f"Estimated output for a {execution_result.get('hectares')}-hectare farm in {city}: "
                f"{r.get('total_output_kwh'):.1f} kWh/day (capacity {r.get('capacity_kwp'):.1f} kWp "
                f"x specific yield {r.get('specific_yield_kwh_per_kwp'):.4f} kWh/kWp/day).")

    if t == "base_output":
        value = execution_result.get("value")
        return f"Estimated daily specific yield in {city}: {value:.4f} kWh/kWp."

    if t == "weather_conditioned_output":
        r = execution_result.get("result") or {}
        if "total_output_kwh" in r:
            return (f"Estimated output in {city} under the given conditions: {r.get('total_output_kwh'):.1f} kWh/day "
                    f"(capacity {r.get('capacity_kwp'):.1f} kWp x specific yield "
                    f"{r.get('specific_yield_kwh_per_kwp'):.4f} kWh/kWp/day).")
        return f"Estimated specific yield in {city} under the given conditions: {r.get('specific_yield_kwh_per_kwp'):.4f} kWh/kWp/day."

    if t == "rainfall_sensitivity":
        r = execution_result.get("result") or {}
        return (f"Baseline: {r.get('baseline_prediction'):.4f} kWh/kWp/day, "
                f"modified: {r.get('modified_prediction'):.4f} kWh/kWp/day "
                f"({r.get('pct_change'):.1f}% change).")

    if t == "compare_cities":
        ranked = execution_result.get("results", [])
        ordering = ", ".join(
            f"{r['city']} ({r['prediction']:.3f} kWh/kWp/day)" for r in ranked if r.get("prediction") is not None
        )
        return f"Ranked by predicted output (highest first): {ordering}."

    if t == "descriptive_stats":
        r = execution_result.get("result") or {}
        return f"{r.get('stat')} of {r.get('feature')} in {r.get('city')}: {r.get('value'):.3f} (n={r.get('n_observations')})."

    return "Computed a result but couldn't phrase it (no LLM available and no template for this type)."


# ---------------------------------------------------------------------------
# End-to-end entry point
# ---------------------------------------------------------------------------

def answer_question(question, model, features, city_features, feature_ranges, list_cities, weather_raw=None, verbose=True):
    intent_data = parse_question(question, list_cities)
    if verbose:
        print(f"  Parsed intent: {intent_data}")
    execution_result = execute_intent(intent_data, model, features, city_features, feature_ranges, weather_raw=weather_raw)
    if verbose:
        print(f"  Execution result: {execution_result}")
    answer = format_answer(execution_result)
    if execution_result.get("warning"):
        answer += f"\n\n{execution_result['warning']}"
    return answer


# ---------------------------------------------------------------------------
# Main - 10 demo questions
# ---------------------------------------------------------------------------

DEMO_QUESTIONS = [
    "What can you do?",
    "What's my expected daily solar output per unit area in Adelaide?", # city out of coverage
    "What's my expected daily solar output per unit area in Sydney?", # city is one of 27 usable cities
    "If I build a 5-hectare solar farm in Melbourne, what's my expected daily output?",
    "What is my expected yield tomorrow in MountGambier given Sunshine of 9 hours and Cloud9am of 1?",
    "Which would yield more solar power: Sydney, Melbourne, or Brisbane?",
    "How would my expected yield in Albany change if rainfall doubled?",
    "How would my expected yield in Melbourne Airport change if rainfall tripled?",
    "What's a reasonable estimate of solar output for Perth?",
    "How variable is rainfall in AliceSprings?",
    "What fraction of days in Canberra have Cloud9am below 3?",
    "If I build a 1-hectare solar farm in Portland, what's my expected daily output?",
]


def main():
    model, features, target, feature_ranges, city_features, weather_raw, list_cities = load_artifacts()

    for q in DEMO_QUESTIONS:
        print(f"\nQ: {q}")
        start_time = datetime.now()
        start_perf = time.perf_counter()
        answer = answer_question(q, model, features, city_features, feature_ranges, list_cities, weather_raw=weather_raw)
        end_time = datetime.now()
        elapsed = time.perf_counter() - start_perf
        print(f"A: {answer}")
        print(f"  Started: {start_time:%H:%M:%S.%f} | Ended: {end_time:%H:%M:%S.%f} | Elapsed: {elapsed:.3f}s")


if __name__ == "__main__":
    main()