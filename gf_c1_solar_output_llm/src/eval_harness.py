"""
eval_harness.py

Agent evaluation harness for Challenge 1's LLM agent (agent.py), covering
the three evaluation strategies described at
https://langfuse.com/guides/cookbook/example_pydantic_ai_mcp_agent_evaluation :

  1. SINGLE-STEP / WHITE-BOX  (run_white_box_tests)
     Unit tests on individual tool functions (get_city_row, check_extrapolation,
     predict_base_output, predict_farm_output, predict_weather_conditioned_output,
     compute_descriptive_stat) against synthetic, hand-computed fixtures.
     No trained model or LLM required -- pure deterministic assertions. This is
     the layer where "high accuracy" is actually provable.

  2. TRAJECTORY / GLASS-BOX  (intent_matches, used inside run_benchmark)
     Checks whether parse_question() extracted the correct intent + arguments
     from a natural-language question, compared against a hand-labeled
     expected intent per benchmark question. Rule-based dict comparison --
     deterministic, no LLM judge needed.

  3. FINAL RESPONSE / BLACK-BOX  (extract_numeric_result, used inside run_benchmark)
     Checks whether the numeric value inside the agent's final result matches
     what calling the underlying deterministic function directly would produce
     for the same structured input. This is NOT compared against a separately
     curated "ground truth" number -- given the small dataset here, the
     benchmark's job is to confirm the LLM-parsed path didn't corrupt or drop
     information on the way in, not to validate the model's real-world
     accuracy (that's what model.py's LOOCV metrics are for).

Usage:
    python eval_harness.py

Requires solar_model.joblib, feature_ranges.json, city_features_lookup.csv
(see agent.py) to exist for the trajectory/final-response benchmark. The
white-box tests run regardless, since they use synthetic fixtures.

Ollama does not need to be running: agent.py's parse_question() falls back
to rule-based parsing if the LLM server is unreachable, so this harness
still executes end-to-end (with a weaker trajectory-accuracy result,
which is itself a useful thing to report -- LLM-parsed vs. rule-based
fallback accuracy).
"""

from __future__ import annotations

import configparser
import math
import time
from datetime import datetime
from types import SimpleNamespace

import pandas as pd

from agent import (
    HECTARE_TO_M2,
    PANEL_KWP_PER_M2,
    check_extrapolation,
    compare_cities,
    compute_descriptive_stat,
    execute_intent,
    format_answer,
    get_city_row,
    load_artifacts,
    parse_question,
    predict_base_output,
    predict_farm_output,
    predict_rainfall_sensitivity,
    predict_weather_conditioned_output,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

cfg = configparser.ConfigParser()
cfg.read('conf.cfg')
EVAL_RESULTS_OUTPUT_PATH = cfg["eval_harness"]["eval_results_output_path"]

# ---------------------------------------------------------------------------
# Small comparison helpers
# ---------------------------------------------------------------------------


def _approx_equal(a, b, tol=1e-3) -> bool:
    """None-safe, tolerant numeric equality. Falls back to exact equality
    for non-numeric values (e.g. comparing None to None)."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b


def _get_path(d: dict, path: str):
    """Dotted-path lookup into a nested dict, e.g. _get_path(d, 'result.value').
    Returns None if any segment is missing, rather than raising."""
    cur = d
    for key in path.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return None
    return cur


DEFAULT_VALUE_PATHS = {
    "base_output": "value",
    "farm_output": "result.total_output_kwh",
    "weather_conditioned_output": "result.specific_yield_kwh_per_kwp",
    "rainfall_sensitivity": "result.delta",
    "descriptive_stats": "result.value",
}


def extract_numeric_result(exec_result: dict, field_path: str | None = None):
    """
    Pull the single numeric value out of an execute_intent() result that
    represents "the answer" for accuracy-checking purposes. Uses a
    type-appropriate default path unless field_path overrides it (e.g. to
    check 'result.total_output_kwh' instead of 'result.specific_yield_kwh_per_kwp' when a
    question includes a farm size).
    """
    if exec_result.get("error"):
        return None
    path = field_path or DEFAULT_VALUE_PATHS.get(exec_result.get("type"))
    if not path:
        return None
    value = _get_path(exec_result, path)
    return float(value) if isinstance(value, (int, float)) else None


def intent_matches(parsed: dict, expected: dict) -> tuple[bool, list[str]]:
    """
    Trajectory check: does the LLM-parsed intent dict match what we
    expected for this question? Returns (pass, list_of_mismatch_reasons)
    so failures are debuggable, not just a bare True/False.
    """
    mismatches = []

    if parsed.get("intent") != expected.get("intent"):
        mismatches.append(f"intent: expected {expected.get('intent')!r}, got {parsed.get('intent')!r}")
        return False, mismatches

    for key, exp_val in expected.items():
        if key == "intent":
            continue
        act_val = parsed.get(key)

        if key == "city":
            if not (isinstance(act_val, str) and act_val.strip().lower() == str(exp_val).strip().lower()):
                mismatches.append(f"city: expected {exp_val!r}, got {act_val!r}")

        elif key == "cities":
            exp_set = {c.strip().lower() for c in exp_val}
            act_set = {c.strip().lower() for c in (act_val or [])}
            if exp_set != act_set:
                mismatches.append(f"cities: expected {exp_val!r}, got {act_val!r}")

        elif key == "weather":
            act_weather = act_val or {}
            for w_key, w_val in (exp_val or {}).items():
                actual_w = act_weather.get(w_key)
                if not _approx_equal(actual_w, w_val, tol=0.5):
                    mismatches.append(f"weather.{w_key}: expected {w_val!r}, got {actual_w!r}")

        elif key in ("hectares", "multiplier", "threshold"):
            if not _approx_equal(act_val, exp_val, tol=0.5):
                mismatches.append(f"{key}: expected {exp_val!r}, got {act_val!r}")

        elif key in ("feature", "stat"):
            if not (isinstance(act_val, str) and act_val.strip().lower() == str(exp_val).strip().lower()):
                mismatches.append(f"{key}: expected {exp_val!r}, got {act_val!r}")

        else:
            if act_val != exp_val:
                mismatches.append(f"{key}: expected {exp_val!r}, got {act_val!r}")

    return (len(mismatches) == 0), mismatches


# ---------------------------------------------------------------------------
# 1. SINGLE-STEP / WHITE-BOX -- unit tests on synthetic fixtures
# ---------------------------------------------------------------------------


def run_white_box_tests() -> bool:
    """
    Pure unit tests on individual tool functions using hand-built synthetic
    data with known correct answers. No trained model, no LLM, no real data
    files needed -- these should always be runnable and should always pass;
    a failure here means a real bug in the tool logic, not a data or LLM
    issue.
    """
    results: list[tuple[str, bool]] = []

    def check(name: str, condition: bool):
        results.append((name, bool(condition)))

    city_features = pd.DataFrame(
        {
            "Location": ["Sydney", "SydneyAirport", "Melbourne"],
            "Sunshine": [7.0, 6.8, 5.5],
            "Cloud9am": [3, 3, 5],
        }
    )
    features = ["Sunshine", "Cloud9am"]
    feature_ranges = {"Sunshine": {"min": 0, "max": 12}, "Cloud9am": {"min": 0, "max": 9}}

    class FakeModel:
        """Deterministic stand-in: prediction = sum of feature values."""

        def predict(self, X):
            return [float(sum(X[0]))]

    model = FakeModel()

    # -- get_city_row: exact match --
    row, note = get_city_row("Sydney", city_features)
    check("exact city match found", row is not None and row["Location"] == "Sydney")
    check("exact city match has no ambiguity note", note is None)

    # -- get_city_row: ambiguous substring fallback --
    ambiguous_features = pd.DataFrame(
        {"Location": ["PerthAirport", "SydneyAirport"], "Sunshine": [8, 6], "Cloud9am": [2, 3]}
    )
    row2, note2 = get_city_row("Air", ambiguous_features)
    check("ambiguous match still returns a row", row2 is not None)
    check("ambiguous match surfaces an explanatory note", bool(note2) and "matched multiple locations" in note2)

    # -- get_city_row: no match --
    row3, note3 = get_city_row("Atlantis", city_features)
    check("unknown city returns no row", row3 is None)

    # -- check_extrapolation --
    check("in-range value is not flagged", check_extrapolation("Sunshine", 5, feature_ranges) is None)
    check("out-of-range value is flagged", check_extrapolation("Sunshine", 20, feature_ranges) is not None)
    check("unknown feature range is not flagged (can't check what we don't have)",
          check_extrapolation("NotAFeature", 999, feature_ranges) is None)

    # -- predict_base_output / predict_farm_output arithmetic --
    base_pred, base_err, _ = predict_base_output("Sydney", model, features, city_features)
    check("base output computes without error", base_err is None and base_pred is not None)

    farm_result, farm_err, _ = predict_farm_output("Sydney", 2, model, features, city_features)
    expected_capacity = 2 * HECTARE_TO_M2 * PANEL_KWP_PER_M2
    expected_farm = base_pred * expected_capacity
    check("farm output = base_pred * hectares * HECTARE_TO_M2 * PANEL_KWP_PER_M2",
          farm_err is None and abs(farm_result["total_output_kwh"] - expected_farm) < 1e-9)

    # -- predict_weather_conditioned_output: override + default tracking --
    wc_result, wc_err, wc_note = predict_weather_conditioned_output(
        "Sydney", {"Sunshine": 9}, model, features, city_features, feature_ranges
    )
    check("weather override changes the prediction vs. base", wc_err is None and abs(wc_result["specific_yield_kwh_per_kwp"] - base_pred) > 1e-9)
    check("weather override correctly tracks the defaulted feature", wc_result["defaulted_features"] == ["Cloud9am"])
    check("weather override correctly tracks the given feature", wc_result["given_features"] == {"Sunshine": 9})

    # -- predict_weather_conditioned_output: extrapolation flagged in the note --
    wc_result2, wc_err2, wc_note2 = predict_weather_conditioned_output(
        "Sydney", {"Sunshine": 50}, model, features, city_features, feature_ranges
    )
    check("extrapolating weather override is flagged in the note", bool(wc_note2) and "extrapolation" in wc_note2.lower())

    # -- predict_rainfall_sensitivity (uses a Rainfall_mean feature) --
    rain_features_df = pd.DataFrame({"Location": ["Sydney"], "Rainfall_mean": [3.0], "Sunshine": [7.0]})
    rain_features_list = ["Rainfall_mean", "Sunshine"]
    rain_ranges = {"Rainfall_mean": {"min": 0, "max": 10}}
    sens_result, sens_err, sens_note = predict_rainfall_sensitivity(
        "Sydney", 2.0, model, rain_features_list, rain_features_df, rain_ranges
    )
    expected_baseline = 3.0 + 7.0
    expected_modified = 6.0 + 7.0
    check("rainfall sensitivity baseline matches hand calculation",
          sens_err is None and abs(sens_result["baseline_prediction"] - expected_baseline) < 1e-9)
    check("rainfall sensitivity modified value matches hand calculation",
          abs(sens_result["modified_prediction"] - expected_modified) < 1e-9)
    check("rainfall sensitivity delta matches hand calculation",
          abs(sens_result["delta"] - (expected_modified - expected_baseline)) < 1e-9)

    # -- compare_cities ranking --
    two_city_features = pd.DataFrame(
        {"Location": ["Sydney", "Melbourne"], "Sunshine": [7.0, 5.5], "Cloud9am": [3, 1]}
    )
    ranked = compare_cities(["Sydney", "Melbourne"], model, features, two_city_features)
    # FakeModel predicts sum(features): Sydney = 7.0+3 = 10.0, Melbourne = 5.5+1 = 6.5 -> Sydney ranks first
    check("compare_cities ranks the higher-prediction city first", ranked[0]["city"] == "Sydney")

    # -- compute_descriptive_stat --
    weather_raw = pd.DataFrame({"Location": ["Sydney"] * 5, "Rainfall": [0, 0, 5, 10, 0]})
    stat_result, stat_err, _ = compute_descriptive_stat("Sydney", "Rainfall", "percent_above", weather_raw, threshold=1)
    check("descriptive stat percent_above matches hand calculation (2/5 = 40%)",
          stat_err is None and abs(stat_result["value"] - 40.0) < 1e-9)

    stat_result2, stat_err2, _ = compute_descriptive_stat("Sydney", "Rainfall", "mean", None)
    check("descriptive stat with no raw data loaded fails honestly (no guess)",
          stat_result2 is None and stat_err2 is not None)

    passed = sum(1 for _, ok in results if ok)
    print("1. SINGLE-STEP / WHITE-BOX tests (synthetic fixtures, no trained model needed):")
    for name, ok in results:
        print(f"   [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"   -> {passed}/{len(results)} passed\n")
    return passed == len(results)


# ---------------------------------------------------------------------------
# Benchmark questions for the trajectory + final-response layers
# ---------------------------------------------------------------------------
#
# City names below assume a representative set of Australian cities/towns
# (Sydney, Melbourne, Adelaide, Perth, PerthAirport, Albany, Cairns,
# Brisbane, Darwin). There's no separate cities.py lookup anymore -- the
# only source of truth for valid cities is city_features['Location'] from
# your trained dataset. If your final trained dataset dropped one of these
# (merge mismatch, missing raster coverage, etc.), the corresponding case's
# expected_value_fn will fail to compute a ground truth and that case is
# reported as SKIPPED rather than FAILED -- it's telling you about data
# coverage, not agent correctness.

CITY_A, CITY_B, CITY_C, CITY_D = "Sydney", "Melbourne", "AliceSprings", "Perth"
CITY_E, CITY_F, CITY_G, CITY_H = "Albany", "Cairns", "Brisbane", "Darwin"


def _first(triple):
    """Unwrap a (value, error, note) tuple -> value, or None if there was an error."""
    value, error, _ = triple
    return None if error else value


def _result_field(triple, field):
    """Unwrap a (result_dict, error, note) tuple and pull one field, or None on error."""
    result, error, _ = triple
    return None if (error or result is None) else result.get(field)


BENCHMARK = [
    # --- Happy path: one case per intent type ---
    {
        "question": f"What's my expected daily solar output per unit area in {CITY_A}?",
        "expected_intent": {"intent": "base_or_farm_output", "city": CITY_A, "hectares": None},
        "expected_value_fn": lambda ctx: _first(
            predict_base_output(CITY_A, ctx.model, ctx.features, ctx.city_features)
        ),
    },
    {
        "question": f"If I build a 5-hectare solar farm in {CITY_B}, what's my expected daily output?",
        "expected_intent": {"intent": "base_or_farm_output", "city": CITY_B, "hectares": 5},
        "expected_value_fn": lambda ctx: _result_field(
            predict_farm_output(CITY_B, 5, ctx.model, ctx.features, ctx.city_features), "total_output_kwh"
        ),
    },
    {
        "question": f"What is my expected yield tomorrow in {CITY_C} given Sunshine of 9 hours and Cloud9am of 1?",
        "expected_intent": {"intent": "weather_conditioned_output", "city": CITY_C,
                             "weather": {"Sunshine": 9, "Cloud9am": 1}},
        "expected_value_fn": lambda ctx: _result_field(
            predict_weather_conditioned_output(
                CITY_C, {"Sunshine": 9, "Cloud9am": 1}, ctx.model, ctx.features, ctx.city_features, ctx.feature_ranges
            ),
            "specific_yield_kwh_per_kwp",
        ),
        # True only if the model has exactly these 2 trained features. With more
        # features, any not supplied here will default and trigger a warning note
        # -- that's correct, expected behavior (see predict_weather_conditioned_output),
        # so set this to True once you know your model's real feature count.
        "expects_warning": True,
    },
    {
        "question": f"How would my expected yield in {CITY_E} change if rainfall doubled?",
        "expected_intent": {"intent": "rainfall_sensitivity", "city": CITY_E, "multiplier": 2.0},
        "expected_value_fn": lambda ctx: _result_field(
            predict_rainfall_sensitivity(CITY_E, 2.0, ctx.model, ctx.features, ctx.city_features, ctx.feature_ranges),
            "delta",
        ),
    },
    {
        "question": f"Which would yield more solar power: {CITY_A}, {CITY_B}, or {CITY_G}?",
        "expected_intent": {"intent": "compare_cities", "cities": [CITY_A, CITY_B, CITY_G]},
        # No single numeric value to check here -- trajectory correctness is what matters.
    },
    {
        "question": f"How variable is rainfall in {CITY_F}?",
        "expected_intent": {"intent": "descriptive_stats", "city": CITY_F, "feature": "Rainfall", "stat": "std"},
        # NOTE: descriptive_stats operates on RAW daily columns (e.g. "Rainfall"),
        # while weather_conditioned_output/rainfall_sensitivity operate on the
        # model's AGGREGATED feature names (e.g. "Rainfall_mean" in agent_v2's
        # convention). These vocabularies can differ -- if your real
        # city_features/model use "Rainfall_mean" but your raw weatherAUS.csv
        # column is "Rainfall", update KNOWN_WEATHER_FEATURES / the parser
        # prompt and this case's "feature" to whatever your raw CSV actually
        # calls it, not the trained-model feature name.
        "expected_value_fn": lambda ctx: _result_field(
            compute_descriptive_stat(CITY_F, "Rainfall", "std", ctx.weather_raw), "value"
        ),
    },

    # --- Edge cases: missing weather / defaults ---
    {
        "question": f"What is my expected yield tomorrow in {CITY_D} given Sunshine of 10 hours?",
        "expected_intent": {"intent": "weather_conditioned_output", "city": CITY_D, "weather": {"Sunshine": 10}},
        "expected_value_fn": lambda ctx: _result_field(
            predict_weather_conditioned_output(
                CITY_D, {"Sunshine": 10}, ctx.model, ctx.features, ctx.city_features, ctx.feature_ranges
            ),
            "specific_yield_kwh_per_kwp",
        ),
        "expects_warning": True,  # should flag that other features were defaulted
    },

    # --- Edge case: extrapolation should be flagged ---
    {
        "question": f"What's my output in {CITY_D} if Sunshine is 40 hours?",  # physically impossible -> extrapolation
        "expected_intent": {"intent": "weather_conditioned_output", "city": CITY_D, "weather": {"Sunshine": 40}},
        "expects_warning": True,
    },

    # --- Edge case: ambiguous / airport-style city name ---
    {
        "question": "What's my expected output for PerthAirport?",
        "expected_intent": {"intent": "base_or_farm_output", "city": "PerthAirport", "hectares": None},
        "expected_value_fn": lambda ctx: _first(
            predict_base_output("PerthAirport", ctx.model, ctx.features, ctx.city_features)
        ),
    },

    # --- Edge case: unknown city ---
    # NOTE: the rule-based fallback parser can only extract city names that
    # appear in its known-city list, so it will mis-classify this as
    # 'unknown' intent rather than 'base_or_farm_output' with an unrecognized
    # city. An LLM parser doesn't have this limitation -- it extracts
    # "Atlantis" as a city name regardless, and execute_intent()'s
    # get_city_row() then correctly reports "no data available". Expect this
    # case to FAIL under the rule-based fallback and PASS once Ollama is
    # running -- that gap is itself useful to report as a fallback limitation.
    {
        "question": "What's my expected solar output in Atlantis?",
        "expected_intent": {"intent": "base_or_farm_output", "city": "Atlantis", "hectares": None},
        "expects_error": True,
    },

    # --- Edge case: descriptive stat with a threshold ---
    {
        "question": f"What fraction of days in {CITY_F} have Cloud9am below 3?",
        "expected_intent": {"intent": "descriptive_stats", "city": CITY_F, "feature": "Cloud9am",
                             "stat": "percent_below", "threshold": 3},
        "expected_value_fn": lambda ctx: _result_field(
            compute_descriptive_stat(CITY_F, "Cloud9am", "percent_below", ctx.weather_raw, threshold=3), "value"
        ),
    },

    # --- Edge case: farm output combined with weather conditions ---
    {
        "question": f"For a 10-hectare farm in {CITY_H} with Sunshine at 8 hours, what's my expected output?",
        "expected_intent": {"intent": "weather_conditioned_output", "city": CITY_H, "hectares": 10,
                             "weather": {"Sunshine": 8}},
        "expected_value_fn": lambda ctx: _result_field(
            predict_weather_conditioned_output(
                CITY_H, {"Sunshine": 8}, ctx.model, ctx.features, ctx.city_features, ctx.feature_ranges, hectares=10
            ),
            "total_output_kwh",
        ),
        "value_field": "result.total_output_kwh",
    },

    # --- Edge case: multiplier phrased in words, not digits ---
    {
        "question": f"What if rainfall in {CITY_C} tripled -- how would that affect output?",
        "expected_intent": {"intent": "rainfall_sensitivity", "city": CITY_C, "multiplier": 3.0},
        "expected_value_fn": lambda ctx: _result_field(
            predict_rainfall_sensitivity(CITY_C, 3.0, ctx.model, ctx.features, ctx.city_features, ctx.feature_ranges),
            "delta",
        ),
    },

    # --- Edge case: comparison with an invalid city mixed in ---
    # Same rule-based-fallback limitation as the Atlantis case above: only
    # 1 of the 2 named cities is recognized by keyword matching, so the
    # fallback won't even reach the 'compare' branch (needs >=2 known
    # cities). Expected to fail under fallback, pass with a real LLM.
    {
        "question": f"Compare solar output between {CITY_A} and Atlantis.",
        "expected_intent": {"intent": "compare_cities", "cities": [CITY_A, "Atlantis"]},
    },

    # --- Edge case: paraphrased/indirect version of a base-output question ---
    {
        "question": f"Roughly how much solar could I generate per square meter in {CITY_A} on a typical day?",
        "expected_intent": {"intent": "base_or_farm_output", "city": CITY_A, "hectares": None},
        "expected_value_fn": lambda ctx: _first(
            predict_base_output(CITY_A, ctx.model, ctx.features, ctx.city_features)
        ),
    },

    # --- Edge case: descriptive stat, percentile phrasing ---
    {
        "question": f"What's a high-end (90th percentile) Sunshine day look like in {CITY_G}?",
        "expected_intent": {"intent": "descriptive_stats", "city": CITY_G, "feature": "Sunshine", "stat": "p90"},
        "expected_value_fn": lambda ctx: _result_field(
            compute_descriptive_stat(CITY_G, "Sunshine", "p90", ctx.weather_raw), "value"
        ),
    },

    # --- Edge case: true unknown / out of scope ---
    {
        "question": "What's the capital of Australia?",
        "expected_intent": {"intent": "unknown"},
    },
    {
        "question": "Can you recommend a good solar panel brand to buy?",
        "expected_intent": {"intent": "unknown"},
    },

    # --- Repeat of a happy-path question with different phrasing (parser robustness) ---
    {
        "question": f"I'm thinking about a 5 hectare farm near {CITY_B} -- what daily output should I expect?",
        "expected_intent": {"intent": "base_or_farm_output", "city": CITY_B, "hectares": 5},
        "expected_value_fn": lambda ctx: _result_field(
            predict_farm_output(CITY_B, 5, ctx.model, ctx.features, ctx.city_features), "total_output_kwh"
        ),
    },

    # --- Edge case: farm output for a city with no data ---
    # Same rule-based-fallback limitation as the Atlantis cases above.
    {
        "question": "What's my output for a 3-hectare farm in Nowhereville?",
        "expected_intent": {"intent": "base_or_farm_output", "city": "Nowhereville", "hectares": 3},
        "expects_error": True,
    },
]


# ---------------------------------------------------------------------------
# 2 & 3. Run TRAJECTORY + FINAL-RESPONSE checks together (they share one
#         agent call per question, so it's cheaper to evaluate them in the
#         same pass rather than re-running the agent twice)
# ---------------------------------------------------------------------------


def run_benchmark(benchmark, model, features, city_features, feature_ranges, weather_raw, list_cities, verbose=True):
    ctx = SimpleNamespace(
        model=model, features=features, city_features=city_features,
        feature_ranges=feature_ranges, weather_raw=weather_raw,
    )
    rows = []

    for case in benchmark:
        question = case["question"]
        start_time = datetime.now()
        start_perf = time.perf_counter()

        parsed = parse_question(question, list_cities)
        trajectory_pass, mismatches = intent_matches(parsed, case["expected_intent"])

        exec_result = execute_intent(parsed, model, features, city_features, feature_ranges, weather_raw=weather_raw)
        answer = format_answer(exec_result)
        if exec_result.get("warning"):
            answer += f"\n\n{exec_result['warning']}"

        end_time = datetime.now()
        elapsed_s = time.perf_counter() - start_perf

        status = "PASS"
        expected_value = None
        actual_value = None
        value_pass = None
        warning_pass = None

        if case.get("expects_error"):
            value_pass = bool(exec_result.get("error"))
            if not value_pass:
                status = "FAIL"

        elif case.get("expected_value_fn"):
            try:
                expected_value = case["expected_value_fn"](ctx)
            except Exception:  # noqa: BLE001 -- treat any computation failure as "can't establish ground truth"
                expected_value = None

            if expected_value is None:
                status = "SKIP"  # data not available for this case (e.g. city missing from dataset)
            else:
                actual_value = extract_numeric_result(exec_result, field_path=case.get("value_field"))
                value_pass = _approx_equal(actual_value, expected_value, tol=case.get("tol", 1e-3))
                if not value_pass:
                    status = "FAIL"

        if "expects_warning" in case and status != "SKIP":
            has_warning = bool(exec_result.get("warning"))
            warning_pass = has_warning == case["expects_warning"]
            if not warning_pass:
                status = "FAIL"

        if status not in ("SKIP",) and not trajectory_pass:
            status = "FAIL"

        rows.append(
            {
                "question": question,
                "expected_intent": case["expected_intent"].get("intent"),
                "actual_intent": parsed.get("intent"),
                "trajectory_pass": trajectory_pass,
                "trajectory_notes": "; ".join(mismatches) if mismatches else "",
                "expected_value": expected_value,
                "actual_value": actual_value,
                "value_pass": value_pass,
                "warning_pass": warning_pass,
                "status": status,
                "answer": answer,
                "started_at": start_time.strftime("%H:%M:%S.%f"),
                "ended_at": end_time.strftime("%H:%M:%S.%f"),
                "elapsed_s": elapsed_s,
            }
        )

        if verbose:
            print(f"[{status}] {question}")
            print(f"    started: {start_time:%H:%M:%S.%f} | ended: {end_time:%H:%M:%S.%f} | elapsed: {elapsed_s:.3f}s")

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame):
    total = len(df)
    scored = df[df["status"] != "SKIP"]
    n_skip = total - len(scored)
    n_pass = (scored["status"] == "PASS").sum()

    traj_acc = scored["trajectory_pass"].mean() * 100 if len(scored) else float("nan")

    value_scored = scored[scored["value_pass"].notna()]
    value_acc = value_scored["value_pass"].mean() * 100 if len(value_scored) else float("nan")

    warn_scored = scored[scored["warning_pass"].notna()]
    warn_acc = warn_scored["warning_pass"].mean() * 100 if len(warn_scored) else float("nan")

    print("\n" + "=" * 78)
    print(f"BENCHMARK SUMMARY -- {total} cases total, {n_skip} skipped (data unavailable)")
    if len(scored):
        print(f"  Overall pass rate (scored cases):  {n_pass}/{len(scored)} ({n_pass / len(scored) * 100:.1f}%)")
        print(f"  2. Trajectory accuracy (glass-box): {traj_acc:.1f}%  -- correct intent + args extracted")
        if not math.isnan(value_acc):
            print(f"  3. Final-value accuracy (black-box): {value_acc:.1f}%  -- matches deterministic recomputation")
        if not math.isnan(warn_acc):
            print(f"     Warning-flag accuracy:            {warn_acc:.1f}%  -- correctly flagged defaults/extrapolation")
    else:
        print("  No scoreable cases -- check that your dataset covers the cities used in BENCHMARK.")
    print("=" * 78)

    fails = df[df["status"] == "FAIL"]
    if not fails.empty:
        print("\nFailures:")
        for _, row in fails.iterrows():
            print(f"- {row['question']}")
            if row["trajectory_notes"]:
                print(f"    trajectory: {row['trajectory_notes']}")
            if row["value_pass"] is False:
                print(f"    value: expected {row['expected_value']}, got {row['actual_value']}")
            if row["warning_pass"] is False:
                print(f"    warning flag mismatch")

    skips = df[df["status"] == "SKIP"]
    if not skips.empty:
        print("\nSkipped (couldn't establish ground truth -- usually means a city/feature isn't in your dataset):")
        for _, row in skips.iterrows():
            print(f"- {row['question']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    white_box_ok = run_white_box_tests()
    if not white_box_ok:
        print("Some white-box tests failed -- fix these before trusting the trajectory/final-response "
              "results below, since they exercise the same underlying functions.\n")

    print("Loading trained pipeline artifacts for trajectory + final-response evaluation...")
    try:
        model, features, target, feature_ranges, city_features, weather_raw, list_cities = load_artifacts()
    except FileNotFoundError as exc:
        print(f"Could not load artifacts ({exc}).")
        print("Run data_prep.py and model.py first (see README.md), then re-run this harness.")
        return

    if weather_raw is None:
        print("Note: no raw weather file found at WEATHER_RAW_PATH -- descriptive_stats cases will be SKIPPED.\n")

    df = run_benchmark(BENCHMARK, model, features, city_features, feature_ranges, weather_raw, list_cities, verbose=True)
    summarize(df)

    df.to_csv(EVAL_RESULTS_OUTPUT_PATH, index=False)
    print(f"\nFull results written to {EVAL_RESULTS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()