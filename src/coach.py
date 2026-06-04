"""
VocalCoach LLM Critique — Phase 3
==================================

Converts structured coaching metrics (from features.py + model inference) into
natural-language feedback via the Claude API. Two output tiers:

  expert  — numerical metrics, technical terminology, specific targets
  beginner — encouragement-first, plain language, actionable tips

Usage
-----
    from vocalcoach.coach import build_report, generate_critique

    report = build_report(phrases, feats_summary, technique_clip, dtw_result)
    critique = generate_critique(report, mode="expert")   # or "beginner"
    print(critique)

Dependencies: anthropic (pip install anthropic)
"""

import json
import os

try:
    import anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

_TECHNIQUE_NAMES = ["vibrato", "breathy", "falsetto", "belt", "straight"]

# ── Prompt templates ────────────────────────────────────────────────────────

_SYSTEM_EXPERT = """You are an expert vocal coach with training in classical and
contemporary technique. Your feedback is precise, uses technical terminology, and
cites specific numeric values. You reference established pedagogical targets where
relevant (e.g. vibrato rate 5.5-6.5 Hz, depth 50-100 cents, HNR >12 dB for clean
phonation). Be concise — no more than 250 words. Structure your response with these
sections: Pitch, Vibrato, Dynamics, Technique, Next Steps."""

_SYSTEM_BEGINNER = """You are a warm and encouraging vocal coach helping a student
improve. Use plain language, avoid jargon, and always lead with something positive
before suggesting an improvement. Be concise — no more than 200 words. Keep the tone
conversational and motivating. Structure: What went well, What to work on, One tip."""

_USER_TEMPLATE = """Here is the coaching analysis for this singing clip:

```json
{report_json}
```

Please provide {mode_label} vocal feedback based on these metrics."""


def build_report(phrases, feats_summary, technique_clip=None, dtw_result=None,
                 clip_duration_s=None):
    """Assemble a structured coaching report dict from analysis outputs.

    Args:
        phrases:        list of phrase dicts from phrase_aggregate()
        feats_summary:  dict from summarise() — clip-level acoustic scalars
        technique_clip: dict {technique_name: float} — clip-level technique means
                        (e.g. from sigmoid outputs averaged over voiced frames)
        dtw_result:     dict from compute_dtw_distance(), or None
        clip_duration_s: total clip duration in seconds, or None

    Returns:
        dict — structured coaching report, JSON-serialisable
    """
    vib_phrases = [p for p in phrases if p.get("vibrato", {}).get("has_vibrato")]

    # Clip-level F0 mean from voiced phrase means (weighted by duration)
    voiced_phrases = [p for p in phrases if p["f0_mean_hz"] == p["f0_mean_hz"] and p["f0_mean_hz"]]
    if voiced_phrases:
        total_dur = sum(p["duration_s"] for p in voiced_phrases)
        f0_mean_clip = sum(p["f0_mean_hz"] * p["duration_s"] for p in voiced_phrases) / total_dur
        # Pitch stability: mean of per-phrase F0 std in Hz
        f0_std_vals = [p["f0_std_hz"] for p in voiced_phrases if p["f0_std_hz"] == p["f0_std_hz"]]
        f0_std_mean = float(sum(f0_std_vals) / len(f0_std_vals)) if f0_std_vals else float("nan")
    else:
        f0_mean_clip = float("nan")
        f0_std_mean  = float("nan")

    report = {
        "clip_duration_s": clip_duration_s,
        "n_phrases": len(phrases),
        "pitch": {
            "f0_mean_hz":         round(f0_mean_clip, 1),
            "f0_stability_std_hz": round(f0_std_mean, 2),
        },
        "dynamics": {
            "rms_mean_db":  round(feats_summary.get("rms_mean_db", float("nan")), 1),
            "rms_range_db": round(feats_summary.get("rms_range_db", float("nan")), 1),
        },
        "voice_quality": {
            "hnr_mean_db":       round(feats_summary.get("hnr_mean_db", float("nan")), 1),
            "jitter_mean_pct":   round(feats_summary.get("jitter_mean_pct", float("nan")), 3),
            "shimmer_mean_pct":  round(feats_summary.get("shimmer_mean_pct", float("nan")), 3),
            "h1_h2_mean_db":     round(feats_summary.get("h1_h2_mean_db", float("nan")), 1),
        },
        "vibrato": {
            "phrase_fraction":    round(feats_summary.get("vibrato_phrase_frac", float("nan")), 2),
            "rate_hz_mean":       round(feats_summary.get("vibrato_rate_hz_mean", float("nan")), 2),
            "depth_cents_mean":   round(feats_summary.get("vibrato_depth_cents_mean", float("nan")), 1),
            "n_vibrato_phrases":  len(vib_phrases),
        },
        "technique": technique_clip or {},
        "breath_control": {
            "n_breaths":  feats_summary.get("n_breaths"),
            "n_onsets":   feats_summary.get("n_onsets"),
        },
    }

    if dtw_result is not None:
        report["reference_comparison"] = {
            "dtw_distance_cents":       round(dtw_result.get("dtw_distance", float("nan")), 2),
            "mean_deviation_cents":     round(dtw_result.get("mean_deviation_cents", float("nan")), 1),
            "max_deviation_cents":      round(dtw_result.get("max_deviation_cents", float("nan")), 1),
        }

    hop_s = 160 / 16000  # 10 ms — matches HOP_LENGTH / SR

    # Per-phrase detail (abbreviated for LLM context)
    report["phrases"] = [
        {
            "i": i + 1,
            "start_s":     round(p["start_frame"] * hop_s, 2),
            "end_s":       round(p["end_frame"]   * hop_s, 2),
            "duration_s":  round(p["duration_s"], 2),
            "f0_mean_hz":  round(p["f0_mean_hz"], 1) if p["f0_mean_hz"] == p["f0_mean_hz"] else None,
            "vibrato":     p["vibrato"]["has_vibrato"],
            "vib_rate_hz": round(p["vibrato"]["rate_hz"], 2) if p["vibrato"]["rate_hz"] else None,
            "vib_depth":   round(p["vibrato"]["depth_cents"], 1) if p["vibrato"]["depth_cents"] else None,
            "rms_arc_db":  round(p["rms_arc_db"], 1) if p.get("rms_arc_db") is not None else None,
            "technique":   {k: round(v, 2) for k, v in (p.get("technique_means") or {}).items()},
        }
        for i, p in enumerate(phrases)
    ]

    return report


def score_report(report):
    """Convert a coaching report into scored, actionable observations.

    No LLM required. Each observation has:
        axis        — coaching dimension (Pitch / Vibrato / Dynamics / Technique /
                      Voice Quality / Breath / Reference)
        metric      — which measurement was used
        value       — the raw value with unit
        rating      — "good" | "acceptable" | "needs_work"
        observation — one sentence describing what was found
        action      — one concrete practice instruction

    Returns:
        dict with keys:
            overall_score  — 0-100 weighted score
            ratings        — counts of good/acceptable/needs_work
            observations   — list of observation dicts (sorted by rating)
            summary        — 2-sentence plain text summary
    """
    obs = []

    def _add(axis, metric, value, unit, rating, observation, action):
        obs.append({
            "axis": axis,
            "metric": metric,
            "value": f"{value} {unit}".strip(),
            "rating": rating,
            "observation": observation,
            "action": action,
        })

    # ── Pitch ──────────────────────────────────────────────────────────────
    f0 = report.get("pitch", {}).get("f0_mean_hz")
    if f0 and f0 == f0:  # not nan
        import librosa as _librosa
        note = _librosa.hz_to_note(f0)
        _add("Pitch", "f0_mean_hz", round(f0, 1), "Hz",
             "good",
             f"Mean singing pitch is {round(f0, 1)} Hz ({note}) — cleanly detected across voiced phrases.",
             "Note your comfortable tessitura; aim to anchor long notes at this pitch centre.")

    f0_std = report.get("pitch", {}).get("f0_stability_std_hz")
    if f0_std and f0_std == f0_std and f0:
        # Express stability as cents (more musical than Hz)
        cents_std = round(1200 * abs(__import__("math").log2((f0 + f0_std) / f0)), 1)
        if cents_std < 20:
            rating, obs_txt, act_txt = (
                "good",
                f"Pitch is very stable (±{cents_std} cents std) — excellent intonation control.",
                "Maintain this stability when adding dynamics or vibrato.")
        elif cents_std < 50:
            rating, obs_txt, act_txt = (
                "acceptable",
                f"Pitch stability is moderate (±{cents_std} cents std) — minor wavering on sustained notes.",
                "Practise long-tone exercises on a drone; target ±15 cents across 4-beat holds.")
        else:
            rating, obs_txt, act_txt = (
                "needs_work",
                f"Pitch instability detected (±{cents_std} cents std) — significant wavering or drift.",
                "Sing sustained vowels on a single pitch with a tuner; stop and reset when deviation >30 cents.")
        _add("Pitch", "f0_stability", cents_std, "cents std", rating, obs_txt, act_txt)

    # ── Vibrato ─────────────────────────────────────────────────────────────
    vib = report.get("vibrato", {})
    n_phrases = report.get("n_phrases", 0)
    vib_frac = vib.get("phrase_fraction", 0) or 0
    rate = vib.get("rate_hz_mean")
    depth = vib.get("depth_cents_mean")

    if vib.get("n_vibrato_phrases", 0) == 0:
        _add("Vibrato", "phrase_fraction", 0, "phrases",
             "needs_work" if n_phrases >= 2 else "acceptable",
             "No vibrato detected across any phrase — singing appears straight-tone throughout.",
             "On held notes (>1.5s), relax the larynx and allow a natural 5-6 Hz oscillation; "
             "start with a gentle diaphragmatic pulse to initiate the wave.")
    else:
        # Rate
        if rate and 5.0 <= rate <= 7.0:
            rate_rating, rate_obs = "good", f"Vibrato rate {round(rate, 2)} Hz is in the classical target range (5-7 Hz)."
        elif rate and 4.0 <= rate < 5.0:
            rate_rating, rate_obs = "acceptable", f"Vibrato rate {round(rate, 2)} Hz is slightly slow (target 5-7 Hz) — sounds calm but can lack warmth."
        elif rate and 7.0 < rate <= 8.5:
            rate_rating, rate_obs = "acceptable", f"Vibrato rate {round(rate, 2)} Hz is slightly fast (target 5-7 Hz) — may sound nervous or pressed."
        else:
            rate_rating, rate_obs = "needs_work", f"Vibrato rate {round(rate, 2) if rate else '?'} Hz is outside the 5-7 Hz target."
        _add("Vibrato", "rate_hz", round(rate, 2) if rate else None, "Hz",
             rate_rating, rate_obs,
             "Slow vibrato: increase breath support and relax the jaw. "
             "Fast vibrato: reduce subglottal pressure slightly and open the soft palate.")

        # Depth
        if depth and 40 <= depth <= 120:
            depth_rating, depth_obs = "good", f"Vibrato depth {round(depth, 1)} cents is within the classical range (40-120 cents)."
        elif depth and depth < 40:
            depth_rating, depth_obs = "acceptable", f"Vibrato depth {round(depth, 1)} cents is narrow (<40 cents) — sounds restrained."
        else:
            depth_rating, depth_obs = "needs_work", f"Vibrato depth {round(depth, 1)} cents is excessive (>120 cents) — pitch centre sounds unstable."
        _add("Vibrato", "depth_cents", round(depth, 1) if depth else None, "cents",
             depth_rating, depth_obs,
             "Narrow depth: increase resonance space (raise soft palate) and allow larynx to float freely. "
             "Wide depth: focus on a stable pitch centre; think 'spin' not 'wobble'.")

        # Coverage
        if vib_frac >= 0.7:
            cov_rating, cov_obs = "good", f"Vibrato present in {round(vib_frac*100)}% of phrases — consistent use."
        elif vib_frac >= 0.4:
            cov_rating, cov_obs = "acceptable", f"Vibrato present in {round(vib_frac*100)}% of phrases — used selectively."
        else:
            cov_rating, cov_obs = "needs_work", f"Vibrato only in {round(vib_frac*100)}% of phrases — inconsistent deployment."
        _add("Vibrato", "phrase_coverage", round(vib_frac, 2), "fraction",
             cov_rating, cov_obs,
             "Practise adding vibrato on the final beat of every long phrase as a default; "
             "then selectively remove it for stylistic straight-tone passages.")

    # ── Dynamics ────────────────────────────────────────────────────────────
    phrases = report.get("phrases", [])
    arcs = [p["rms_arc_db"] for p in phrases if p.get("rms_arc_db") is not None]
    if arcs:
        mean_arc = round(sum(arcs) / len(arcs), 1)
        if mean_arc >= 6:
            arc_r, arc_o, arc_a = (
                "good",
                f"Average dynamic arc {mean_arc} dB per phrase — good energy shaping across phrases.",
                "Exaggerate the arc further on climactic phrases; aim for 8-10 dB peak-to-tail difference.")
        elif mean_arc >= 3:
            arc_r, arc_o, arc_a = (
                "acceptable",
                f"Dynamic arc {mean_arc} dB per phrase — present but understated.",
                "Crescendo into the mid-phrase peak more intentionally; use diaphragmatic engagement "
                "to drive the louder portion rather than pushing from the throat.")
        else:
            arc_r, arc_o, arc_a = (
                "needs_work",
                f"Dynamic arc only {mean_arc} dB per phrase — phrases sound flat in energy.",
                "Mark phrase peaks in the score. Practise each phrase mp→mf→mp, then scale up. "
                "A 5 dB arc is a minimum for expressive singing.")
        _add("Dynamics", "rms_arc_db", mean_arc, "dB", arc_r, arc_o, arc_a)

    # ── Technique ───────────────────────────────────────────────────────────
    tech = report.get("technique", {})
    TECH_THRESHOLDS = {
        "vibrato":  (0.35, 0.55),   # (warn, strong)
        "breathy":  (0.35, 0.55),
        "falsetto": (0.25, 0.45),
        "belt":     (0.30, 0.50),
    }
    TECH_TIPS = {
        "vibrato":  ("Vibrato detected at frame level — consistent with musical context.",
                     "Vibrato very strong throughout — ensure it is intentional and not a tension artefact. "
                     "Try alternating straight-tone and vibrato phrases."),
        "breathy":  ("Breathy phonation detected — adds softness but reduces projection.",
                     "Heavy breathiness detected — check for glottal gap or excess air flow. "
                     "Practise 'ng' onset exercises to encourage adduction before vowels."),
        "falsetto": ("Falsetto register detected — check if intentional.",
                     "Heavy falsetto detected — if unintended, work on mixed voice or passaggio exercises "
                     "at the F4-G4 break (for tenor) to blend registers."),
        "belt":     ("Belt/chest mix detected — strong projection.",
                     "Very strong belt detected — monitor for laryngeal tension above the passaggio. "
                     "Ensure jaw is free and tongue root is not pulled back."),
    }
    for tname, prob in tech.items():
        if prob is None or prob != prob:
            continue
        warn_t, strong_t = TECH_THRESHOLDS.get(tname, (0.35, 0.55))
        if prob < warn_t:
            continue  # not significant enough to flag
        tip_mod, tip_strong = TECH_TIPS.get(tname, ("Technique detected.", "Technique very strong."))
        if prob >= strong_t:
            t_rating, t_obs, t_act = (
                "needs_work" if tname in ("breathy", "falsetto") else "acceptable",
                f"{tname.capitalize()} probability {round(prob, 2)} (strong signal).",
                tip_strong)
        else:
            t_rating, t_obs, t_act = (
                "acceptable",
                f"{tname.capitalize()} probability {round(prob, 2)} (moderate signal).",
                tip_mod)
        _add("Technique", tname, round(prob, 2), "prob", t_rating, t_obs, t_act)

    # Per-phrase technique flags
    for p in phrases:
        pmeans = p.get("technique") or {}
        for tname, prob in pmeans.items():
            if prob is None or prob != prob:
                continue
            _, strong_t = TECH_THRESHOLDS.get(tname, (0.35, 0.55))
            if prob >= strong_t:
                tip_mod, tip_strong = TECH_TIPS.get(tname, ("", ""))
                _add("Technique", f"{tname}_phrase_{p['i']}",
                     round(prob, 2), "prob", "acceptable",
                     f"Phrase {p['i']} ({p['start_s']}-{p['end_s']}s): "
                     f"{tname} confidence {round(prob, 2)} — notable in this phrase.",
                     tip_strong if prob > strong_t + 0.1 else tip_mod)

    # ── Breath ──────────────────────────────────────────────────────────────
    bc = report.get("breath_control", {})
    n_onsets = bc.get("n_onsets", 0) or 0
    n_ph = report.get("n_phrases", 1) or 1
    if n_onsets > 0:
        onsets_per_phrase = round(n_onsets / n_ph, 1)
        if onsets_per_phrase <= 3:
            b_r, b_o, b_a = (
                "good",
                f"{n_onsets} onsets across {n_ph} phrases ({onsets_per_phrase} per phrase) — legato line maintained.",
                "Continue prioritising legato; use breath marks intentionally rather than between every note.")
        elif onsets_per_phrase <= 6:
            b_r, b_o, b_a = (
                "acceptable",
                f"{n_onsets} onsets across {n_ph} phrases ({onsets_per_phrase} per phrase) — some phrase fragmentation.",
                "Identify which gaps are intentional breaths vs micro-breaks from support drop. "
                "Mark only musical breath points and sustain through the rest.")
        else:
            b_r, b_o, b_a = (
                "needs_work",
                f"{n_onsets} onsets across {n_ph} phrases ({onsets_per_phrase} per phrase) — significant phrase fragmentation.",
                "Practise messa di voce on each phrase to build breath reservoir. "
                "Target completing the full phrase arc on a single breath.")
        _add("Breath", "onsets_per_phrase", onsets_per_phrase, "per phrase", b_r, b_o, b_a)

    # Axis priority weights — defined here so MOS block below can extend it
    AXIS_W = {"Pitch": 2.0, "Vibrato": 1.5, "Dynamics": 1.0,
               "Technique": 1.0, "Breath": 1.0, "Reference": 2.0,
               "Voice Quality": 0.5, "Perceptual Quality": 1.5}

    # ── MOS (SingMOS perceptual quality) ────────────────────────────────────
    mos_block = report.get("mos", {})
    mos = mos_block.get("score") if mos_block else None
    if mos is not None:
        # MOS 1-5 → coaching bands
        if mos >= 4.2:
            m_r, m_o, m_a = (
                "good",
                f"Perceptual quality MOS {mos:.2f}/5 — listener-rated excellent.",
                "Maintain this quality; focus refinements on musical expression rather than tone.")
        elif mos >= 3.5:
            m_r, m_o, m_a = (
                "acceptable",
                f"Perceptual quality MOS {mos:.2f}/5 — good but room to improve tonal clarity.",
                "Work on breath support consistency to reduce micro-fluctuations that lower perceived quality.")
        elif mos >= 2.8:
            m_r, m_o, m_a = (
                "needs_work",
                f"Perceptual quality MOS {mos:.2f}/5 — listeners perceive noticeable quality issues.",
                "Focus on resonant placement (forward/mask resonance) and reduce pressed phonation. "
                "Record in a drier acoustic environment to separate room from voice quality.")
        else:
            m_r, m_o, m_a = (
                "needs_work",
                f"Perceptual quality MOS {mos:.2f}/5 — significant quality issues detected.",
                "Address fundamental phonation issues: check for vocal fatigue, tension, or recording noise. "
                "A MOS below 2.5 often reflects acoustic environment as much as voice quality.")
        _add("Perceptual Quality", "mos_score", mos, "/5", m_r, m_o, m_a)

    # ── Reference DTW ───────────────────────────────────────────────────────
    ref = report.get("reference_comparison")
    if ref:
        dev = ref.get("mean_deviation_cents")
        max_dev = ref.get("max_deviation_cents")
        if dev is not None and dev == dev:
            if dev < 25:
                r_r, r_o, r_a = (
                    "good",
                    f"Mean pitch deviation from reference: {round(dev, 1)} cents — excellent matching.",
                    "Focus on matching the reference's phrasing and dynamics next.")
            elif dev < 50:
                r_r, r_o, r_a = (
                    "acceptable",
                    f"Mean pitch deviation from reference: {round(dev, 1)} cents — within half a semitone on average.",
                    "Identify which phrase has the largest deviation and work that section in isolation.")
            else:
                r_r, r_o, r_a = (
                    "needs_work",
                    f"Mean pitch deviation from reference: {round(dev, 1)} cents — "
                    f"significant drift (max {round(max_dev, 1)} cents).",
                    "Slow-practise the passage at 60% tempo against the reference; "
                    "stop at each deviation >50 cents and re-tune before continuing.")
            _add("Reference", "mean_deviation_cents", round(dev, 1), "cents", r_r, r_o, r_a)

    # ── Score ────────────────────────────────────────────────────────────────
    WEIGHTS = {"good": 100, "acceptable": 60, "needs_work": 20}
    total_w = sum(AXIS_W.get(o["axis"], 1.0) for o in obs)
    weighted_score = sum(
        WEIGHTS[o["rating"]] * AXIS_W.get(o["axis"], 1.0) for o in obs
    ) / (total_w or 1)
    overall = round(weighted_score)

    counts = {"good": 0, "acceptable": 0, "needs_work": 0}
    for o in obs:
        counts[o["rating"]] += 1

    # Sort: needs_work first (most actionable), then acceptable, then good
    priority = {"needs_work": 0, "acceptable": 1, "good": 2}
    obs_sorted = sorted(obs, key=lambda o: priority[o["rating"]])

    # Two-sentence plain summary
    strengths = [o["observation"] for o in obs if o["rating"] == "good"]
    weaknesses = [o["observation"] for o in obs if o["rating"] == "needs_work"]
    strength_txt = strengths[0] if strengths else "Several aspects are developing well."
    weakness_txt = weaknesses[0] if weaknesses else "Continue refining consistency across all axes."
    summary = f"{strength_txt} Priority focus: {weakness_txt}"

    return {
        "overall_score": overall,
        "ratings": counts,
        "observations": obs_sorted,
        "summary": summary,
    }


def compare_to_baselines(report, baselines_path="data/popbutfy_baselines.json"):
    """Add population context to a coaching report using PopBuTFy baselines.

    For each metric in the report, emits a percentile band relative to the
    amateur and professional distributions:
        "below_amateur" | "amateur_range" | "approaching_pro" | "pro_range" | "above_pro"

    Args:
        report:          dict from build_report()
        baselines_path:  path to the JSON produced by buildPopBuTFyBaselines.py

    Returns:
        dict — keyed by metric name, each value:
            {band, amateur_median, pro_median, your_value, interpretation}
        Returns {} if baselines file not found (graceful degradation).
    """
    if not os.path.exists(baselines_path):
        return {}

    with open(baselines_path) as f:
        bl = json.load(f)

    thresh = bl.get("thresholds", {})

    # Map report fields to baseline metric keys
    def _get(path, default=None):
        node = report
        for k in path:
            if not isinstance(node, dict):
                return default
            node = node.get(k, default)
        return node

    METRIC_MAP = {
        "pitch_stability_cents": _get(["pitch", "f0_stability_std_hz"]),
        "hnr_mean_db":           _get(["voice_quality", "hnr_mean_db"]),
        "jitter_mean_pct":       _get(["voice_quality", "jitter_mean_pct"]),
        "shimmer_mean_pct":      _get(["voice_quality", "shimmer_mean_pct"]),
        "vibrato_phrase_frac":   _get(["vibrato", "phrase_fraction"]),
        "vibrato_rate_hz":       _get(["vibrato", "rate_hz_mean"]),
        "vibrato_depth_cents":   _get(["vibrato", "depth_cents_mean"]),
        "rms_arc_mean_db":       (
            float(sum(p["rms_arc_db"] for p in report.get("phrases", [])
                      if p.get("rms_arc_db") is not None) /
                  max(1, sum(1 for p in report.get("phrases", [])
                             if p.get("rms_arc_db") is not None)))
            if any(p.get("rms_arc_db") is not None for p in report.get("phrases", []))
            else None
        ),
        "technique_vibrato":     _get(["technique", "vibrato"]),
        "technique_breathy":     _get(["technique", "breathy"]),
        "technique_falsetto":    _get(["technique", "falsetto"]),
        "technique_belt":        _get(["technique", "belt"]),
        "mos":                   _get(["mos", "score"]),
    }

    # Higher-is-better metrics (for all others, lower = better)
    HIGHER_IS_BETTER = {"hnr_mean_db", "vibrato_phrase_frac", "vibrato_rate_hz",
                        "vibrato_depth_cents", "rms_arc_mean_db",
                        "technique_vibrato", "technique_belt", "mos"}

    context = {}
    for metric, value in METRIC_MAP.items():
        if value is None or value != value:  # None or nan
            continue
        t = thresh.get(metric, {})
        am_med = t.get("ama_median") or t.get("am_median") or t.get("ama_p75")
        pr_med = t.get("pro_median")
        am_p75 = t.get("ama_p75") or t.get("am_p75")
        pr_p25 = t.get("pro_p25")

        if am_med is None and pr_med is None:
            continue

        hib = metric in HIGHER_IS_BETTER

        def _band(v, am_m, am_75, pr_25, pr_m):
            if am_m is None or pr_m is None:
                return "unknown"
            if hib:
                if v >= pr_m:      return "pro_range"
                if v >= pr_25:     return "approaching_pro"
                if v >= am_m:      return "amateur_range"
                return "below_amateur"
            else:  # lower is better
                if v <= pr_m:      return "pro_range"
                if v <= pr_25:     return "approaching_pro"
                if v <= am_m:      return "amateur_range"
                return "below_amateur"

        band = _band(value, am_med, am_p75, pr_p25, pr_med)

        BAND_TEXT = {
            "pro_range":      "at professional level",
            "approaching_pro":"approaching professional standard",
            "amateur_range":  "within typical amateur range",
            "below_amateur":  "below typical amateur level",
            "unknown":        "insufficient baseline data",
        }

        context[metric] = {
            "your_value":     round(float(value), 3),
            "amateur_median": am_med,
            "pro_median":     pr_med,
            "band":           band,
            "interpretation": BAND_TEXT[band],
        }

    return context


def generate_critique(report, mode="expert", api_key=None, model="claude-opus-4-7"):
    """Call Claude API to generate a coaching critique from a structured report.

    Args:
        report:   dict from build_report()
        mode:     "expert" or "beginner"
        api_key:  Anthropic API key; if None, reads ANTHROPIC_API_KEY env var
        model:    Claude model ID (default: claude-opus-4-7)

    Returns:
        str — the critique text, or an error message string if API unavailable
    """
    if not _HAS_ANTHROPIC:
        return (
            "[anthropic package not installed — run `pip install anthropic` "
            "to enable LLM critique]"
        )

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return "[ANTHROPIC_API_KEY not set — cannot generate LLM critique]"

    system = _SYSTEM_EXPERT if mode == "expert" else _SYSTEM_BEGINNER
    mode_label = "expert technical" if mode == "expert" else "beginner-friendly"

    report_json = json.dumps(report, indent=2, default=str)
    user_msg = _USER_TEMPLATE.format(report_json=report_json, mode_label=mode_label)

    client = anthropic.Anthropic(api_key=key)
    message = client.messages.create(
        model=model,
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )

    return message.content[0].text
