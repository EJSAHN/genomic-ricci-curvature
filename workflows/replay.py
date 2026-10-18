"""Numerical replay from distributed derived inputs; no raw-read alignment."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, SOURCES, POOLS, close, execute, load_module, make_workspace, metrics, read_rows, require, verify_metadata, write_json


def validate_reporting() -> dict:
    endpoints = read_rows(ROOT / "reporting/Endpoints.tsv")
    reps = read_rows(ROOT / "reporting/Replicates.tsv")
    for row in endpoints:
        sub = [r for r in reps if r["endpoint_id"] == row["endpoint_id"]]
        require(len(sub) == int(row["n_replicates"]), "Replicate count mismatch: " + row["endpoint_id"])
        for name, field in [("auc", "auc_mean"), ("ap", "ap_mean")]:
            values = [float(r[name]) for r in sub]
            require(close(statistics.mean(values), row[field]), "Reporting mean mismatch: " + row["endpoint_id"])
            sd = row[name + "_sd_sample"]
            if len(values) > 1:
                require(close(statistics.stdev(values), sd), "Sample SD mismatch: " + row["endpoint_id"])
            else:
                require(sd == "", "Single calibration must not acquire a replicate SD")
        prevalence = int(row["n_mixtures"]) / (int(row["n_mixtures"]) + int(row["n_controls"]))
        require(close(prevalence, row["prevalence"]), "Denominator mismatch")
        require(close(float(row["ap_mean"]) / prevalence, row["ap_lift"]), "AP-lift mismatch")
    return {"endpoints": len(endpoints), "SD_denominator": "n-1", "score_scaling_changed": False}


def check_sensitivity() -> dict:
    support = load_module(ROOT / "pipelines/coffee/calibration/audit_support.py", "calibration_audit_support")
    records = read_rows(ROOT / "reporting/Sweep_Runs.tsv")
    require(len(records) == 210, "Incomplete sensitivity table")
    maximum = 0.0
    for row in records:
        folder = ROOT / "examples/coffee/sensitivity" / row["run_dir"]
        nodes = read_rows(folder / "node_scores.csv", ",")
        truth = read_rows(folder / "synthetic_truth.csv", ",")
        require(len(nodes) == 52, "Sensitivity node count mismatch")
        require({r["sample"] for r in nodes if r["label"] == "individual"} == set(SOURCES), "Sensitivity source mismatch")
        expected = support.expected_truth(SOURCES, {"seed": int(row["seed"]), "n_synth": 40, "max_parents": 3, "min_minor": 0.10})
        require(len(truth) == len(expected), "Incomplete synthetic design")
        for a, b in zip(truth, expected):
            for k in ["sample", "parents", "weights", "n_parents"]:
                require(str(a[k]) == str(b[k]), f"Parent/seed mismatch in {row['run_dir']}")
        mm = metrics([int(r["label"] == "synthetic_pool") for r in nodes], [r["mixture_score_ot"] for r in nodes])
        for key, col in [("auc", "roc_auc"), ("ap", "avg_precision")]:
            difference = abs(mm[key] - float(row[col]))
            maximum = max(maximum, difference)
            require(difference < 1e-10, "Sensitivity metric mismatch")
    return {"runs": 210, "metrics_recomputed_from_scores": True, "parents_and_seeds_checked": True, "max_difference": maximum}



def check_ablation() -> dict:
    """Use the original ablation functions and compare every stored component row."""
    core = load_module(ROOT / "pipelines/coffee/core/10_summarize_score_components.py", "ablation_kernel")
    table = pd.read_csv(ROOT / "reporting/Ablation.tsv", sep="\t")
    checked = 0
    maximum = 0.0
    for dirname, label in [("calibration_r1", "historical_13_sources"), ("calibration", "corrected_12_sources")]:
        scores = pd.read_csv(ROOT / "examples/coffee" / dirname / "node_scores.csv")
        truth = pd.read_csv(ROOT / "examples/coffee" / dirname / "synthetic_truth.csv")
        scores = scores.merge(truth[["sample", "entropy_norm"]], on="sample", how="left", validate="one_to_one")
        y = (scores.label == "synthetic_pool").astype(int).to_numpy()
        bet, neg = scores.betweenness, scores.neg_orc_incidence
        components = {
            "Full synthetic score (mixture_score_ot)": scores.mixture_score_ot,
            "Betweenness only (Z)": core.zscore(bet),
            "NegORC only (Z)": core.zscore(neg),
            "Z(Betweenness) + Z(NegORC)": core.zscore(bet) + core.zscore(neg),
            "Betweenness + NegORC (raw sum)": bet + neg,
        }
        for name, values in components.items():
            m = core.compute_metrics(y, values.to_numpy(dtype=float), scores.entropy_norm.to_numpy())
            recorded = table[(table.source_set == label) & (table.component == name)]
            require(len(recorded) == 1, "Ablation row absent/duplicated")
            r = recorded.iloc[0]
            for field in ["auc", "ap", "best_f1", "best_thr"]:
                diff = abs(float(m[field]) - float(r[field]))
                require(diff < 1e-10, "Ablation metric differs: " + field)
                maximum = max(maximum, diff)
            require(m["n"] == int(r["n"]), "Ablation denominator differs")
            checked += 1
    return {"rows_checked": checked, "official_ablation_functions_used": True, "maximum_metric_difference": maximum}

def calibration_from_matrix(work: Path) -> dict:
    core = load_module(ROOT / "pipelines/coffee/calibration/calibration_core.py", "calibration_kernel")
    support = load_module(ROOT / "pipelines/coffee/calibration/run_calibration.py", "guarded_calibration")
    core.linprog = support.checked_transport(core.linprog)
    results = {}
    for dirname, n in [("calibration_r1", 13), ("calibration", 12)]:
        folder = ROOT / "examples/coffee" / dirname
        distance = pd.read_csv(folder / "js_distance.csv", index_col=0)
        expected = pd.read_csv(folder / "node_scores.csv")
        graph = core.build_symmetric_knn_graph(distance, k=4)
        curvature = core.ollivier_ricci_curvature(graph, distance, alpha=0.5)
        bet = core.nx.betweenness_centrality(graph, weight="d", normalized=True)
        negative = {v: 0.0 for v in graph.nodes()}
        for u, v in graph.edges():
            kappa = float(curvature[tuple(sorted((u, v)))])
            if kappa < 0:
                negative[u] += -kappa
                negative[v] += -kappa
        order = list(graph.nodes())
        values = core.zscore(np.asarray([bet[v] for v in order])) + core.zscore(np.asarray([negative[v] for v in order]))
        scored = dict(zip(order, values))
        maximum = max(abs(scored[r.sample] - r.mixture_score_ot) for r in expected.itertuples())
        require(maximum < 1e-10, "Calibration graph-score replay mismatch")
        mm = metrics((expected.label == "synthetic_pool").astype(int), [scored[r] for r in expected["sample"]])
        results[dirname] = {**mm, "sources": n, "max_score_difference": maximum}
    write_json(work / "CALIBRATION_MATRIX_REPLAY.json", results)
    return results


def rare_event(work: Path, source_analysis: Path | None = None, truth: Path | None = None, scenarios=("primary", "conservative")) -> dict:
    pipeline = ROOT / "pipelines/coffee/rare_event"
    source = ROOT / "examples/coffee/rare_input"
    truth = truth or source / "truth_manifest.tsv"
    source_analysis = source_analysis or source / "analysis"
    summary = load_module(pipeline / "summary_core.py", "rare_summary")
    results = {}
    for scenario in scenarios:
        require(scenario in {"primary", "conservative"}, "Unknown scenario")
        out = work / ("rare_" + scenario)
        out.mkdir(parents=True, exist_ok=True)
        schedule = out / "schedule.tsv"
        schedule.write_text("scenario\tanalysis_mode\tcontext\tinjection_counts\n" + scenario + "\tpaired\tall_controls\t1,2,4\n")
        execute(pipeline / "01_build_rare_event_design.py", ["--truth_manifest", truth, "--schedule", schedule, "--outdir", out / "design", "--seed", "43000"], work, scenario + "_rare_design")
        execute(pipeline / "02_analyze_rare_event_injection.py", ["--design", out / "design/rare_event_design.tsv", "--truth_manifest", truth, "--source_analysis_dir", source_analysis, "--outdir", out / "analysis", "--knn", "4", "--alpha", "0.5"], work, scenario + "_rare_analysis")
        observed = pd.read_csv(out / "analysis/graph_metrics.tsv", sep="\t")
        expected = pd.read_csv(ROOT / "examples/coffee/expected" / ("rare_event_R1.tsv" if scenario == "primary" else "rare_event_corrected.tsv"), sep="\t")
        keys = ["scenario", "analysis_mode", "context", "injection_count"]
        columns = [c[:-5] for c in expected.columns if c.endswith("_mean")]
        require(len(observed) == 210 and set(observed.scenario) == {scenario}, "Rare graph count/scenario mismatch")
        n_controls = 13 if scenario == "primary" else 12
        require(observed.n_controls.eq(n_controls).all(), "Rare denominator mismatch")
        actual = summary.describe_by_replicate(observed, columns)
        a = actual.sort_values(keys).reset_index(drop=True)
        b = expected.sort_values(keys).reset_index(drop=True)
        require(a[keys].equals(b[keys]), "Rare group identity mismatch")
        numeric = [c for c in b if c not in keys]
        require(np.allclose(a[numeric], b[numeric], rtol=0, atol=1e-10, equal_nan=True), "Rare-event replay differs from recorded result")
        actual.to_csv(out / "summary.tsv", sep="\t", index=False)
        r = actual[actual.injection_count == 1].iloc[0]
        results[scenario] = {"auc": float(r.roc_auc_mean), "sample_SD": float(r.roc_auc_sd), "AP": float(r.average_precision_mean), "graphs": 210, "controls": n_controls}
    write_json(work / "RARE_REPLAY.json", results)
    return results


def reference(work: Path, features: Path | None = None) -> dict:
    base = ROOT / "pipelines/coffee/reference_based"
    support = load_module(base / "reference_support.py", "reference_support")
    features = features or ROOT / "examples/coffee/reference_features"
    refs = ROOT / "config/coffee/reference_based/reference_assemblies.tsv"
    scores = work / "reference_scores"
    support.validate_feature_root(features, "conservative", 12)
    execute(base / "crossfit/01_score_reference_qc_crossfit.py", ["--reference_manifest", refs, "--features_root", features, "--outdir", scores, "--min_control_call_rate", ".30", "--max_features", "5000", "--max_pca_components", "5", "--min_overlap", "10"], work, "reference_crossfit")
    audit = support.validate_crossfit(scores, "conservative", 12)
    comparisons = {}
    for name, keys in [("crossfit_generated_reference_scores.tsv", ["reference_id", "sample_id"]), ("crossfit_real_reference_scores.tsv", ["reference_id", "sample_id"])]:
        comparisons[name] = support.compare_frames(pd.read_csv(scores / name, sep="\t"), pd.read_csv(ROOT / "examples/coffee/expected/reference" / name, sep="\t"), keys, atol=1e-8)
    provenance = pd.read_csv(scores / "crossfit_generated_fold_provenance.tsv", sep="\t")
    recorded = pd.read_csv(ROOT / "examples/coffee/expected/reference/crossfit_generated_fold_provenance.tsv", sep="\t")
    for name in [c for c in recorded if "sha256" in c or "hash" in c]:
        require(list(provenance[name]) == list(recorded[name]), "Marker/split hash changed: " + name)
    raw = support.make_consensus(scores, scores / "consensus", "conservative", ROOT / "reporting/C_Batch_Scores.tsv")
    reps = pd.read_csv(scores / "consensus/crossfit_generated_consensus_metrics.tsv", sep="\t")
    reps = reps[reps.score_variant == "reference_qc_crossfit"]
    published = {"auc_mean": float(reps.roc_auc.mean()), "auc_sd_sample": float(reps.roc_auc.std(ddof=1)), "ap_mean": float(reps.average_precision.mean()), "ap_sd_sample": float(reps.average_precision.std(ddof=1)), "reporting_ddof": 1}
    expected = next(r for r in read_rows(ROOT / "reporting/Endpoints.tsv") if r["endpoint_id"] == "coffee_corrected_12_reference_batch")
    for field in ["auc_mean", "auc_sd_sample", "ap_mean", "ap_sd_sample"]:
        require(close(published[field], expected[field]), "Reference reporting mismatch: " + field)
    result = {"crossfit": audit, "score_replay": comparisons, "reporting": published, "historical_summary_ddof": raw["sd_ddof"], "raw_alignment_rerun": False}
    write_json(work / "REFERENCE_REPLAY.json", result)
    return result


def main():
    from common import verify_design_contract
    verify_design_contract()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true", help="Check summaries and 210 score checkpoints without refitting graphs/models")
    args = parser.parse_args()
    work = make_workspace(args.output)
    report = {"scope": "Derived-input replay, not raw-read alignment", "metadata": verify_metadata()}
    try:
        report["reporting"] = validate_reporting()
        report["sensitivity"] = check_sensitivity()
        report["ablation"] = check_ablation()
        if not args.quick:
            report["calibration"] = calibration_from_matrix(work)
            report["rare_event"] = rare_event(work)
            report["reference"] = reference(work)
        report["result"] = "PASS"
    except BaseException as exc:
        report["result"] = "FAIL"
        report["error"] = repr(exc)
        write_json(work / "REPLAY_VERIFICATION.json", report)
        raise
    write_json(work / "REPLAY_VERIFICATION.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
