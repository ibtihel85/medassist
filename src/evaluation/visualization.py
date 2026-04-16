from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass


def compute_calibration_data(
    confidences: list[float],
    quality_scores: list[float],
    n_bins: int = 5,
) -> dict:
    """
    Calibration analysis: correlates confidence scores with actual quality.

    For each confidence bin, computes mean quality score.
    Expected Calibration Error (ECE): weighted mean |confidence - quality| per bin.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_data = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = [(lo <= c < hi) for c in confidences]
        if not any(mask):
            continue
        bin_confs = [c for c, m in zip(confidences, mask) if m]
        bin_quality = [q for q, m in zip(quality_scores, mask) if m]
        bin_data.append(
            {
                "bin_center": (lo + hi) / 2,
                "mean_conf": float(np.mean(bin_confs)),
                "mean_quality": float(np.mean(bin_quality)),
                "count": len(bin_confs),
            }
        )

    if not bin_data:
        return {"ece": None, "bins": []}

    total = sum(b["count"] for b in bin_data)
    ece = sum(
        b["count"] / total * abs(b["mean_conf"] - b["mean_quality"]) for b in bin_data
    )

    return {"ece": float(ece), "bins": bin_data}


def plot_calibration_curve(
    calib_data: dict,
    results_df: pd.DataFrame,
    confidence_threshold: float,
    confidence_refuse: float,
    output_path: str,
) -> None:
    import matplotlib.pyplot as plt

    bins = calib_data["bins"]
    if not bins:
        return

    confs = [b["mean_conf"] for b in bins]
    quality = [b["mean_quality"] for b in bins]
    counts = [b["count"] for b in bins]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1.5)
    ax.scatter(
        confs,
        quality,
        s=[c * 200 for c in counts],
        color="royalblue",
        alpha=0.7,
        zorder=3,
        label="System bins",
    )
    ax.plot(confs, quality, "b-", alpha=0.5)
    ax.fill_between(confs, confs, quality, alpha=0.1, color="red", label="Calibration gap")
    ax.set_xlabel("Mean Confidence Score")
    ax.set_ylabel("Mean Answer Quality (BERTScore/Faithfulness)")
    ax.set_title(f"MedAssist Confidence Calibration\nECE = {calib_data['ece']:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax2 = axes[1]
    all_confs = results_df["confidence"].dropna().tolist()
    ax2.hist(all_confs, bins=20, color="steelblue", alpha=0.7, edgecolor="black")
    ax2.axvline(
        confidence_threshold,
        color="orange",
        linestyle="--",
        label=f"Warn threshold ({confidence_threshold})",
    )
    ax2.axvline(
        confidence_refuse,
        color="red",
        linestyle="--",
        label=f"Refuse threshold ({confidence_refuse})",
    )
    ax2.set_xlabel("Confidence Score")
    ax2.set_ylabel("Count")
    ax2.set_title("Confidence Score Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_evaluation_dashboard(
    results_df: pd.DataFrame,
    ret_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    hall_df: pd.DataFrame,
    judge_df: pd.DataFrame,
    confidence_threshold: float,
    output_path: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("MedAssist AI — Evaluation Dashboard", fontsize=16, fontweight="bold")

    # 1. Error distribution pie
    ax = axes[0, 0]
    error_counts = results_df["error_type"].value_counts()
    colors = [
        "#2ecc71"
        if e == "OK"
        else "#e74c3c"
        if e in ["HALLUCINATION", "SAFETY_GAP", "GENERATION_FAIL"]
        else "#f39c12"
        for e in error_counts.index
    ]
    ax.pie(
        error_counts.values,
        labels=error_counts.index,
        colors=colors,
        autopct="%1.0f%%",
        startangle=90,
    )
    ax.set_title("Error Type Distribution")

    # 2. Confidence by category
    ax = axes[0, 1]
    cat_conf = results_df.groupby("category")["confidence"].mean().sort_values()
    cat_conf.plot(kind="barh", ax=ax, color="steelblue")
    ax.axvline(confidence_threshold, color="orange", linestyle="--", label="Warn threshold")
    ax.set_title("Mean Confidence by Query Category")
    ax.legend(fontsize=8)
    ax.set_xlabel("Confidence Score")

    # 3. Latency by routing path
    ax = axes[0, 2]
    lat_by_tool = (
        results_df[results_df["latency_s"] > 0].groupby("tool_used")["latency_s"].mean()
    )
    lat_by_tool.plot(kind="bar", ax=ax, color="coral")
    ax.set_title("Mean Latency by Routing Path")
    ax.set_ylabel("Latency (seconds)")
    ax.tick_params(axis="x", rotation=30)

    # 4. Hallucination rates by category
    ax = axes[1, 0]
    if len(hall_df) > 0:
        hall_by_cat = (
            hall_df.groupby("category")["hallucination_rate"].mean().sort_values()
        )
        colors_hall = [
            "#e74c3c"
            if r > 0.30
            else "#f39c12"
            if r > 0.15
            else "#2ecc71"
            for r in hall_by_cat
        ]
        hall_by_cat.plot(kind="barh", ax=ax, color=colors_hall)
        ax.axvline(0.15, color="orange", linestyle="--", label="Warn (15%)")
        ax.axvline(0.30, color="red", linestyle="--", label="Risk (30%)")
        ax.set_title("Hallucination Rate by Category")
        ax.legend(fontsize=8)

    # 5. Retrieval metrics grouped bar
    ax = axes[1, 1]
    if len(ret_df) > 0:
        ret_means = ret_df[["precision", "recall", "relevance_mean", "mrr"]].mean().dropna()
        thresholds_ret = {
            "precision": 0.70,
            "recall": 0.60,
            "relevance_mean": 0.55,
            "mrr": 0.65,
        }
        x = np.arange(len(ret_means))
        ax.bar(x, ret_means.values, color="teal", alpha=0.7, label="System")
        ax.bar(
            x,
            [thresholds_ret.get(k, 0.6) for k in ret_means.index],
            color="gray",
            alpha=0.3,
            label="Threshold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(ret_means.index, rotation=20)
        ax.set_ylim(0, 1)
        ax.set_title("Retrieval Quality vs Thresholds")
        ax.legend(fontsize=8)

    # 6. LLM Judge scores
    ax = axes[1, 2]
    if len(judge_df) > 0:
        judge_means = judge_df[
            ["clinical_accuracy", "completeness", "safety", "groundedness", "clarity"]
        ].mean()
        colors_judge = [
            "#e74c3c" if v < 3 else "#f39c12" if v < 4 else "#2ecc71"
            for v in judge_means
        ]
        judge_means.plot(kind="bar", ax=ax, color=colors_judge)
        ax.axhline(3, color="orange", linestyle="--", label="Min acceptable (3/5)")
        ax.axhline(4, color="green", linestyle="--", label="Target (4/5)")
        ax.set_ylim(0, 5)
        ax.set_title("LLM Judge Scores (1-5)")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
