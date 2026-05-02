"""
evaluate.py — Évaluation des performances du modèle
Calcul : Précision, Rappel, F1-score, Matrice de confusion
Comparaison avec un modèle baseline (règles simples)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sans GUI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from model import analyze_batch
from data_loader import load_imdb_sample


# ─────────────────────────────────────────────
# Baseline naïf : mots-clés positifs/négatifs
# ─────────────────────────────────────────────
POSITIVE_WORDS = {"great","excellent","amazing","wonderful","fantastic","good",
                  "best","love","perfect","enjoyed","brilliant","superb"}
NEGATIVE_WORDS = {"bad","terrible","awful","horrible","worst","boring","poor",
                  "disappointing","hate","dreadful","waste","ugly"}

def baseline_predict(text: str) -> str:
    words = set(text.lower().split())
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    if pos > neg:
        return "POSITIVE"
    elif neg > pos:
        return "NEGATIVE"
    return "NEUTRAL"


# ─────────────────────────────────────────────
# Évaluation complète
# ─────────────────────────────────────────────
def evaluate_model(n_samples: int = 200, output_dir: str = "."):
    """
    Évalue RoBERTa vs Baseline sur le dataset IMDB.
    Génère les graphiques de performance.
    """
    df = load_imdb_sample(n_samples=n_samples)

    # Prédictions RoBERTa
    print(" Prédictions RoBERTa en cours...")
    preds = analyze_batch(df["text"].tolist())
    df["roberta_label"] = [p["label"] for p in preds]
    df["roberta_score"] = [p["score"] for p in preds]

    # Prédictions Baseline
    print("📏 Prédictions Baseline en cours...")
    df["baseline_label"] = df["text"].apply(baseline_predict)

    # ── Mapping pour sklearn (NEUTRAL → NEGATIVE car IMDB n'a pas de label neutre)
    label_map = {"POSITIVE": 1, "NEGATIVE": 0, "NEUTRAL": 0}
    y_true     = df["true_label"].tolist()
    y_roberta  = df["roberta_label"].map(label_map).tolist()
    y_baseline = df["baseline_label"].map(label_map).tolist()

    # ── Rapport classification
    print("\n" + "="*55)
    print(" RAPPORT — RoBERTa")
    print("="*55)
    report_bert = classification_report(
        y_true, y_roberta,
        target_names=["NEGATIVE", "POSITIVE"],
        output_dict=True
    )
    print(classification_report(y_true, y_roberta, target_names=["NEGATIVE", "POSITIVE"]))

    print("\n" + "="*55)
    print(" RAPPORT — Baseline (mots-clés)")
    print("="*55)
    report_base = classification_report(
        y_true, y_baseline,
        target_names=["NEGATIVE", "POSITIVE"],
        output_dict=True
    )
    print(classification_report(y_true, y_baseline, target_names=["NEGATIVE", "POSITIVE"]))

    # ── Graphiques
    _plot_confusion_matrices(y_true, y_roberta, y_baseline, output_dir)
    _plot_f1_comparison(report_bert, report_base, output_dir)

    return {
        "roberta":           report_bert,
        "baseline":          report_base,
        "accuracy_roberta":  accuracy_score(y_true, y_roberta),
        "accuracy_baseline": accuracy_score(y_true, y_baseline),
        "f1_roberta":        f1_score(y_true, y_roberta,  average="weighted"),
        "f1_baseline":       f1_score(y_true, y_baseline, average="weighted"),
    }


def _plot_confusion_matrices(y_true, y_bert, y_base, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Matrices de Confusion", fontsize=14, fontweight="bold")

    for ax, preds, title in zip(
        axes,
        [y_bert, y_base],
        ["RoBERTa", "Baseline (mots-clés)"]
    ):
        cm = confusion_matrix(y_true, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["NEGATIVE", "POSITIVE"],
                    yticklabels=["NEGATIVE", "POSITIVE"])
        ax.set_title(title)
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")

    plt.tight_layout()
    path = f"{output_dir}/confusion_matrices.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Graphique sauvegardé : {path}")


def _plot_f1_comparison(report_bert, report_base, output_dir):
    metrics = ["precision", "recall", "f1-score"]
    classes = ["NEGATIVE", "POSITIVE"]

    bert_vals = [[report_bert[c][m] for m in metrics] for c in classes]
    base_vals = [[report_base[c][m] for m in metrics] for c in classes]

    x = np.arange(len(metrics))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Comparaison RoBERTa vs Baseline", fontsize=14, fontweight="bold")

    for ax, bert_v, base_v, cls in zip(axes, bert_vals, base_vals, classes):
        b1 = ax.bar(x - width/2, bert_v, width, label="RoBERTa",  color="#4C9BE8")
        b2 = ax.bar(x + width/2, base_v, width, label="Baseline", color="#E88B4C")
        ax.set_title(f"Classe : {cls}")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.bar_label(b1, fmt="%.2f", padding=2)
        ax.bar_label(b2, fmt="%.2f", padding=2)

    plt.tight_layout()
    path = f"{output_dir}/f1_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Graphique sauvegardé : {path}")


if __name__ == "__main__":
    results = evaluate_model(n_samples=200, output_dir="static")
    print(f"\n Accuracy RoBERTa  : {results['accuracy_roberta']:.2%}")
    print(f" Accuracy Baseline : {results['accuracy_baseline']:.2%}")
    print(f" F1 RoBERTa        : {results['f1_roberta']:.4f}")
    print(f" F1 Baseline       : {results['f1_baseline']:.4f}")
