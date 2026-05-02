"""
model.py — Chargement et inférence du modèle DistilBERT
Modèle utilisé : distilbert-base-uncased-finetuned-sst-2-english (HuggingFace)
Ce modèle préentraîné classe les textes en positif/négatif.
On ajoute une logique de score pour détecter les avis neutres.
"""

from transformers import pipeline
from typing import Union


# ──────────────────────────────────────────────
# Chargement du pipeline (une seule fois)
# ──────────────────────────────────────────────
print("⏳ Chargement du modèle DistilBERT...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
    max_length=512,
)
print("✅ Modèle chargé avec succès.")


# ──────────────────────────────────────────────
# Fonction principale d'analyse
# ──────────────────────────────────────────────
def analyze_sentiment(text: str, neutral_threshold: float = 0.65) -> dict:
    """
    Analyse le sentiment d'un texte.

    Retourne :
        - label : 'POSITIVE', 'NEUTRAL', ou 'NEGATIVE'
        - score : score de confiance (0 à 1)
        - raw_label : label brut du modèle (POSITIVE/NEGATIVE)
    """
    result = sentiment_pipeline(text)[0]
    raw_label = result["label"]   # POSITIVE ou NEGATIVE
    score = result["score"]       # confiance

    # Si la confiance est faible → NEUTRAL
    if score < neutral_threshold:
        label = "NEUTRAL"
    else:
        label = raw_label  # POSITIVE ou NEGATIVE

    return {
        "label": label,
        "score": round(score, 4),
        "raw_label": raw_label,
    }


def analyze_batch(texts: list[str], neutral_threshold: float = 0.65) -> list[dict]:
    """
    Analyse une liste de textes en batch (plus rapide).
    """
    results = sentiment_pipeline(texts, batch_size=16, truncation=True, max_length=512)
    output = []
    for text, result in zip(texts, results):
        raw_label = result["label"]
        score = result["score"]
        label = "NEUTRAL" if score < neutral_threshold else raw_label
        output.append({
            "text": text[:120] + "..." if len(text) > 120 else text,
            "label": label,
            "score": round(score, 4),
            "raw_label": raw_label,
        })
    return output
