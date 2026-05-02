"""
model.py — Chargement et inférence du modèle RoBERTa
Modèle utilisé : cardiffnlp/twitter-roberta-base-sentiment-latest (HuggingFace)
Ce modèle préentraîné classe les textes en positif/neutre/négatif nativement.
"""
from transformers import pipeline
from typing import Union

# ──────────────────────────────────────────────
# Chargement du pipeline (une seule fois)
# ──────────────────────────────────────────────
print("Chargement du modèle RoBERTa...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    truncation=True,
    max_length=512,
)
print(" Modèle chargé avec succès.")

# Mapping des labels bruts du modèle
LABEL_MAP = {
    "positive": "POSITIVE",
    "neutral":  "NEUTRAL",
    "negative": "NEGATIVE",
}

# ──────────────────────────────────────────────
# Fonction principale d'analyse
# ──────────────────────────────────────────────
def analyze_sentiment(text: str) -> dict:
    """
    Analyse le sentiment d'un texte.
    Retourne :
        - label : 'POSITIVE', 'NEUTRAL', ou 'NEGATIVE'
        - score : score de confiance (0 à 1)
        - raw_label : label brut du modèle
    """
    result = sentiment_pipeline(text)[0]
    raw_label = result["label"].lower()
    score = result["score"]
    label = LABEL_MAP.get(raw_label, raw_label.upper())

    return {
        "label": label,
        "score": round(score, 4),
        "raw_label": raw_label,
    }

def analyze_batch(texts: list[str]) -> list[dict]:
    """
    Analyse une liste de textes en batch (plus rapide).
    """
    results = sentiment_pipeline(texts, batch_size=16, truncation=True, max_length=512)
    output = []
    for text, result in zip(texts, results):
        raw_label = result["label"].lower()
        score = result["score"]
        label = LABEL_MAP.get(raw_label, raw_label.upper())
        output.append({
            "text": text[:120] + "..." if len(text) > 120 else text,
            "label": label,
            "score": round(score, 4),
            "raw_label": raw_label,
        })
    return output
