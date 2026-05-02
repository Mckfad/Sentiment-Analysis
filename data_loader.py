"""
data_loader.py — Chargement du dataset IMDB
Le dataset IMDB est le plus simple à utiliser : 50 000 avis films (pos/neg).
Source : https://huggingface.co/datasets/imdb
"""

from datasets import load_dataset
import pandas as pd
import random


def load_imdb_sample(n_samples: int = 500, split: str = "test", seed: int = 42) -> pd.DataFrame:
    """
    Charge un échantillon du dataset IMDB.

    Args:
        n_samples : nombre d'avis à charger (défaut 500 pour aller vite)
        split     : 'train' ou 'test'
        seed      : graine pour la reproductibilité

    Returns:
        DataFrame avec colonnes ['text', 'true_label']
        true_label : 0 = négatif, 1 = positif
    """
    print(f"📥 Chargement du dataset IMDB ({split}, {n_samples} exemples)...")
    dataset = load_dataset("imdb", split=split)

    # Sélection aléatoire reproductible
    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    subset = dataset.select(indices)

    df = pd.DataFrame({
        "text": subset["text"],
        "true_label": subset["label"],  # 0=neg, 1=pos
    })

    # Ajout d'une colonne lisible
    df["true_sentiment"] = df["true_label"].map({0: "NEGATIVE", 1: "POSITIVE"})

    print(f"✅ Dataset chargé : {len(df)} avis | "
          f"Positifs: {df['true_label'].sum()} | "
          f"Négatifs: {(df['true_label']==0).sum()}")
    return df


def get_sample_reviews(n: int = 5) -> list[str]:
    """
    Retourne quelques avis IMDB pour les démos rapides.
    """
    dataset = load_dataset("imdb", split="test")
    return [dataset[i]["text"][:300] for i in range(n)]
