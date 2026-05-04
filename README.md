# 🔍 Analyse des Avis Clients — Groupe 6

> Application NLP d'analyse de sentiments basée sur **RoBERTa** (3 classes natives) avec traduction automatique FR→EN, interface Streamlit interactive et déploiement sur HuggingFace Spaces.

---

## 🌐 Application déployée

👉 **https://sentiment-analysis-cpyhvepdfafxafrywst7eg.streamlit.app/**

---

## 📌 Contexte

Les entreprises cherchent à comprendre les retours clients pour améliorer leurs produits et services. Cette application analyse automatiquement des avis clients (Amazon, Trustpilot, IMDB…) et les classe en **Positif / Neutre / Négatif** grâce à un modèle NLP de pointe, avec support du **français et de l'anglais**.

---

## 🏗️ Architecture du projet

```
sentiment-analysis/
├── app.py              # Application Streamlit complète (interface + logique NLP)
├── requirements.txt    # Dépendances Python
└── README.md
```

---

## 🤖 Modèles NLP utilisés

| Rôle | Modèle |
|------|--------|
| **Analyse de sentiments** | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| **Traduction FR→EN** | `Helsinki-NLP/opus-mt-fr-en` (MarianMT) |
| **Détection de langue** | `langdetect` + dictionnaire de fallback étendu |

### Pourquoi RoBERTa et pas DistilBERT ?

| Critère | DistilBERT SST-2 | **RoBERTa (choisi)** |
|---------|-----------------|----------------------|
| Classes | 2 (positif / négatif) | **3 (positif / neutre / négatif)** ✅ |
| Détection du neutre | Artificielle (seuil de confiance) | **Native** ✅ |
| Données d'entraînement | Films uniquement | **Twitter multidomaine** ✅ |
| Fiabilité sur argot/langage courant | Faible | **Bonne** ✅ |

---

## 📦 Dataset d'évaluation

**IMDB Movie Reviews** — via HuggingFace Datasets

```python
from datasets import load_dataset
dataset = load_dataset("imdb")  # 50 000 avis (25k train / 25k test)
```

- 🔗 https://huggingface.co/datasets/imdb
- 50 000 avis de films · Labels : Positif / Négatif
- Utilisé uniquement pour **évaluer les performances** du modèle (aucun réentraînement)

---

## 🖥️ Fonctionnalités

### 📝 Onglet — Avis Unique
- Saisie libre en **français ou anglais**
- Détection automatique de la langue (argot, jurons, expressions familières inclus)
- Traduction FR→EN automatique avant analyse + affichage du texte traduit
- **Carte résultat colorée** : vert (positif) / rouge (négatif) / gris (neutre) + barre de confiance
- 5 exemples rapides (FR et EN)
- **Historique de session** : graphique donut + barres de progression + liste des analyses
- Champ de saisie vidé automatiquement après chaque analyse

### 📦 Onglet — Analyse en Batch
- Saisie manuelle (1 avis par ligne) ou import **CSV**
- Statistiques globales + graphique donut de répartition
- Tableau coloré avec indicateur de langue par avis
- Export des résultats en **CSV**

### 📊 Onglet — Performances
- Choix libre du nombre d'exemples (n = 50 à 500) avec estimation du temps :

| n | Fiabilité | Temps estimé |
|---|-----------|--------------|
| 50 | Indicative | ~20 sec |
| 100 | Correcte | ~40 sec |
| 200 | Bonne ✅ | ~1 min 30 |
| 300+ | Très bonne | ~3 min+ |

- Résultats mis en **cache session** : un seul calcul, affichage instantané ensuite
- **Accuracy** et **F1-score** : RoBERTa vs Baseline (mots-clés)
- Rapport de classification complet
- Matrices de confusion côte à côte
- Graphique comparatif Précision / Rappel / F1

---

## 📊 Résultats de référence (n=200)

| Modèle | Accuracy | F1-score |
|--------|----------|----------|
| **RoBERTa** | ~**93%** | ~**0.93** |
| Baseline (mots-clés) | ~72% | ~0.71 |

---

## 🚀 Lancement en local

```bash
# 1. Cloner
git clone https://github.com/TON-USERNAME/sentiment-analysis.git
cd sentiment-analysis

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer
streamlit run app.py
# → http://localhost:8501
```

> ⚠️ Au premier lancement, les modèles RoBERTa (~500 MB) et Helsinki (~300 MB) se téléchargent automatiquement depuis HuggingFace.

---


---

## 👥 Mackéols FADEGNON — Projet NLP

**Stack :** Python · Streamlit · HuggingFace Transformers · scikit-learn  
**Modèles :** `cardiffnlp/twitter-roberta-base-sentiment-latest` · `Helsinki-NLP/opus-mt-fr-en`  
**Dataset :** IMDB (HuggingFace Datasets)
