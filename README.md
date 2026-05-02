# 🔍 Analyse des Avis Clients — Groupe 6

> Application NLP d'analyse de sentiments basée sur **RoBERTa** (3 classes) avec traduction automatique FR→EN, interface Streamlit interactive et déploiement sur HuggingFace Spaces.

---

## 🌐 Application déployée

👉 **https://huggingface.co/spaces/TON-USERNAME/sentiment-analysis**

---

## 📌 Contexte

Les entreprises cherchent à comprendre les retours clients pour améliorer leurs produits et services. Cette application analyse automatiquement des avis clients (Amazon, Trustpilot, IMDB…) et les classe en **Positif / Neutre / Négatif** grâce à un modèle NLP de pointe.

---

## 🏗️ Architecture du projet

```
sentiment_project/
├── app.py              # Application Streamlit (interface complète)
├── model.py            # Fonctions d'inférence RoBERTa
├── data_loader.py      # Chargement dataset IMDB (HuggingFace)
├── evaluate.py         # Script d'évaluation standalone (CLI)
├── requirements.txt    # Dépendances Python
└── .streamlit/
    └── config.toml     # Configuration thème + port HuggingFace Spaces
```

---

## 🤖 Modèles NLP utilisés

| Rôle | Modèle | Source |
|------|--------|--------|
| **Analyse de sentiments** | `cardiffnlp/twitter-roberta-base-sentiment-latest` | HuggingFace 🤗 |
| **Traduction FR→EN** | `Helsinki-NLP/opus-mt-fr-en` | HuggingFace 🤗 |
| **Détection de langue** | `langdetect` + dictionnaire de fallback | PyPI |

### Pourquoi RoBERTa ?

| Critère | DistilBERT SST-2 | **RoBERTa (choisi)** |
|---------|-----------------|----------------------|
| Classes | 2 (pos/neg) | **3 (pos/neu/neg)** ✅ |
| Données d'entraînement | Films | **Twitter multidomaine** ✅ |
| Détection du neutre | Artificielle (seuil) | **Native** ✅ |
| Taille | ~250 MB | ~500 MB |

---

## 📦 Dataset

**IMDB Movie Reviews** — via HuggingFace Datasets

```python
from datasets import load_dataset
dataset = load_dataset("imdb")  # 50 000 avis (25k train / 25k test)
```

- 🔗 https://huggingface.co/datasets/imdb
- **50 000 avis** de films · Labels : Positif / Négatif
- Utilisé uniquement pour **évaluer les performances** du modèle (aucun réentraînement)

---

## 🖥️ Fonctionnalités de l'application

### 📝 Onglet — Avis Unique
- Saisie libre d'un avis en **français ou en anglais**
- **Détection automatique de la langue** (langdetect + dictionnaire étendu incluant argot, jurons)
- **Traduction FR→EN** automatique avant l'analyse si nécessaire
- Affichage du texte traduit pour vérification
- **Carte résultat colorée** : vert (positif) / rouge (négatif) / gris (neutre) avec barre de confiance
- **5 exemples rapides** (FR et EN) pour tester immédiatement
- **Historique de session** en colonne gauche :
  - Graphique donut mis à jour en temps réel
  - Barres de progression par sentiment
  - Liste des 8 dernières analyses
  - Bouton pour effacer l'historique
- Le champ de saisie se **vide automatiquement** après chaque analyse

### 📦 Onglet — Analyse en Batch
- Saisie manuelle (1 avis par ligne)
- Import de fichier **CSV** avec sélection de la colonne
- Statistiques globales (% positif / négatif / neutre)
- Graphique donut de répartition
- Tableau coloré détaillé avec indicateur de langue
- Export des résultats en **CSV**

### 📊 Onglet — Performances
- Choix libre du nombre d'exemples (**n = 50 à 500**)
- Estimation du temps de calcul affichée dynamiquement :
  - n=50 → ~20 sec · n=100 → ~40 sec · n=200 → ~1 min 30 · n=300+ → ~3 min+
- Résultats **mis en cache** dans la session : un seul calcul par session, affichage instantané ensuite
- Métriques affichées : **Accuracy, F1-score, Précision, Rappel**
- **Rapport de classification** complet (tableau)
- **Matrices de confusion** côte à côte (RoBERTa vs Baseline)
- **Graphique comparatif** Précision / Rappel / F1 par classe

---

## 📊 Résultats de performance (référence n=200)

| Modèle | Accuracy | F1-score pondéré |
|--------|----------|-----------------|
| **RoBERTa** | ~**93%** | ~**0.93** |
| Baseline (mots-clés) | ~72% | ~0.71 |

> Le modèle RoBERTa surpasse la baseline de +21 points d'accuracy, notamment grâce à sa compréhension contextuelle du langage.

---

## 🚀 Installation et lancement local

### 1. Cloner le dépôt
```bash
git clone https://github.com/TON-USERNAME/sentiment-analysis.git
cd sentiment-analysis
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

> ⚠️ Au premier lancement, les modèles RoBERTa (~500 MB) et Helsinki (~300 MB) se téléchargent automatiquement depuis HuggingFace. Prévoir ~5 min selon la connexion.

### 3. Lancer l'application
```bash
streamlit run app.py
```

Ouvrir dans le navigateur : **http://localhost:8501**

---

## 🧪 Dépendances principales

```
streamlit==1.35.0
transformers==4.40.0
torch==2.2.0
datasets==2.19.0
scikit-learn==1.4.2
pandas==2.2.2
matplotlib==3.8.4
seaborn==0.13.2
sentencepiece==0.2.0
sacremoses==0.1.1
langdetect==1.0.9
```

---

## 👥 Groupe 6 — Projet NLP

**Cours :** Traitement Automatique du Langage Naturel  
**Dataset :** IMDB (HuggingFace)  
**Modèles :** RoBERTa · Helsinki-NLP  
**Stack :** Python · Streamlit · HuggingFace Transformers · scikit-learn
