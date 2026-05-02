"""
app.py — Interface Streamlit pour l'analyse de sentiments
Groupe 6 : Analyse des Avis Clients
Modèle : RoBERTa | Dataset : IMDB
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

# ── Config page ───────────────────────────────
st.set_page_config(
    page_title="Analyse des Avis Clients",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
    .positive { color: #4ade80; font-weight: bold; font-size: 1.3rem; }
    .negative { color: #f87171; font-weight: bold; font-size: 1.3rem; }
    .neutral  { color: #a8a29e; font-weight: bold; font-size: 1.3rem; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════
# Chargement des modèles (mis en cache)
# ════════════════════════════════════════════════
@st.cache_resource(show_spinner="⏳ Chargement du modèle RoBERTa (3 classes)...")
def load_model():
    from transformers import pipeline
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True,
        max_length=512,
    )


@st.cache_resource(show_spinner="🌍 Chargement du traducteur FR→EN...")
def load_translator():
    from transformers import MarianMTModel, MarianTokenizer
    model_name = "Helsinki-NLP/opus-mt-fr-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model


# ════════════════════════════════════════════════
# Détection de langue
# ════════════════════════════════════════════════
def detect_language(text: str) -> str:
    import re as _re
    clean = _re.sub(r"[^\w\s]", " ", text.lower())
    clean = clean.replace("'", " ").replace("'", " ")
    words = set(clean.split())

    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 42
        lang = detect(text)
        if lang in ("fr", "en"):
            return lang
    except Exception:
        pass

    fr_markers = {
        "je","tu","il","elle","nous","vous","ils","elles","on","me","te","lui","y",
        "le","la","les","un","une","des","du","au","aux","mon","ma","mes",
        "ton","ta","tes","son","sa","ses","notre","votre","leur","leurs",
        "de","et","est","pas","ne","se","en","ce","qui","que","où","dont",
        "sur","avec","suis","bien","très","mais","pour","dans","par","sans",
        "plus","moins","ça","si","car","donc","puis","aussi","encore","même",
        "avoir","être","faire","aller","venir","voir","savoir","veux","peux",
        "ai","as","avons","avez","ont","es","sommes","êtes","sont",
        "perdu","trouvé","aimé","détesté","adoré","voulu",
        "content","heureux","triste","bon","mauvais","nul","mal",
        "super","génial","horrible","affreux","magnifique","excellent",
        "dommage","bizarre","incroyable","parfait","terrible",
        "putain","merde","bordel","mince","flemme","chiant","relou","naze",
        "ouf","bof","mouais","ouais","wesh","oklm","trop","grave","vachement",
        "carrément","franchement","sympa","chouette","cool","nan","zéro",
        "pourri","galère","chiante","arnaque","déçu","déçue",
        "temps","rien","tout","tous","toute","toutes","quelque","aucun",
        "jamais","toujours","souvent","parfois","aujourd","hier","demain",
        "film","quel","quelle","quels","quelles",
    }
    if len(words & fr_markers) >= 1:
        return "fr"
    return "en"


# ════════════════════════════════════════════════
# Traduction + Prédiction
# ════════════════════════════════════════════════
def translate_if_needed(text: str, translator) -> tuple[str, str]:
    lang = detect_language(text)
    if lang == "fr":
        tokenizer, model = translator
        inputs = tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        translated_tokens = model.generate(**inputs)
        translated = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return translated, "fr"
    return text, "en"


def predict(text: str, pipe, translator) -> dict:
    text_en, lang = translate_if_needed(text, translator)
    r = pipe(text_en)[0]
    label = r["label"].upper()
    return {
        "label": label,
        "score": round(r["score"], 4),
        "lang": lang,
        "translated": text_en if lang == "fr" else None,
    }


def predict_batch(texts: list, pipe, translator) -> list:
    texts_en, langs = [], []
    for t in texts:
        t_en, lang = translate_if_needed(t, translator)
        texts_en.append(t_en)
        langs.append(lang)

    results = pipe(texts_en, batch_size=16, truncation=True, max_length=512)
    out = []
    for text, r, lang in zip(texts, results, langs):
        label = r["label"].upper()
        out.append({
            "Avis": text[:120] + "..." if len(text) > 120 else text,
            "Sentiment": label,
            "Confiance": round(r["score"], 4),
            "Langue": "🇫🇷 FR" if lang == "fr" else "🇬🇧 EN",
        })
    return out


# ════════════════════════════════════════════════
# Baseline mots-clés
# ════════════════════════════════════════════════
POS_WORDS = {"great","excellent","amazing","wonderful","fantastic","good",
             "best","love","perfect","enjoyed","brilliant","superb"}
NEG_WORDS = {"bad","terrible","awful","horrible","worst","boring","poor",
             "disappointing","hate","dreadful","waste","ugly"}

def baseline(text: str) -> str:
    words = set(text.lower().split())
    p, n = len(words & POS_WORDS), len(words & NEG_WORDS)
    return "POSITIVE" if p > n else "NEGATIVE" if n > p else "NEUTRAL"


# ════════════════════════════════════════════════
# UI principale
# ════════════════════════════════════════════════
pipe = load_model()
translator = load_translator()

st.title("🔍 Analyse des Avis Clients")
st.caption("Modèle : `cardiffnlp/twitter-roberta-base-sentiment-latest` (3 classes) · Traduction FR→EN : `Helsinki-NLP/opus-mt-fr-en` · Dataset : IMDB · Groupe 6")
st.divider()

tab1, tab2, tab3 = st.tabs(["📝 Avis Unique", "📦 Analyse en Batch", "📊 Performances"])

# ── Session state ──
if "example_text" not in st.session_state:
    st.session_state.example_text = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False
if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ────────────────────────────────────────────────
# TAB 1 — Avis unique + historique
# ────────────────────────────────────────────────
with tab1:
    st.subheader("Analyser un avis client")

    EXAMPLES = [
        "This movie was absolutely fantastic! The acting was superb.",
        "Terrible film. Complete waste of time. The plot made no sense.",
        "Je suis vraiment content de ce produit, c'est excellent !",
        "Putain quel film de merde, j'ai perdu mon temps.",
        "Bof, c'est ni bien ni mal, j'ai pas trop d'avis.",
    ]

    col_hist, col_main = st.columns([1, 2], gap="large")

    with col_hist:
        st.markdown("### Historique")
        hist = st.session_state.history
        if not hist:
            st.caption("Aucune analyse effectuée pour l'instant.\nLes résultats apparaîtront ici.")
        else:
            total_h = len(hist)
            counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
            for h in hist:
                counts[h["label"]] += 1

            fig_h, ax_h = plt.subplots(figsize=(3.2, 3.2))
            fig_h.patch.set_facecolor("#0f172a")
            ax_h.set_facecolor("#0f172a")
            vals_h = [counts["POSITIVE"], counts["NEGATIVE"], counts["NEUTRAL"]]
            clrs_h = ["#4ade80", "#f87171", "#a8a29e"]
            lbls_h = [
                f"Positif ({counts['POSITIVE']})",
                f"Negatif ({counts['NEGATIVE']})",
                f"Neutre ({counts['NEUTRAL']})",
            ]
            non_zero = [(v, c, l) for v, c, l in zip(vals_h, clrs_h, lbls_h) if v > 0]
            if non_zero:
                vv, cc, ll = zip(*non_zero)
                wedges_h, texts_h, auto_h = ax_h.pie(
                    vv, labels=ll, autopct="%1.0f%%",
                    colors=cc, startangle=90,
                    wedgeprops={"width": 0.55},
                )
                for t in list(texts_h) + list(auto_h):
                    t.set_color("white")
                    t.set_fontsize(8)
            ax_h.set_title(
                f"Repartition - {total_h} analyse{'s' if total_h > 1 else ''}",
                color="white", fontsize=9,
            )
            st.pyplot(fig_h, use_container_width=True)
            plt.close()

            for lbl, clr, emo in [
                ("POSITIVE", "#4ade80", "Positif"),
                ("NEGATIVE", "#f87171", "Negatif"),
                ("NEUTRAL",  "#a8a29e", "Neutre"),
            ]:
                pct = counts[lbl] / total_h
                st.markdown(
                    f"**{emo}** "
                    f"<span style='color:{clr};'>{counts[lbl]} ({pct*100:.0f}%)</span>",
                    unsafe_allow_html=True,
                )
                st.progress(pct)

            st.divider()
            st.markdown("**Dernieres analyses :**")
            for h in reversed(hist[-8:]):
                clr_i = {"POSITIVE": "#4ade80", "NEGATIVE": "#f87171", "NEUTRAL": "#a8a29e"}[h["label"]]
                short = h["text"][:55] + ("..." if len(h["text"]) > 55 else "")
                st.markdown(
                    "<div style='background:#1e293b;border-radius:8px;padding:8px 10px;"
                    "margin-bottom:6px;font-size:0.8rem;'>"
                    f"<span style='color:{clr_i};font-weight:bold;'>{h['label']}</span> "
                    f"<span style='color:#64748b;'>({h['score']*100:.0f}%)</span><br>"
                    f"<span style='color:#94a3b8;'>{short}</span>"
                    "</div>",
                    unsafe_allow_html=True,
                )

            if st.button("Effacer l'historique", use_container_width=True):
                st.session_state.history = []
                st.rerun()

        with col_main:
        if st.session_state.clear_input:
            st.session_state.text_input = ""
            st.session_state.clear_input = False
    
        st.markdown("**Exemples rapides**")
        btn_cols = st.columns(5)
        for i, (bcol, ex) in enumerate(zip(btn_cols, EXAMPLES)):
            with bcol:
                if st.button(f"Ex.{i+1}", key=f"ex{i}", use_container_width=True):
                    st.session_state["text_input"] = ex
                    st.rerun()
    
        user_text = st.text_area(
            "Entrez un avis (francais ou anglais) :",
            height=140,
            placeholder="Ex: Ce produit est absolument incroyable !",
            key="text_input",
        )
    
        if st.button("🔍 Analyser", type="primary", use_container_width=True):
            if not user_text.strip():
                st.warning("Veuillez entrer un texte.")
            else:
                with st.spinner("Analyse en cours..."):
                    result = predict(user_text, pipe, translator)
    
                st.session_state.last_result = {
                    "text":       user_text,
                    "label":      result["label"],
                    "score":      result["score"],
                    "lang":       result["lang"],
                    "translated": result.get("translated"),
                }
                st.session_state.history.append({
                    "text":  user_text,
                    "label": result["label"],
                    "score": result["score"],
                    "lang":  result["lang"],
                })
                st.session_state.clear_input = True
                st.rerun()
    
        # ✅ FIX : récupération correcte du résultat
        if st.session_state.last_result:
            r = st.session_state.last_result
    
            CFG = {
                "POSITIVE": ("#052e16", "#16a34a", "0 0 18px rgba(74,222,128,0.35)",  "#4ade80", "😊", "POSITIF"),
                "NEGATIVE": ("#1c0707", "#dc2626", "0 0 18px rgba(248,113,113,0.35)", "#f87171", "😞", "NÉGATIF"),
                "NEUTRAL":  ("#1c1917", "#78716c", "0 0 18px rgba(168,162,158,0.25)", "#a8a29e", "😐", "NEUTRE"),
            }
    
            import html as _html
    
            bg, border, glow, color, emoji_r, lbl = CFG[r["label"]]
            pct        = int(r["score"] * 100)
            safe_text  = _html.escape(r["text"])
            safe_transl = _html.escape(r["translated"]) if r.get("translated") else None
    
            lang_badge = (
                f"<span style='background:#1e3a5f;color:#93c5fd;padding:2px 8px;"
                f"border-radius:999px;font-size:0.7rem;margin-left:8px;'>🌍 FR→EN</span>"
                if r["lang"] == "fr" else
                f"<span style='background:#1e293b;color:#94a3b8;padding:2px 8px;"
                f"border-radius:999px;font-size:0.7rem;margin-left:8px;'>🇬🇧 EN</span>"
            )
    
            transl_line = (
                f"<p style='color:#64748b;font-size:0.78rem;margin:8px 0 0;font-style:italic;'>"
                f"Traduit : &laquo; {safe_transl} &raquo;</p>"
            ) if safe_transl else ""
    
            card = (
                f"<div style='background:{bg};border:1.5px solid {border};"
                f"border-radius:14px;padding:20px 22px;margin-top:18px;box-shadow:{glow};'>"
                f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:16px;'>"
                f"<span style='font-size:2.4rem;line-height:1;'>{emoji_r}</span>"
                f"<div>"
                f"<span style='color:{color};font-size:1.6rem;font-weight:800;letter-spacing:1px;'>{lbl}</span>"
                f"{lang_badge}"
                f"</div></div>"
                f"<div style='background:rgba(255,255,255,0.04);border-left:3px solid {border};"
                f"border-radius:0 6px 6px 0;padding:10px 14px;color:#cbd5e1;"
                f"font-size:0.92rem;line-height:1.6;'>"
                f"{safe_text}"
                f"</div>"
                f"{transl_line}"
                f"<div style='margin-top:16px;'>"
                f"<div style='display:flex;justify-content:space-between;"
                f"color:#94a3b8;font-size:0.78rem;margin-bottom:6px;'>"
                f"<span>Confiance du modèle</span>"
                f"<span style='color:{color};font-weight:700;'>{pct}%</span></div>"
                f"<div style='background:#1e293b;border-radius:999px;height:8px;overflow:hidden;'>"
                f"<div style='width:{pct}%;height:100%;"
                f"background:linear-gradient(90deg,{border},{color});"
                f"border-radius:999px;'></div></div>"
                f"</div>"
                f"</div>"
            )
    
            st.markdown(card, unsafe_allow_html=True)


# ────────────────────────────────────────────────
# TAB 2 — Batch
# ────────────────────────────────────────────────
with tab2:
    st.subheader("Analyse en lot")
    st.caption("Entrez un avis par ligne, ou uploadez un fichier CSV.")

    input_mode = st.radio("Mode d'entrée", ["✏️ Saisie manuelle", "📂 Upload CSV"], horizontal=True)
    texts = []

    if input_mode == "✏️ Saisie manuelle":
        batch_default = "\n".join([
            "This movie was absolutely fantastic!",
            "Terrible film, complete waste of time.",
            "It was okay, nothing special.",
            "One of the best movies I have ever seen!",
            "Boring and predictable, I hated it.",
        ])
        raw = st.text_area("Avis (1 par ligne) :", value=batch_default, height=180)
        texts = [t.strip() for t in raw.split("\n") if t.strip()]
    else:
        uploaded = st.file_uploader("Fichier CSV avec une colonne texte", type=["csv"])
        if uploaded:
            df_up = pd.read_csv(uploaded)
            col = st.selectbox("Colonne contenant les avis :", df_up.columns)
            texts = df_up[col].dropna().astype(str).tolist()
            st.info(f"{len(texts)} avis chargés.")

    if st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True) and texts:
        with st.spinner(f"Analyse de {len(texts)} avis..."):
            rows = predict_batch(texts, pipe, translator)
            df_res = pd.DataFrame(rows)

        counts = df_res["Sentiment"].value_counts().to_dict()
        pos = counts.get("POSITIVE", 0)
        neg = counts.get("NEGATIVE", 0)
        neu = counts.get("NEUTRAL",  0)
        total = len(df_res)

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("😊 Positifs", f"{pos}", f"{pos/total*100:.1f}%")
        c2.metric("😞 Négatifs", f"{neg}", f"{neg/total*100:.1f}%")
        c3.metric("😐 Neutres",  f"{neu}", f"{neu/total*100:.1f}%")

        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")
        wedges, labels_, autopcts = ax.pie(
            [pos, neg, neu],
            labels=["Positif", "Négatif", "Neutre"],
            autopct="%1.1f%%",
            colors=["#4ade80", "#f87171", "#a8a29e"],
            startangle=90,
            wedgeprops={"width": 0.5},
        )
        for t in labels_ + autopcts:
            t.set_color("white")
        ax.set_title("Répartition des sentiments", color="white")
        col_chart, _ = st.columns([1, 2])
        with col_chart:
            st.pyplot(fig)
        plt.close()

        st.divider()
        st.subheader("Détail des avis")

        def color_sentiment(val):
            c = {"POSITIVE": "color: #4ade80", "NEGATIVE": "color: #f87171", "NEUTRAL": "color: #a8a29e"}
            return c.get(val, "")

        st.dataframe(
            df_res.style.map(color_sentiment, subset=["Sentiment"]),
            use_container_width=True, height=350
        )

        csv = df_res.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Télécharger les résultats (CSV)", csv,
                           "resultats_sentiments.csv", "text/csv")


# ────────────────────────────────────────────────
# TAB 3 — Performances
# ────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_evaluation(n_samples: int):
    from datasets import load_dataset
    import random
    dataset = load_dataset("imdb", split="test")
    random.seed(42)
    indices = random.sample(range(len(dataset)), n_samples)
    subset  = dataset.select(indices)
    texts_eval = list(subset["text"])
    y_true     = list(subset["label"])

    _pipe = load_model()
    preds_raw = _pipe(texts_eval, batch_size=16, truncation=True, max_length=512)
    label_map = {"POSITIVE": 1, "NEGATIVE": 0, "NEUTRAL": 0}
    y_roberta = [label_map[r["label"].upper()] for r in preds_raw]
    y_base    = [label_map[baseline(t)] for t in texts_eval]

    acc_r  = accuracy_score(y_true, y_roberta)
    acc_b  = accuracy_score(y_true, y_base)
    f1_r   = f1_score(y_true, y_roberta, average="weighted")
    f1_b   = f1_score(y_true, y_base,    average="weighted")
    report_r = classification_report(y_true, y_roberta,
                                     target_names=["NEGATIVE","POSITIVE"], output_dict=True)
    report_b = classification_report(y_true, y_base,
                                     target_names=["NEGATIVE","POSITIVE"], output_dict=True)
    return {
        "acc_r": acc_r, "acc_b": acc_b,
        "f1_r":  f1_r,  "f1_b":  f1_b,
        "report_r": report_r, "report_b": report_b,
        "y_true": y_true, "y_roberta": y_roberta, "y_base": y_base,
        "n": n_samples,
    }


def display_eval_results(res):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RoBERTa Accuracy", f"{res['acc_r']:.2%}",
              f"+{(res['acc_r']-res['acc_b']):.2%} vs baseline")
    c2.metric("Baseline Accuracy",   f"{res['acc_b']:.2%}")
    c3.metric("RoBERTa F1-score",    f"{res['f1_r']:.4f}")
    c4.metric("Baseline F1-score",   f"{res['f1_b']:.4f}")

    st.divider()
    st.subheader("Rapport de classification — RoBERTa")
    st.dataframe(pd.DataFrame(res["report_r"]).T.round(3), use_container_width=True)

    st.divider()
    st.subheader("Matrices de Confusion")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#1e293b")
    for ax, preds, title in zip(axes,
                                [res["y_roberta"], res["y_base"]],
                                ["RoBERTa", "Baseline (mots-cles)"]):
        cm = confusion_matrix(res["y_true"], preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["NEGATIVE","POSITIVE"],
                    yticklabels=["NEGATIVE","POSITIVE"])
        ax.set_title(title, color="white", fontsize=13)
        ax.set_xlabel("Predit", color="white")
        ax.set_ylabel("Reel", color="white")
        ax.tick_params(colors="white")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Comparaison Precision / Rappel / F1")
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(metrics))
    width = 0.35
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4))
    fig2.patch.set_facecolor("#1e293b")
    for ax, cls in zip(axes2, ["NEGATIVE", "POSITIVE"]):
        ax.set_facecolor("#0f172a")
        b1 = ax.bar(x - width/2, [res["report_r"][cls][m] for m in metrics],
                    width, label="RoBERTa",  color="#4C9BE8")
        b2 = ax.bar(x + width/2, [res["report_b"][cls][m] for m in metrics],
                    width, label="Baseline", color="#E88B4C")
        ax.set_title(f"Classe : {cls}", color="white")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, color="white")
        ax.set_ylim(0, 1.15)
        ax.tick_params(colors="white")
        ax.legend(facecolor="#1e293b", labelcolor="white")
        ax.bar_label(b1, fmt="%.2f", padding=2, color="white")
        ax.bar_label(b2, fmt="%.2f", padding=2, color="white")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()


with tab3:
    st.subheader("📊 Performances du modèle RoBERTa")
    st.markdown(
        "Évaluation sur le dataset **IMDB** · Comparaison **RoBERTa vs Baseline (mots-clés)** · "
        "Métriques : Précision · Rappel · F1-score · Accuracy"
    )
    st.divider()

    st.markdown("#### Choisir le nombre d'exemples à évaluer")
    st.markdown(
        "| n | Fiabilité | Temps estimé |\n"
        "|---|---|---|\n"
        "| **50** | Indicative | ~20 sec |\n"
        "| **100** | Correcte | ~40 sec |\n"
        "| **200** | Bonne (recommandé) | ~1 min 30 |\n"
        "| **300+** | Très bonne | ~3 min+ |"
    )

    n_eval = st.slider(
        "Nombre d'exemples IMDB",
        min_value=50, max_value=500, value=100, step=50,
    )

    if n_eval <= 50:
        est, fiab, color_est = "~20 secondes", "indicative", "🟡"
    elif n_eval <= 100:
        est, fiab, color_est = "~40 secondes", "correcte", "🟡"
    elif n_eval <= 200:
        est, fiab, color_est = "~1 min 30", "bonne ✅ recommandé", "🟢"
    elif n_eval <= 300:
        est, fiab, color_est = "~2 min 30", "très bonne", "🟢"
    else:
        est, fiab, color_est = f"~{n_eval // 100 * 1.5:.0f} min", "excellente", "🟢"

    st.info(f"{color_est} **n={n_eval}** — Durée estimée : **{est}** · Fiabilité : {fiab}")

    if st.button(f"▶️ Lancer l'évaluation sur {n_eval} exemples", type="primary", use_container_width=True):
        with st.spinner(f"Évaluation en cours sur {n_eval} exemples… ({est})"):
            res = run_evaluation(n_eval)
        st.session_state.eval_default = res
        st.rerun()

    if "eval_default" in st.session_state:
        res = st.session_state.eval_default
        st.success(f"✅ Résultats sur **{res['n']} exemples IMDB** — instantané depuis la mémoire de session.")
        display_eval_results(res)
        if st.button("🔄 Effacer et relancer une nouvelle évaluation", use_container_width=True):
            del st.session_state.eval_default
            st.rerun()
