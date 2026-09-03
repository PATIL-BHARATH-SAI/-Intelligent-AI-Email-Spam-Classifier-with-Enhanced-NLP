import os, re, json, pickle, nltk, numpy as np, pandas as pd, streamlit as st
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize

st.set_page_config(page_title="AI Email Spam Detector", page_icon="🛡️", layout="wide")

st.markdown("""<style>
.main-title { font-size: 2.1rem; font-weight: 700; background: linear-gradient(135deg, #4f46e5, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.spam-badge { background: #fee2e2; color: #991b1b; border: 1px solid #f87171; border-radius: 8px; padding: 10px; font-weight: 700; text-align: center; }
.ham-badge { background: #dcfce7; color: #166534; border: 1px solid #4ade80; border-radius: 8px; padding: 10px; font-weight: 700; text-align: center; }
</style>""", unsafe_allow_html=True)

# ponytail: simple NLTK setup, download if missing
for r in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    try: nltk.download(r, quiet=True)
    except: pass

stop_words, lemmatizer = set(stopwords.words('english')), WordNetLemmatizer()
tag_map = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'R': wordnet.ADV}

def clean_text(t: str) -> str:
    words = [w for w in word_tokenize(re.sub(r'[^a-z\s]', ' ', str(t).lower())) if w not in stop_words and len(w) > 2]
    return ' '.join(lemmatizer.lemmatize(w, tag_map.get(p[0], wordnet.NOUN)) for w, p in pos_tag(words))

@st.cache_resource
def load_assets():
    base = os.path.dirname(__file__) if os.path.exists(os.path.join(os.path.dirname(__file__), 'spam_classifier.pkl')) else '.'
    vec = pickle.load(open(os.path.join(base, 'tfidf_vectorizer.pkl'), 'rb'))
    clf = pickle.load(open(os.path.join(base, 'spam_classifier.pkl'), 'rb'))
    metrics = json.load(open(os.path.join(base, 'model_metrics.json'), 'r')) if os.path.exists(os.path.join(base, 'model_metrics.json')) else {}
    return vec, clf, metrics

vec, model, metrics = load_assets()

# Sidebar
st.sidebar.title("🛡️ NLP Spam Shield")
st.sidebar.markdown("**Features:** POS Lemmatization, TF-IDF Bi-Grams, LinearSVC")
if metrics and 'best_model' in metrics:
    m = metrics['models_evaluation'].get(metrics['best_model'], {})
    st.sidebar.metric("Best Model", metrics['best_model'])
    st.sidebar.metric("Accuracy", f"{m.get('accuracy', 0):.4f}")
    st.sidebar.metric("F1-Score", f"{m.get('f1_score', 0):.4f}")

st.markdown('<div class="main-title">🛡️ Intelligent Email Spam Classifier</div>', unsafe_allow_html=True)
st.caption("Enhanced NLP classification with POS-aware lemmatization & TF-IDF vectorization")

t1, t2, t3 = st.tabs(["📧 Scanner", "📊 Benchmark", "⚙️ Architecture"])

PRESETS = {
    "Custom Input": "",
    "Phishing Alert": "URGENT: Your bank account access is suspended. Click here immediately to verify your identity.",
    "Lottery Prize": "Congratulations! You won $1,000,000 in our international lottery. Reply with your details to claim.",
    "Meeting Sync": "Hi team, let's reschedule our weekly sprint sync to 3:00 PM tomorrow. Please update your agendas.",
    "Release Update": "Hello, version 2.4 has been deployed to staging. Please run integration tests."
}

with t1:
    c_in, c_pre = st.columns([2, 1])
    with c_pre:
        preset = st.radio("Quick samples:", list(PRESETS.keys()))
    with c_in:
        text = st.text_area("Email subject / body:", value=PRESETS[preset], height=140)
        scan = st.button("🔍 Scan Message", type="primary", use_container_width=True)

    if (scan or (preset != "Custom Input" and text)) and text.strip():
        cleaned = clean_text(text)
        pred = model.predict(vec.transform([cleaned]))[0]
        score = model.decision_function(vec.transform([cleaned]))[0] if hasattr(model, 'decision_function') else 0.5
        conf = 1 / (1 + np.exp(-abs(score)))

        r1, r2 = st.columns([1, 2])
        with r1:
            st.markdown('<div class="spam-badge">🚨 SPAM DETECTED</div>' if pred == 1 else '<div class="ham-badge">✅ LEGITIMATE (HAM)</div>', unsafe_allow_html=True)
            st.progress(float(min(conf, 1.0)), text=f"Confidence Margin: {conf*100:.1f}%")
        with r2:
            m1, m2, m3 = st.columns(3)
            m1.metric("Words", len(text.split()))
            m2.metric("Lemmas", len(cleaned.split()))
            m3.metric("Chars", len(text))
            with st.expander("Extracted Lemmas"): st.code(cleaned or "(None)")

with t2:
    if metrics and 'models_evaluation' in metrics:
        df_m = pd.DataFrame(metrics['models_evaluation']).T
        df_m.columns = [c.replace('_', ' ').title() for c in df_m.columns]
        max_vals = df_m.max()
        
        rows = ""
        for model_name, r in df_m.iterrows():
            cols = f"<td style='padding:12px; font-weight:700; border-bottom:1px solid #334155; color:#f1f5f9;'>{model_name}</td>"
            for c in df_m.columns:
                v = r[c]
                if v == max_vals[c]:
                    cell = f"<span style='background-color:#4ade80; color:#052e16; font-weight:900; padding:4px 10px; border-radius:6px; font-size:0.95rem;'>{v:.4f}</span>"
                else:
                    cell = f"<span style='color:#cbd5e1; font-weight:600; font-size:0.95rem;'>{v:.4f}</span>"
                cols += f"<td style='padding:12px; text-align:center; border-bottom:1px solid #334155;'>{cell}</td>"
            rows += f"<tr>{cols}</tr>"
        
        th_html = "".join([f"<th style='padding:10px; text-align:center; color:#38bdf8; font-size:0.95rem; border-bottom:2px solid #475569;'>{c}</th>" for c in df_m.columns])
        st.markdown(f"""
        <table style='width:100%; border-collapse:collapse; margin-bottom:25px; font-family:sans-serif;'>
            <thead><tr><th style='padding:10px; text-align:left; color:#38bdf8; font-size:0.95rem; border-bottom:2px solid #475569;'>Model</th>{th_html}</tr></thead>
            <tbody>{rows}</tbody>
        </table>
        """, unsafe_allow_html=True)
        
        st.markdown("##### 📈 Metric Comparison (Grouped by Metric)")
        st.bar_chart(df_m[['Accuracy', 'F1 Score', 'Roc Auc']].T)

with t3:
    st.markdown("`Raw Text` → `Deduplication` → `POS Lemmatization` → `TF-IDF (1-2 N-Grams)` → `LinearSVC` → `Prediction`")
