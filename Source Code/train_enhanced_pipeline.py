import os, re, json, pickle, nltk, pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Download required NLTK resources if not already present
REQUIRED_NLTK_PACKAGES = [
    'punkt',
    'punkt_tab',
    'stopwords',
    'wordnet',
    'omw-1.4',
    'averaged_perceptron_tagger',
    'averaged_perceptron_tagger_eng'
]

for package in REQUIRED_NLTK_PACKAGES:
    try:
        nltk.download(package, quiet=True)
    except Exception:
        pass

stop_words, lemmatizer = set(stopwords.words('english')), WordNetLemmatizer()
tag_map = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'R': wordnet.ADV}

def clean_and_lemmatize(t: str) -> str:
    words = [w for w in word_tokenize(re.sub(r'[^a-z\s]', ' ', str(t).lower())) if w not in stop_words and len(w) > 2]
    return ' '.join(lemmatizer.lemmatize(w, tag_map.get(p[0], wordnet.NOUN)) for w, p in pos_tag(words))

def main():
    path = next((p for p in ['Dataset/emails.csv', 'emails.csv'] if os.path.exists(p)), None)
    df = pd.read_csv(path).drop_duplicates(keep='last').reset_index(drop=True)
    df['text'] = df['text'].str.replace(r'^Subject:\s*', '', regex=True)
    df['clean'] = df['text'].apply(clean_and_lemmatize)

    X_tr, X_te, y_tr, y_te = train_test_split(df['clean'], df['spam'], test_size=0.2, stratify=df['spam'], random_state=42)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=6000, min_df=2, sublinear_tf=True)
    X_train, X_test = tfidf.fit_transform(X_tr), tfidf.transform(X_te)

    models = {
        'MultinomialNB': MultinomialNB(),
        'LinearSVC': LinearSVC(class_weight='balanced', random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    }
    
    results = {}
    for name, m in models.items():
        m.fit(X_train, y_tr)
        preds = m.predict(X_test)
        score = m.decision_function(X_test) if hasattr(m, 'decision_function') else getattr(m, 'predict_proba', lambda x: preds)(X_test)
        if len(getattr(score, 'shape', [])) > 1: score = score[:, 1]
        results[name] = {
            'accuracy': float(accuracy_score(y_te, preds)),
            'precision': float(precision_score(y_te, preds)),
            'recall': float(recall_score(y_te, preds)),
            'f1_score': float(f1_score(y_te, preds)),
            'roc_auc': float(roc_auc_score(y_te, score)),
            'cv_5fold_f1': float(cross_val_score(m, X_train, y_tr, cv=5, scoring='f1').mean())
        }
        print(f"[{name}] Acc: {results[name]['accuracy']:.4f} | F1: {results[name]['f1_score']:.4f}")

    best_name = max(results, key=lambda k: results[k]['f1_score'])
    os.makedirs('Source Code', exist_ok=True)
    pickle.dump(tfidf, open('Source Code/tfidf_vectorizer.pkl', 'wb'))
    pickle.dump(models[best_name], open('Source Code/spam_classifier.pkl', 'wb'))
    json.dump({'best_model': best_name, 'models_evaluation': results}, open('Source Code/model_metrics.json', 'w'), indent=2)
    print(f"Saved best model ({best_name}) to Source Code/")

if __name__ == '__main__':
    main()
