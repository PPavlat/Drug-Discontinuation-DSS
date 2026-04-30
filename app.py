import streamlit as st
import pandas as pd
import numpy as np
import html
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from scipy.sparse import hstack
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split


analyzer = SentimentIntensityAnalyzer()

# keywords
SIDE_EFFECTS = [
    'nausea', 'nauseous', 'vomiting', 'threw up', 'diarrhea', 'constipation',
    'stomach ache', 'upset stomach', 'bloating', 'cramps', 'heartburn', 'acid reflux',
    'headache', 'migraine', 'dizzy', 'dizziness', 'vertigo', 'lightheaded',
    'tremors', 'shaking', 'seizure', 'blurred vision', 'ringing in ears', 'tinnitus',
    'faint', 'passed out', 'numbness', 'tingling', 'pins and needles',
    'anxiety', 'panic attack', 'depression', 'depressed', 'mood swings', 'irritable',
    'agitated', 'brain fog', 'confusion', 'disorientation', 'hallucinations',
    'suicidal', 'self-harm', 'crying spells', 'emotional', 'zombie', 'numb',
    'insomnia', 'can\'t sleep', 'nightmares', 'vivid dreams', 'tired', 'fatigue',
    'lethargy', 'exhausted', 'drowsy', 'sleepy', 'restless',
    'weight gain', 'gained weight', 'weight loss', 'appetite', 'rash', 'hives',
    'itching', 'swelling', 'edema', 'palpitations', 'racing heart', 'chest pain',
    'shortness of breath', 'sweating', 'night sweats', 'dry mouth', 'hair loss',
    'acne', 'breakouts', 'libido', 'sex drive', 'joint pain', 'muscle aches'
]

QUIT_KEYWORDS = [
    'quit', 'stopped', 'stop', 'discontinued', 'ceased', 'halted', 'abandoned',
    'dropped', 'gave up', 'giving up', 'finished with', 'terminated',
    'switched', 'switching', 'changed to', 'moved to', 'replaced with', 'alternative',
    'no more', 'never again', 'last dose', 'threw it away', 'flushed', 'off of it',
    'done with', 'waste of money', 'not worth it', 'cannot take this',
    'could not tolerate', 'stopped taking', 'doctor took me off', 
    'physician told me to stop', 'was advised to quit'
]

# data prep
@st.cache_data
def load_and_merge_data():
    train_url = "https://drive.google.com/uc?export=download&id=19DwE3-BC82awdo045yGgkhlmxkwOCGwf"
    test_url = "https://drive.google.com/uc?export=download&id=1Rvp9BF_Btwxix0uEklmANJibwEOv_zNW"
    train = pd.read_parquet(train_url)
    test = pd.read_parquet(test_url)
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    return df

@st.cache_resource
def train_brain(df):
    def clean_review_text(text):
        text = html.unescape(str(text))
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        return " ".join(text.split())

    df['review_clean'] = df['review'].apply(clean_review_text)
    
    # row labels
    def derive_discontinuation(row):
        has_se = any(word in row['review_clean'] for word in SIDE_EFFECTS)
        has_quit = any(word in row['review_clean'] for word in QUIT_KEYWORDS)
        if row['rating'] <= 4 and has_se and has_quit: return 1
        if row['rating'] <= 3 and (has_se or has_quit): return 1
        if row['rating'] <= 2: return 1
        return 0

    df['pred_discontinued'] = df.apply(derive_discontinuation, axis=1)
    
    # features
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    X_text = tfidf.fit_transform(df['review_clean'])
    X_final = hstack([X_text, df[['usefulCount']].values])
    y = df['pred_discontinued']

    # split train
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    
    # metrics
    y_pred = rf.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return df, tfidf, rf, f1, cm


raw_df = load_and_merge_data()
df, tfidf_vec, rf_model, f1_val, cm_matrix = train_brain(raw_df)

# ui

st.set_page_config(page_title="Drug DSS Dashboard", layout="wide")
st.title("Medication Discontinuation Decision Support System (DSS)")

# Metric Section for Performance
st.header("Requirement 5: Performance Evaluation")
m1, m2, m3 = st.columns(3)
m1.metric("Model F1-Score", f"{f1_val:.2f}")
m2.metric("Total Records Processed", len(df))
m3.metric("Data Source", "Cloud Parquet")

with st.expander("View Confusion Matrix (Reliability Check)"):
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                xticklabels=['Continued', 'Discontinued'], 
                yticklabels=['Continued', 'Discontinued'])
    ax_cm.set_ylabel('True Label')
    ax_cm.set_xlabel('Predicted Label')
    st.pyplot(fig_cm)

st.divider()

# Requirement 6 & Bonus: Community Impact
st.header("Community Impact & Sentiment Reliability")
st.write("This metric weights sentiment by the community 'Helpfulness' score using a logarithmic scale: $Sentiment \times \ln(usefulCount + 1)$")

df['sentiment'] = df['review'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
df['community_impact'] = df['sentiment'] * np.log1p(df['usefulCount'])

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Global Predictors (Feature Importance)")
    importances = rf_model.feature_importances_
    feature_names = tfidf_vec.get_feature_names_out().tolist() + ['usefulCount']
    top_indices = np.argsort(importances)[-10:]
    fig_imp, ax_imp = plt.subplots()
    ax_imp.barh([feature_names[i] for i in top_indices], [importances[i] for i in top_indices], color='orange')
    st.pyplot(fig_imp)

with col_right:
    st.subheader("Top 'Red Flag' Reviews (High Community Agreement)")
    red_flags = df.sort_values(by='community_impact', ascending=True).head(5)
    st.dataframe(red_flags[['drugName', 'condition', 'sentiment', 'usefulCount', 'community_impact']])

st.divider()

# Drug-Specific Deep Dive
st.header("Drug-Specific Analysis")
drug_list = sorted(df['drugName'].unique())
selected_drug = st.selectbox("Select a Drug:", drug_list)

drug_data = df[df['drugName'] == selected_drug]
likelihood = drug_data['pred_discontinued'].mean()
avg_sent = drug_data['sentiment'].mean()

c1, c2, c3 = st.columns(3)
c1.metric("Discontinuation Risk", f"{likelihood:.1%}")
c2.metric("Avg Patient Rating", f"{drug_data['rating'].mean():.1f}/10")
c3.metric("Avg Sentiment", f"{avg_sent:.2f}")

if likelihood > 0.3:
    st.error(f"WARNING: {selected_drug} shows a high risk of patient discontinuation based on clinical keywords.")
else:
    st.success(f"ADHERENCE: {selected_drug} shows a low risk of discontinuation.")

# Custom Prediction
st.divider()
st.header("Bonus: Custom Patient Review Assessment")
user_text = st.text_area("Paste a patient review here:")
user_useful = st.number_input("Useful Count:", 0, 1000, 0)

if st.button("Analyze Risk"):
    if user_text:
        processed = re.sub(r'[^a-zA-Z\s]', '', user_text.lower())
        vec = tfidf_vec.transform([processed])
        inp = hstack([vec, np.array([[user_useful]])])
        prob = rf_model.predict_proba(inp)[0][1]
        
        st.write(f"**Discontinuation Probability:** {prob:.1%}")
        if prob > 0.5: st.error("High Risk of Discontinuation")
        else: st.success("Low Risk of Discontinuation")