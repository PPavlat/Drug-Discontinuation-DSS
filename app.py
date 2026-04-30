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
    df['model_pred'] = rf.predict(X_final)
    df['model_prob'] = rf.predict_proba(X_final)[:, 1]
    
    return df, tfidf, rf, f1, cm


raw_df = load_and_merge_data()
df, tfidf_vec, rf_model, f1_val, cm_matrix = train_brain(raw_df)

# ui

st.set_page_config(page_title="Drug DSS Dashboard", layout="wide")
st.title("Medication Discontinuation Decision Support System (DSS)")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    top_drugs = df['drugName'].value_counts().head(15)
    fig_counts, ax_counts = plt.subplots(figsize=(6, 3))
    top_drugs.plot(kind='bar', ax=ax_counts, color='teal')
    ax_counts.set_title("Top 15 Drugs by Number of Reviews", fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_counts, use_container_width=True)


st.header("Global Predictors of Discontinuation")
importances = rf_model.feature_importances_
feature_names = tfidf_vec.get_feature_names_out().tolist() + ['usefulCount']
top_indices = np.argsort(importances)[-10:]

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    fig_imp, ax_imp = plt.subplots(figsize=(5, 3))
    ax_imp.barh([feature_names[i] for i in top_indices], 
                [importances[i] for i in top_indices], 
                color='orange')
    ax_imp.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    st.pyplot(fig_imp, use_container_width=True)


st.divider()
st.header("Drug-Specific Discontinuation & Sentiment Analysis")
drug_list = sorted(df['drugName'].unique())
selected_drug = st.selectbox("Select a Drug:", drug_list)

drug_data = df[df['drugName'] == selected_drug].copy()


actual_risk = drug_data['pred_discontinued'].mean()  
model_risk = drug_data['model_prob'].mean()        
avg_sentiment = drug_data['review'].apply(lambda x: analyzer.polarity_scores(x)['compound']).mean()


col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reviews", len(drug_data))
col2.metric("Actual Risk (Rules)", f"{actual_risk:.1%}")
col3.metric("Predicted Risk (AI)", f"{model_risk:.1%}", 
            delta=f"{model_risk - actual_risk:+.1%}", delta_color="inverse")
col4.metric("Avg Sentiment", f"{avg_sentiment:.2f}")

st.divider()

# model vs labels
if abs(actual_risk - model_risk) < 0.1:
    st.info(f"**Model Alignment:** Model confident in the historical trend for {selected_drug}.")
else:
    st.warning(f"**Model Divergence:** Model predicts a different risk level than labels suggested.")


col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.write(f"### Sentiment Distribution for {selected_drug}")
    fig_sent, ax_sent = plt.subplots(figsize=(6, 3))
    ax_sent.hist(drug_data['sentiment'], bins=20, color='mediumpurple', edgecolor='black')
    ax_sent.set_title(f"Sentiment Range: -1 (Negative) to +1 (Positive)")
    plt.tight_layout()
    st.pyplot(fig_sent, use_container_width=True)

# result text
if model_risk > 0.5 or avg_sentiment < -0.3:
    st.error(f"HIGH RISK: {selected_drug} shows high overall discontinuation risk with a {model_risk:.1%} discontinuation rate and a {avg_sentiment:.2f} sentiment average.")
elif model_risk > 0.3 or avg_sentiment < -0.15:
    st.warning(f"MODERATE RISK: {selected_drug} shows moderate overall discontinuation risk with a {model_risk:.1%} discontinuation rate and a {avg_sentiment:.2f} sentiment average.")
elif model_risk > 0.2 and avg_sentiment < 0.1:
    st.warning(f"MODERATE RISK: {selected_drug} shows moderate overall discontinuation risk with a {model_risk:.1%} discontinuation rate and a {avg_sentiment:.2f} sentiment average.")
else:
    st.success(f"LOW RISK: {selected_drug} shows low overall discontinuation risk with a {model_risk:.1%} discontinuation rate and a {avg_sentiment:.2f} sentiment average.")

# custom review
st.divider()
st.header("Custom Review Risk Assessment")
user_text = st.text_area("Paste a patient review here to get a likelihood prediction:")
user_useful = st.number_input("How many 'Useful' votes does this review have?", 0, 1000, 0)

if st.button("Predict Discontinuation Risk"):
    if user_text:
        processed_text = re.sub(r'[^a-zA-Z\s]', '', user_text.lower())
        text_vec = tfidf_vec.transform([processed_text])
        final_input = hstack([text_vec, np.array([[user_useful]])])
        
        prob = rf_model.predict_proba(final_input)[0][1]
        user_sentiment = analyzer.polarity_scores(user_text)['compound']
        
        st.subheader(f"Results for Custom Input")
        st.write(f"**Calculated Sentiment:** {user_sentiment:.2f}")
        st.write(f"**Discontinuation Probability:** {prob:.1%}")
        
        if prob > 0.5:
            st.error("Model predicts a HIGH likelihood of medication discontinuation")
        else:
            st.success("Model predicts a LOW likelihood of medication discontinuation.")

# evaluation

st.divider()
st.header("Model Performance Evaluation")

col_eval1, col_eval2, col_eval3 = st.columns([1, 2, 1])

with col_eval2:
    st.metric("Model F1-Score", f"{f1_val:.2f}")
    
    st.write("### Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                xticklabels=['Continued', 'Discontinued'], 
                yticklabels=['Continued', 'Discontinued'])
    ax_cm.set_ylabel('Actual Status')
    ax_cm.set_xlabel('Predicted Status')
    plt.tight_layout()
    st.pyplot(fig_cm, use_container_width=True)
        