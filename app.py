import streamlit as st
import pandas as pd
import numpy as np
import html
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


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
@st.cache_resource
def process_full_pipeline():

    train_url = "https://drive.google.com/uc?export=download&id=19DwE3-BC82awdo045yGgkhlmxkwOCGwf"
    test_url = "https://drive.google.com/uc?export=download&id=1Rvp9BF_Btwxix0uEklmANJibwEOv_zNW"
    
    train = pd.read_parquet(train_url)
    test = pd.read_parquet(test_url)
    
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
   
    def clean_review_text(text):
        text = html.unescape(str(text))
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        return " ".join(text.split())

    df['review_clean'] = df['review'].apply(clean_review_text)
    
    # labeling
    def derive_discontinuation(row):
        has_se = any(word in row['review_clean'] for word in SIDE_EFFECTS)
        has_quit = any(word in row['review_clean'] for word in QUIT_KEYWORDS)
        if row['rating'] <= 4 and has_se and has_quit: return 1
        if row['rating'] <= 3 and has_se: return 1
        if row['rating'] <= 2: return 1
        return 0

    df['pred_discontinued'] = df.apply(derive_discontinuation, axis=1)
    
    # train
    tfidf = TfidfVectorizer(max_features=2000, stop_words='english')
    X_text = tfidf.fit_transform(df['review_clean'])
    X_final = hstack([X_text, df[['usefulCount']].values])
    
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_final, df['pred_discontinued'])
    
    return df, tfidf, rf

df, tfidf_vec, rf_model = process_full_pipeline()

# ui
st.set_page_config(page_title="Drug DSS Dashboard", layout="wide")
st.title("Medication Discontinuation Decision Support")


st.header("1. Dataset Overview: Review Popularity")
top_drugs = df['drugName'].value_counts().head(15)
fig_counts, ax_counts = plt.subplots(figsize=(10, 5))
top_drugs.plot(kind='bar', ax=ax_counts, color='teal')
ax_counts.set_title("Top 15 Drugs by Number of Reviews")
st.pyplot(fig_counts)


st.header("2. Global Predictors of Discontinuation")
importances = rf_model.feature_importances_
feature_names = tfidf_vec.get_feature_names_out().tolist() + ['usefulCount']
top_indices = np.argsort(importances)[-10:]

fig_imp, ax_imp = plt.subplots()
ax_imp.barh([feature_names[i] for i in top_indices], [importances[i] for i in top_indices], color='orange')
st.pyplot(fig_imp)


st.divider()
st.header("3. Drug-Specific Discontinuation & Sentiment Analysis")
drug_list = sorted(df['drugName'].unique())
selected_drug = st.selectbox("Select a Drug to see Discontinuation Likelihood & Sentiment Results:", drug_list)


drug_data = df[df['drugName'] == selected_drug].copy()
drug_data['discontinued'] = drug_data['pred_discontinued']


drug_data['sentiment'] = drug_data['review'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
avg_sentiment = drug_data['sentiment'].mean()
likelihood = drug_data['discontinued'].mean()


col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reviews", len(drug_data))
col2.metric("Avg Patient Rating", f"{drug_data['rating'].mean():.1f}/10")
col3.metric("Discontinuation Risk", f"{likelihood:.1%}")
col4.metric("Avg Sentiment Score", f"{avg_sentiment:.2f}")


st.write(f"### Sentiment Distribution for {selected_drug}")
fig_sent, ax_sent = plt.subplots(figsize=(8, 3))
ax_sent.hist(drug_data['sentiment'], bins=20, color='mediumpurple', edgecolor='black')
ax_sent.set_title(f"Sentiment Range: -1 (Negative) to +1 (Positive)")
st.pyplot(fig_sent)

# resultt text
if likelihood > 0.3 or avg_sentiment < -0.1:
    st.error(f"HIGH RISK: {selected_drug} shows a {likelihood:.1%} discontinuation rate and a negative sentiment trend.")
else:
    st.success(f"LOW RISK: {selected_drug} shows high treatment adherence and positive patient sentiment.")

# custom review
st.divider()
st.header("4. Bonus: Custom Review Risk Assessment")
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
            st.error("Model predicts a HIGH likelihood of the patient stopping this medication.")
        else:
            st.success("Model predicts a LOW likelihood of discontinuation.")