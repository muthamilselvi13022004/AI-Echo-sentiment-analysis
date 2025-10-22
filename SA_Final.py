import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import joblib


# Title of the Streamlit app
st.title("ChatGPT Style Reviews Dataset")

try:
    # Load CSV into DataFrame
    df = pd.read_csv("C:/Users/manju/Downloads/chatgpt_style_reviews_dataset.xlsx - Sheet1.csv")

    # Display the DataFrame
    st.write("#### Dataset")
    st.dataframe(df)

except Exception as e:
    st.error(f"Error loading file: {e}")

# Convert ratings into sentiment labels
if "rating" in df.columns:
    def rating_to_sent(r):
        if pd.isna(r): 
            return None
        if r >= 4: 
            return "positive"
        if r == 3: 
            return "neutral"
        return "negative"

    df["sentiment"] = df["rating"].apply(rating_to_sent)
else:
    st.error("The dataset does not have a 'rating' column to derive sentiment.")

model_path = os.path.join(os.path.dirname(__file__), "NLP_model.pkl")
# Load model
with open(model_path, "rb") as f:
    rf_model = pickle.load(f)

# Streamlit 

st.set_page_config(page_title="ChatGPT Reviews Sentiment App", layout="wide")

# Sidebar navigation
pages = ["Home"]
choice = st.sidebar.radio("Navigate", pages)

# 1. HOME PAGE

if choice == "Home":
    st.title("📊 ChatGPT Reviews Sentiment Analysis")
    st.markdown("""
    **AI Echo: Your Smartest Conversational Partner** is a sentiment analysis project 
    designed to uncover insights from user reviews of the ChatGPT application. 
    By applying **Natural Language Processing (NLP)**, **Machine Learning**, 
    and **Deep Learning techniques**, this project classifies reviews into 
    **Positive, Neutral, or Negative** sentiments.  

    The goal is to understand customer experiences, identify areas for improvement, 
    and provide actionable insights for enhancing user satisfaction. Through 
    **data preprocessing, exploratory data analysis, sentiment modeling, and visualization**, 
    the project not only highlights key trends but also compares user feedback across 
    **ratings, platforms, locations, versions, and verified purchases**.  

    Deployed as an interactive **Streamlit dashboard**, the app allows users to:  
    - Explore sentiment distributions and trends over time  
    - Compare experiences across platforms and regions  
    - Visualize keyword patterns in positive and negative reviews  
    - Analyze satisfaction by ChatGPT versions and user categories  

    This project bridges **Customer Experience and Business Analytics**, offering 
    a data-driven approach to improving product features, customer engagement, 
    and brand reputation management. 
    """)
#-----------------------------------2.Sentiment Analysis 10 Question-------------------
# Sidebar
question = st.sidebar.selectbox("Select a Question 10", [
    "1. Overall Sentiment Distribution",
    "2. Sentiment vs Rating",
    "3. Keywords per Sentiment",
    "4. Sentiment Trend Over Time",
    "5. Verified vs Non-Verified Users",
    "6. Review Length vs Sentiment",
    "7. Sentiment by Location",
    "8. Sentiment by Platform",
    "9. Sentiment by ChatGPT Version",
    "10. Negative Feedback Themes"
])
# ============================
# 1. Overall Sentiment Distribution
# ============================
if question.startswith("1"):
    st.subheader("Overall Sentiment of User Reviews")
    sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100
    st.bar_chart(sentiment_counts)

# ============================
# 2. Sentiment vs Rating
# ============================
elif question.startswith("2"):
    st.subheader("How Does Sentiment Vary by Rating?")
    crosstab = pd.crosstab(df['rating'], df['sentiment'], normalize="index") * 100
    st.write(crosstab)
    st.bar_chart(crosstab)

# ============================
# 3. Keywords per Sentiment
# ============================
elif question.startswith("3"):
    st.subheader("Keywords Associated with Each Sentiment")
    sentiment_choice = st.selectbox("Choose Sentiment", df['sentiment'].unique())
    text = " ".join(df[df['sentiment']==sentiment_choice]['review'].astype(str))
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    st.image(wc.to_array())

# ============================
# 4. Sentiment Trend Over Time
# ============================
elif question.startswith("4"):
    st.subheader("How Sentiment Has Changed Over Time")

    # 👇 Replace 'date' and 'sentiment' with your actual column names if different
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Group by month (you can change freq="W" for weekly)
    trend = df.groupby([pd.Grouper(key="date", freq="M"), "sentiment"]).size().unstack(fill_value=0)

    # Display line chart
    st.line_chart(trend)


# ============================
# 5. Verified vs Non-Verified
# ============================
elif question.startswith("5"):
    st.subheader("Do Verified Users Leave Different Reviews?")
    crosstab = pd.crosstab(df['verified_purchase'], df['sentiment'], normalize="index") * 100
    st.write(crosstab)
    st.bar_chart(crosstab)

# ============================
# 6. Review Length vs Sentiment
# ============================
elif question.startswith("6"):
    st.subheader("Are Longer Reviews More Positive or Negative?")
    length_stats = df.groupby("sentiment")['review_length'].mean()
    st.write(length_stats)
    sns.boxplot(x="sentiment", y="review_length", data=df)
    st.pyplot(plt.gcf())

# ============================
# 7. Sentiment by Location
# ============================
elif question.startswith("7"):
    st.subheader("Which Locations Show Most Positive/Negative Sentiment?")
    loc_counts = pd.crosstab(df['location'], df['sentiment'], normalize="index") * 100
    st.write(loc_counts)
    st.bar_chart(loc_counts)

# ============================
# 8. Sentiment by Platform
# ============================
elif question.startswith("8"):
    st.subheader("Sentiment by Platform (Web vs Mobile)")
    plat_counts = pd.crosstab(df['platform'], df['sentiment'], normalize="index") * 100
    st.write(plat_counts)
    st.bar_chart(plat_counts)

# ============================
# 9. Sentiment by Version
# ============================
elif question.startswith("9"):
    st.subheader("Sentiment Across ChatGPT Versions")
    ver_counts = pd.crosstab(df['version'], df['sentiment'], normalize="index") * 100
    st.write(ver_counts)
    st.bar_chart(ver_counts)

# ============================
# 10. Negative Feedback Themes
# ============================
elif question.startswith("10"):
    st.subheader("Common Themes in Negative Reviews")
    negative_text = " ".join(df[df['sentiment']=="Negative"]['review'].astype(str))
    wc = WordCloud(width=800, height=400, background_color="white").generate(negative_text)
    st.image(wc.to_array())
    st.write("Word Cloud shows the recurring pain points in negative feedback.")


















