import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()

# AI model code
HF_API_KEY = os.getenv("HF_API_KEY")

API_URL = "https://api-inference.huggingface.co/models/bhadresh-savani/bert-base-go-emotion"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# Set page title
st.title("GoEmotions Dashboard - Analyzing Emotions in Text")

# Add page description
description = "The GoEmotions Dashboard is a web-based user interface for analyzing emotions in text. The dashboard is powered by a pre-trained natural language processing model that can detect emotions in text input. Users can input any text and the dashboard will display the detected emotions in a set of gauges, with each gauge representing the intensity of a specific emotion category. The gauge colors are based on a predefined color map for each emotion category. This dashboard is useful for anyone who wants to understand the emotional content of a text, including content creators, marketers, and researchers."
st.markdown(description)

def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

# Define color map for each emotion category
color_map = {
        'admiration': ['#1f77b4', '#aec7e8', '#ff7f0e', '#d62728'],
        'amusement': ['#ff7f0e', '#ffbb78', '#2ca02c', '#d62728'],
        'anger': ['#d62728', '#ff9896', '#2ca02c', '#bcbd22'],
        'annoyance': ['#d62728', '#ff9896', '#2ca02c', '#bcbd22'],
        'approval': ['#1f77b4', '#aec7e8', '#ff7f0e', '#d62728'],
        'caring': ['#1f77b4', '#aec7e8', '#ff7f0e', '#d62728'],
        'confusion': ['#9467bd', '#c5b0d5', '#ff7f0e', '#d62728'],
        'curiosity': ['#9467bd', '#c5b0d5', '#ff7f0e', '#d62728'],
        'desire': ['#ff7f0e', '#ffbb78', '#2ca02c', '#d62728'],
        'disappointment': ['#d62728', '#ff9896', '#2ca02c', '#bcbd22'],
        'disapproval': ['#d62728', '#ff9896', '#2ca02c', '#bcbd22'],
        'disgust': ['#d62728', '#ff9896', '#2ca02c', '#bcbd22'],
        'embarrassment': ['#9467bd', '#c5b0d5', '#ff7f0e', '#d62728'],
        'excitement': ['#ff7f0e', '#ffbb78', '#2ca02c', '#d62728'],
        'fear': ['#d62728', '#ff9896', '#2ca02c', '#bcbd22'],
        'gratitude': ['#1f77b4', '#aec7e8', '#ff7f0e', '#d62728'],
        'grief': ['#d62728', '#ff9896', '#2ca02c', '#bcbd22'],
        'joy': ['#ff7f0e', '#ffbb78', '#2ca02c', '#d62728'],
        'love': ['#1f77b4', '#aec7e8', '#ff7f0e', '#d62728'],
        'nervousness': ['#9467bd', '#c5b0d5', '#ff7f0e', '#d62728'],
        'optimism': ['#1f77b4', '#aec7e8', '#ff7f0e', '#d62728'],
        'pride': ['#1f77b4', '#aec7e8', '#ff7f0e', '#d62728'],
        'realization': ['#9467bd', '#c5b0d5', '#ff7f0e', '#d62728'],
        'relief': ['#1f77b4', '#aec7e8', '#ff7f0e', '#d62728'],
        'remorse': ['#d62728', '#ff9896', '#2ca02c', '#bcbd22'],
        'sadness': ['#d62728', '#ff9896', '#2ca02c', '#bcbd22'],
        'surprise': ['#9467bd', '#c5b0d5', '#ff7f0e', '#d62728'],
        'neutral': ['#1f77b4', '#aec7e8', '#ff7f0e', '#d62728']
}

# Define default options
default_options = [
    "I'm so excited for my vacation next week!",
    "I'm feeling so stressed about work.",
    "I just received great news from my doctor!",
    "I can't wait to see my best friend tomorrow.",
    "I'm feeling so lonely and sad today."
]


# Create dropdown with default options
selected_option = st.selectbox("Select a default option or enter your own text:", default_options)

# Display text input with selected option as default value
text_input = st.text_input("Enter text to analyze emotions:", selected_option)

# Add submit button
if st.button("Submit"):

    # Call API and get predicted probabilities for each emotion category
    response = query(text_input)
    predicted_probabilities = response[0]

    # Sort the predicted probabilities in descending order
    sorted_probs = sorted(predicted_probabilities, key=lambda x: x['score'], reverse=True)

    # Get the top 4 emotion categories and their scores
    top_emotions = sorted_probs[:4]
    top_scores = [e['score'] for e in top_emotions]

    # Normalize the scores so that they add up to 100%
    total = sum(top_scores)
    normalized_scores = [score/total * 100 for score in top_scores]

    # Create the gauge charts for the top 4 emotion categories using the normalized scores
    fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
                                                [{'type': 'indicator'}, {'type': 'indicator'}]],
                        vertical_spacing=0.4)

    for i, emotion in enumerate(top_emotions):
        category = emotion['label']
        color = color_map[category]
        value = normalized_scores[i]
        row = i // 2 + 1
        col = i % 2 + 1
        fig.add_trace(go.Indicator(
            domain={'x': [0, 1], 'y': [0, 1]},
            value=value,
            mode="gauge+number",
            title={'text': category.capitalize()},
            gauge={'axis': {'range': [None, 100]},
                'bar': {'color': color[3]},
                'bgcolor': 'white',
                'borderwidth': 2,
                'bordercolor': color[1],
                'steps': [{'range': [0, 33], 'color': color[0]},
                            {'range': [33, 66], 'color': color[1]},
                            {'range': [66, 100], 'color': color[2]}],
                'threshold': {'line': {'color': "black", 'width': 4},
                                'thickness': 0.5,
                                'value': 50}}), row=row, col=col)


    # Update layout
    fig.update_layout(height=400, margin=dict(t=50, b=5, l=0, r=0))

    # Display gauge charts
    st.plotly_chart(fig, use_container_width=True)

