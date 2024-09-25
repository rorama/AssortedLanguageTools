# Natural Language Tools
# Richard Orama - September 2024

#x = st.slider('Select a value')
#st.write(x, 'squared is', x * x)

import streamlit as st
from transformers import pipeline
import ast

st.title("Assorted Language Tools - Orama's AI Craze")

################ CHAT BOT #################

# Load the GPT model
generator = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")

# Streamlit chat UI
#st.title("GPT-3 Chatbox")

# user_input = st.text_input("You: ", "Hello, how are you?")

# if user_input:
#     response = generator(user_input, max_length=100, num_return_sequences=1)[0]['generated_text']
#     st.write(f"GPT-3: {response}")

# Define the summarization function
def chat(txt):
    st.write('\n\n')
    #st.write(txt[:100])  # Display the first 100 characters of the article
    #st.write('--------------------------------------------------------------')
    #summary = summarizer(txt, max_length=500, min_length=30, do_sample=False)
    #st.write(summary[0]['summary_text'])
    response = generator(txt, max_length=500, num_return_sequences=1)[0]['generated_text']
    st.write(f"GPT-3: {response}")    
    
DEFAULT_CHAT = ""
# Create a text area for user input
CHAT = st.sidebar.text_area('Enter Chat (String)', DEFAULT_CHAT, height=150)

# Enable the button only if there is text in the CHAT variable
if CHAT:
    if st.sidebar.button('Chat Statement'):
        # Call your Summarize function here
        chat(CHAT)  # Directly pass the your
else:
    st.sidebar.button('Chat Statement', disabled=True)
    st.warning('👈 Please enter Chat!')    
    
    
################ STATEMENT SUMMARIZATION #################

# Load the summarization model
#summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")  # smaller version of the model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Define the summarization function
def summarize_statement(txt):
    st.write('\n\n')
    #st.write(txt[:100])  # Display the first 100 characters of the article
    #st.write('--------------------------------------------------------------')
    summary = summarizer(txt, max_length=500, min_length=30, do_sample=False)
    st.write(summary[0]['summary_text'])

DEFAULT_STATEMENT = ""
# Create a text area for user input
STATEMENT = st.sidebar.text_area('Enter Statement (String)', DEFAULT_STATEMENT, height=150)

# Enable the button only if there is text in the SENTIMENT variable
if STATEMENT:
    if st.sidebar.button('Summarize Statement'):
        # Call your Summarize function here
        summarize_statement(STATEMENT)  # Directly pass the STATEMENT
else:
    st.sidebar.button('Summarize Statement', disabled=True)
    st.warning('👈 Please enter Statement!')    
    

################ SENTIMENT ANALYSIS #################

# Initialize the sentiment analysis pipeline
# No model was supplied, defaulted to distilbert-base-uncased-finetuned-sst-2-english
sentiment_pipeline = pipeline("sentiment-analysis")

def is_valid_list_string(string):
    try:
        result = ast.literal_eval(string)
        return isinstance(result, list)
    except (ValueError, SyntaxError):
        return False
        
# Define the summarization function
def analyze_sentiment(txt):
    
    st.write('\n\n')
    #st.write(txt[:100])  # Display the first 100 characters of the article
    #st.write('--------------------------------------------------------------')
    
    # Display the results
    if is_valid_list_string(txt):        
        txt_converted = ast.literal_eval(txt) #convert string to actual content, e.g. list
        # Perform Hugging sentiment analysis on multiple texts
        results = sentiment_pipeline(txt_converted)        
        for i, text in enumerate(txt_converted):
            st.write(f"Text: {text}")
            st.write(f"Sentiment: {results[i]['label']}, Score: {results[i]['score']:.2f}\n")
    else:
        # Perform Hugging sentiment analysis on multiple texts
        results = sentiment_pipeline(txt)        
        st.write(f"Text: {txt}")
        st.write(f"Sentiment: {results[0]['label']}, Score: {results[0]['score']:.2f}\n")


DEFAULT_SENTIMENT = ""
# Create a text area for user input
SENTIMENT = st.sidebar.text_area('Enter Sentiment (String or List of Strings)', DEFAULT_SENTIMENT, height=150)

# Enable the button only if there is text in the SENTIMENT variable
if SENTIMENT:
    if st.sidebar.button('Analyze Sentiment'):
        analyze_sentiment(SENTIMENT)  # Directly pass the SENTIMENT
else:
    st.sidebar.button('Analyze Sentiment', disabled=True)
    st.warning('👈 Please enter Sentiment!')    
