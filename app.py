#x = st.slider('Select a value')
#st.write(x, 'squared is', x * x)

import streamlit as st
from transformers import pipeline
import ast

# Load the summarization model
#summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")  # smaller version of the model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Default article text
# DEFAULT_ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
# A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
# Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
# In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
# Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
# 2010 marriage license application, according to court documents.
# """

# DEFAULT_STATEMENT = ""

# # Create a text area for user input
# STATEMENT = st.sidebar.text_area('Enter Article (String)', DEFAULT_ARTICLE, height=150)

# Define the summarization function
def summarize(txt):
    st.write('\n\n')
    st.write(txt[:100])  # Display the first 100 characters of the article
    st.write('--------------------------------------------------------------')
    summary = summarizer(txt, max_length=130, min_length=30, do_sample=False)
    st.write(summary[0]['summary_text'])

DEFAULT_STATEMENT = ""
# Create a text area for user input
STATEMENT = st.sidebar.text_area('Enter Statement (String or List of Strings)', DEFAULT_STATEMENT, height=150)

# Enable the button only if there is text in the SENTIMENT variable
if STATEMENT:
    if st.sidebar.button('Summarize Statement'):
        # Call your Summarize function here
        #st.write(f"Summarizing: {STATEMENT}")
        summarize(STATEMENT)  # Directly pass the STATEMENT
else:
    st.sidebar.button('Summarize Statement', disabled=True)
    st.warning('👈 Please enter Statement!')    
    
# # Create a button and trigger the summarize function when clicked
# if st.sidebar.button('Summarize Article'):
#     summarize(ARTICLE)
# else:
#     st.warning('👈 Please enter Article!')



#################################


# Initialize the sentiment analysis pipeline
# No model was supplied, defaulted to distilbert-base-uncased-finetuned-sst-2-english
sentiment_pipeline = pipeline("sentiment-analysis")

# Default article text
# DEFAULT_SENTIMENT = """[
#     "I'm so happy today!",
#     "This is the worst experience ever.",
#     "It's a decent product, nothing special."
# ]"""

# DEFAULT_SENTIMENT = "I'm so happy today!"


def is_valid_list_string(string):
    try:
        result = ast.literal_eval(string)
        return isinstance(result, list)
    except (ValueError, SyntaxError):
        return False
        
# Define the summarization function
def analyze(txt):
    
    st.write('\n\n')
    #st.write(txt[:100])  # Display the first 100 characters of the article
    st.write('--------------------------------------------------------------')
    
    # Display the results
    #if type(txt_converted) == 'list':
    #if isinstance(txt_converted, list):
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
        # Call your Analyze function here
        #st.write(f"Summarizing: {SENTIMENT}")
        analyze(SENTIMENT)  # Directly pass the SENTIMENT
else:
    st.sidebar.button('Summarize Sentiment', disabled=True)
    st.warning('👈 Please enter Sentiment!')    
    
# # Create a button and trigger the summarize function when clicked
# if st.sidebar.button('Summarize Sentiment'):
#     #ast.literal_eval() is a function in Python that safely evaluates a string containing a valid Python expression, 
#     #such as lists, dictionaries, tuples, sets, integers, and floats. It parses the string and returns the corresponding 
#     #Python object, without executing any arbitrary code, which makes it safer than using eval().    
#     #summarize(str(SENTIMENT)) #explicitly change SENTIMENT to string so that even when ypu provide unquoted string, it still works
#     analyze(SENTIMENT)  # Directly pass the SENTIMENT
# else:
#     st.warning('👈 Please enter Sentiment!')