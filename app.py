#x = st.slider('Select a value')
#st.write(x, 'squared is', x * x)

import streamlit as st
from transformers import pipeline

# Load the summarization model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")  # smaller version of the model

# Default article text
DEFAULT_ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
2010 marriage license application, according to court documents.
"""

# Create a text area for user input
ARTICLE = st.sidebar.text_area('Enter Article', DEFAULT_ARTICLE, height=150)

# Define the summarization function
def summarize(txt):
    st-write('\n\n')
    st.write(txt[:100])  # Display the first 100 characters of the article
    st.write('--------------------------------------------------------------')
    summary = summarizer(txt, max_length=130, min_length=30, do_sample=False)
    st.write(summary[0]['summary_text'])

# Create a button and trigger the summarize function when clicked
if st.sidebar.button('Summarize Article'):
    summarize(ARTICLE)
else:
    st.warning('👈 Please enter Article!')
