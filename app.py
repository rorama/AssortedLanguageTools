# Simply Assorted Language Tools (SALT)
# Richard Orama - September 2024

#x = st.slider('Select a value')
#st.write(x, 'squared is', x * x)

import streamlit as st
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
import ast

#st.title("Assorted Language Tools")
st.markdown("<h1 style='text-align: center; font-size: 40px; color: yellow;'>O - S A L T</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; font-size: 16px;'>Orama's Selectively Assorted Language Tools</h3>", unsafe_allow_html=True)
#st.markdown("---")  # Horizontal line to separate content from the rest
#st.markdown("<h3 style='text-align: center; font-size: 20px; color: yellow;'>Orama's AI Craze</h3>", unsafe_allow_html=True)


################ SENTIMENT ANALYSIS - side bar - pippeline #################

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
    
    #st.write('\n\n')
    #st.write(txt[:100])  # Display the first 100 characters of the article
    #st.write('--------------------------------------------------------------')
    
    # Display the results
    if is_valid_list_string(txt):        
        txt_converted = ast.literal_eval(txt) #convert string to actual content, e.g. list
        # Perform Hugging sentiment analysis on multiple texts
        results = sentiment_pipeline(txt_converted)        
        for i, text in enumerate(txt_converted):
            st.sidebar.write(f"Text: {text}")
            st.sidebar.write(f"Sentiment: {results[i]['label']}, Score: {results[i]['score']:.2f}\n")
    else:
        # Perform Hugging sentiment analysis on multiple texts
        results = sentiment_pipeline(txt)        
        st.sidebar.write(f"Text: {txt}")
        st.sidebar.write(f"Sentiment: {results[0]['label']}, Score: {results[0]['score']:.2f}\n")

st.sidebar.markdown("<h3 style='text-align: center; font-size: 16px; background-color: white; color: black;'>Sentiment Analysis - Pipeline</h3>", unsafe_allow_html=True)
DEFAULT_SENTIMENT = ""
# Create a text area for user input
SENTIMENT = st.sidebar.text_area('Enter Sentiment (String or List of Strings)', DEFAULT_SENTIMENT, height=150)

# Enable the button only if there is text in the SENTIMENT variable
if SENTIMENT:
    if st.sidebar.button('Analyze Sentiment'):
        analyze_sentiment(SENTIMENT)  # Directly pass the SENTIMENT
else:
    st.sidebar.button('Analyze Sentiment', disabled=True)
    #st.warning('👈 Please enter Sentiment!')   

    
################ STATEMENT SUMMARIZATION - side bar - tokenizer #################

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the summarization model and tokenizer
MODEL_NAME = "facebook/bart-large-cnn"  # A commonly used summarization model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

st.sidebar.markdown("<h3 style='text-align: center; font-size: 16px; background-color: white; color: black;'>Text Summarization - BART Tokenizer</h3>", unsafe_allow_html=True)
DEFAULT_STATEMENT = ""
# Create a text area for user input
STATEMENT = st.sidebar.text_area('Enter Statement (String1)', DEFAULT_STATEMENT, height=150)

# Enable the button only if there is text in the SENTIMENT variable
if STATEMENT:
    if st.sidebar.button('Summarize Statement1'):
        # Call your Summarize function here
        # summarize_statement(STATEMENT)  # Directly pass the STATEMENT

        # Tokenize input article
        inputs = tokenizer(STATEMENT, return_tensors="pt", truncation=True, padding="longest", max_length=1024)

        # Generate summary
        summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)

        # Decode summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Display the summary
        st.sidebar.write("**Summary:**")
        st.sidebar.write(summary)        
else:
    st.sidebar.button('Summarize Statement1', disabled=True)
    #st.warning('👈 Please enter Statement!')   
    

################ STATEMENT SUMMARIZATION - side bar - pipeline #################

# Load the summarization model
#summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")  # smaller version of the model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

st.sidebar.markdown("<h3 style='text-align: center; font-size: 16px; background-color: white; color: black;'>Text Summarization - BART Pipeline</h3>", unsafe_allow_html=True)
DEFAULT_STATEMENT = ""
# Create a text area for user input
STATEMENT = st.sidebar.text_area('Enter Statement (String2)', DEFAULT_STATEMENT, height=150)

# Enable the button only if there is text in the SENTIMENT variable
if STATEMENT:
    if st.sidebar.button('Summarize Statement2'):
        # Call your Summarize function here
        #st.write('\n\n')
        summary = summarizer(STATEMENT, max_length=500, min_length=30, do_sample=False)
        st.sidebar.write(summary[0]['summary_text'])
        #summarize_statement(STATEMENT)  # Directly pass the STATEMENT
else:
    st.sidebar.button('Summarize Statement2', disabled=True)
    #st.warning('👈 Please enter Statement!')    
    


# ################ CHAT BOT - main area #################

# # Load the GPT model
# generator = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")

# # Streamlit chat UI
# #st.title("GPT-3 Chatbox")

# # user_input = st.text_input("You: ", "Hello, how are you?")

# # if user_input:
# #     response = generator(user_input, max_length=100, num_return_sequences=1)[0]['generated_text']
# #     st.write(f"GPT-3: {response}")

# # Define the summarization function
# def chat(txt):
#     st.write('\n\n')
#     #st.write(txt[:100])  # Display the first 100 characters of the article
#     #st.write('--------------------------------------------------------------')
#     #summary = summarizer(txt, max_length=500, min_length=30, do_sample=False)
#     #st.write(summary[0]['summary_text'])
#     response = generator(txt, max_length=500, num_return_sequences=1)[0]['generated_text']
#     st.write(f"GPT-3: {response}")    
    
# DEFAULT_CHAT = ""
# # Create a text area for user input
# CHAT = st.sidebar.text_area('Enter Chat (String)', DEFAULT_CHAT, height=150)

# # Enable the button only if there is text in the CHAT variable
# if CHAT:
#     if st.sidebar.button('Chat Statement'):
#         # Call your Summarize function here
#         chat(CHAT)  # Directly pass the your
# else:
#     st.sidebar.button('Chat Statement', disabled=True)
#     st.warning('👈 Please enter Chat!')    




# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2-medium" # "gpt2"  # Use "gpt-3.5-turbo" or another model from Hugging Face if needed
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Initialize the text generation pipeline
gpt_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Streamlit UI
st.markdown("<h3 style='text-align: center; font-size: 20px; background-color: white; color: black;'>Chat - gpt2-medium</h3>", unsafe_allow_html=True)

if 'conversation' not in st.session_state:
    st.session_state.conversation = ""

def chat_with_gpt(user_input):
    # Append user input to the conversation
    st.session_state.conversation += f"User: {user_input}\n"

    # Generate response
    response = gpt_pipeline(user_input, max_length=100, num_return_sequences=1)[0]['generated_text']
    response_text = response.replace(user_input, '')  # Strip the user input part from response

    # Append GPT's response to the conversation
    st.session_state.conversation += f"GPT: {response_text}\n"
    return response_text

# Text input for user query
user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input:
        chat_with_gpt(user_input)

# Display conversation history
st.text_area("Conversation", value=st.session_state.conversation, height=400)


#############

# # LLaMA 7B model from Hugging Face
# # MODEL_NAME = "huggyllama/llama-7b"  # Example of a LLaMA model

# # Try this OpenAssistant model available on Hugging Face
# MODEL_NAME = "OpenAssistant/oasst-sft-1-pythia-12b"  # Example of an OpenAssistant model

# import streamlit as st
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# # Load the model and tokenizer (OpenAssistant or LLaMA)
# MODEL_NAME = "OpenAssistant/oasst-sft-1-pythia-12b"  # Replace with "huggyllama/llama-7b" for LLaMA
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# # Streamlit UI for input
# st.markdown("<h3 style='text-align: center; font-size: 20px;'>Chat with OpenAssistant/LLaMA</h3>", unsafe_allow_html=True)

# # Input text area
# user_input = st.text_area("You:", "", height=150)

# if st.button('Generate Response'):
#     if user_input:
#         # Tokenize the input and generate response
#         inputs = tokenizer(user_input, return_tensors="pt")
#         outputs = model.generate(**inputs, max_length=150)

#         # Decode the generated response
#         response = tokenizer.decode(outputs[0], skip_special_tokens=True)

#         # Display the model's response
#         st.write("Assistant: ", response)
#     else:
#         st.warning('Please enter some text to get a response!')



# ################ END #################


# Add a footnote at the bottom
st.markdown("---")  # Horizontal line to separate content from footnote
st.markdown("<h3 style='text-align: center; font-size: 20px; color: yellow;'>Orama's AI Craze</h3>", unsafe_allow_html=True)