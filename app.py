import pandas as pd
import boto3
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import streamlit as st
import io
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import os

# Tokenization function for text analysis
def tokenize_and_filter(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return filtered_words

# Function to configure Boto3 to use with Backblaze B2
def configure_boto3(key_id, application_key):
    return boto3.Session(
        aws_access_key_id=key_id,
        aws_secret_access_key=application_key
    )

# Function to load data from Backblaze B2
def load_data_from_b2(session, bucket_name, object_name):
    s3 = session.resource('s3', endpoint_url='https://s3.us-east-005.backblazeb2.com')
    obj = s3.Object(bucket_name, object_name)
    data = obj.get()['Body'].read()
    return pd.read_csv(io.StringIO(data.decode('utf-8')))  

# Fetch the environment variables directly
key_id = os.getenv('B2_MASTER_KEY_ID')
secret_key = os.getenv('B2_MASTER_APPLICATION_KEY')
bucket_name = 'streamlit1'  
file_path = 'all_haiku.csv'  

# Configure Boto3 with Backblaze B2 credentials
session = configure_boto3(key_id, secret_key)

# Streamlit app function
def haiku_app():
    st.title('Haiku Analysis App')

    # Loading the haiku dataset
    haiku_df = load_data_from_b2(session, bucket_name, file_path) 
    haiku_df['full_haiku'] = haiku_df[['0', '1', '2']].astype(str).agg(' '.join, axis=1)

    # Visualization
    st.subheader('Most Common Words in Haikus')
    haiku_df['tokenized'] = haiku_df['full_haiku'].apply(tokenize_and_filter)
    all_words = [word for sublist in haiku_df['tokenized'] for word in sublist]
    word_freq = Counter(all_words)
    top_N = 20
    most_common_words = word_freq.most_common(top_N)

    fig, ax = plt.subplots()
    words, counts = zip(*most_common_words)
    sns.barplot(x=list(words), y=list(counts), ax=ax)   
    ax.set_title('Top 20 Most Common Words in Haikus')
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    # Display 
    st.subheader('Sample Haikus')
    st.dataframe(haiku_df.sample(10)) 

# Run the app
haiku_app()
