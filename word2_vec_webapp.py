import streamlit as st
import numpy as np
import pandas as pd
import word2vec_sim as w2v

if 'clicked_get_similarity_score' not in st.session_state:
    st.session_state.clicked_get_similarity_score = False

if 'clicked_get_similar_words' not in st.session_state:
    st.session_state.clicked_get_similar_words = False

def click_get_similarity_score():
    st.session_state.clicked_get_similarity_score = True

def click_get_similar_words():
    st.session_state.clicked_get_similar_words = True

@st.cache_resource
def load_model(model_name):
    try:
       model =  w2v.load_model(model_name)
       return model
    except Exception as ex:
        print(f" Exception {ex} occurred. ")


st.header("Calculate Sentence Similarity: ", divider='rainbow')


model_name = st.selectbox(
    "Which model you would like to select ?",
    ("glove-wiki-gigaword-50","glove-wiki-gigaword-100", "glove-twitter-25", "glove-twitter-50")
    )

text1 = st.text_input(label = "Enter text")

text2 = st.text_input(label = "Enter text to compare")

st.divider()

calculate_similarity = st.button(label= "calculate_similarity", on_click= click_get_similarity_score)

if st.session_state.clicked_get_similarity_score:
    
    model = load_model(model_name)
    
    # Get paragraph vectors
    vector1 = w2v.get_paragraph_vector(model, text1)
    vector2 = w2v.get_paragraph_vector(model, text2)

    # Calculate similarity
    similarity = w2v.cosine_similarity(vector1, vector2)
    st.write(f"Cosine similarity between the paragraphs: {similarity}")


st.divider()

get_similar_words = st.button(label = "get_similar_words", on_click= click_get_similar_words)

if st.session_state.clicked_get_similar_words:

    model = load_model(model_name)

    words_list1  = w2v.preprocess(text1)
    words_list2 =  w2v.preprocess(text2)

    similarity_scores_list = w2v.get_word_vector_scores(model, words_list1, words_list2)

    similarity_scores_df = pd.DataFrame( similarity_scores_list, columns=["text1","text2", "score"] )

    top_similar_words_df = similarity_scores_df.groupby('text1').apply(lambda x: x.nlargest(5, 'score')).reset_index(drop=True).drop_duplicates()

    st.dataframe(top_similar_words_df)

st.divider()