import gensim
import gensim.downloader as api
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd

def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")


# Function to compute paragraph vector 
def get_paragraph_vector(model, paragraph):
    words = gensim.utils.simple_preprocess(paragraph)
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)


# Function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))



def get_word_vectors(model,text):
    try:
        # preprocess
        words = gensim.utils.simple_preprocess(text)
        
        # get word vectors
        word_vectors = [model[word] for word in words if word in model]
        
        # take mean
        word_vectors_mean = np.mean(word_vectors, axis=1)
        
        return word_vectors_mean
        
    except Exception as ex:
        print(f" Exception {ex} occurred. ")

def get_word_vector_scores(model,words_list1, words_list2):
    """ get pairwise scores of word vectors """
    try:
        # create a list to store scores
        similarity_scores_list = []
        
        # loop through lists to get pairwise similarity scores
        for word1 in words_list1:
            for word2 in words_list2:
                if word1 in model and word2 in model:
                    word_similarity_val = model.similarity(word1,word2)
                    similarity_scores_list.append(list((word1,word2,word_similarity_val)))
                else :
                    continue
        
        return similarity_scores_list
        
    except Exception as ex:
        print(f" Exception {ex} occurred. ")



def load_model(model_name):
    """ loading pretrained model """
    try:
        model = api.load(model_name)
        return model
    except Exception as ex:
        print(f" Exception {ex} occurred. ")

def preprocess(text):
    """ loading pretrained model """
    try:
        # preprocess
        words = gensim.utils.simple_preprocess(text)
        return words
    except Exception as ex:
        print(f" Exception {ex} occurred. ")




if __name__ == "__main__" :
    
    # Load pre-trained Word2Vec model
    print(f" loading word2vec model")

    model = api.load('word2vec-google-news-300') 

    print(f"finished loading word2vec model")
    
    job_desp = '../input/job_description.txt'
    resume = '../input/job_description_2.txt'

    paragraph1 = read_file(job_desp)
    paragraph2 = read_file(resume)

    # Get paragraph vectors
    vector1 = get_paragraph_vector(model, paragraph1)
    vector2 = get_paragraph_vector(model, paragraph2)

    # Calculate similarity
    similarity = cosine_similarity(vector1, vector2)
    print(f"Cosine similarity between the paragraphs: {similarity}")

    # get word lists
    words_list1  = gensim.utils.simple_preprocess(paragraph1)
    words_list2 =  gensim.utils.simple_preprocess(paragraph2)

    similarity_scores_list = get_word_vector_scores(model, words_list1, words_list2)

    similarity_scores_df = pd.DataFrame( similarity_scores_list, columns=["job_desp_word","resume_word", "score"] )

    top_similar_words_df = similarity_scores_df.groupby('job_desp_word').apply(lambda x: x.nlargest(5, 'score')).reset_index(drop=True).drop_duplicates()

    print(f" Similar words between 2 paragraphs: \n  {top_similar_words_df}")

