import re
import string
import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords
import json
from nrclex import NRCLex


def get_word_lengths(input_string):
    """
    This function takes a string as input and returns a list of lengths of each word in the string.

    Parameters:
    input_string (str): The input string from which word lengths are to be calculated.

    Returns:
    list: A list of integers representing the lengths of each word in the input string.
    """
    # Split the input string into words using the default delimiter (whitespace)
    words = input_string.split()
    
    # Calculate the length of each word
    word_lengths = [len(word) for word in words]
    
    return word_lengths

def wp(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    text = re.sub(" reuters ","",text)
    text = re.sub("  "," ",text)
    return text

def remove_stopwords(text, cleaned_stopwords):
    text = " ".join([word for word in text.split() if word not in cleaned_stopwords])
    return text

def check_stopwords(emotional_words_path = r'/Users/AdamHarris/Documents/neuromatch_nlp/Neuromatch_NLP/dataset/nrc_en.json'):
    s_words = stopwords.words('english')
    with open(emotional_words_path, 'r') as file:
        emotional_words = json.load(file)
    cleaned_stopwords = [word for word in s_words if word not in emotional_words.keys()]
    return cleaned_stopwords

def preprocess_all(x, json_path = r'/Users/AdamHarris/Documents/neuromatch_nlp/Neuromatch_NLP/dataset/nrc_en.json'):
    x = wp(x)
    cleaned_stopwords = check_stopwords(json_path)
    x = remove_stopwords(x, cleaned_stopwords)
    return x

# def emotion_score_article(article):
#     if not isinstance(article, str):
#         print(article)
#     norm_len = len(article)
#     e_scores = NRCLex(article)
#     norm_e_scores = {}
#     norm_e_scores_vect = []
#     for i in e_scores.raw_emotion_scores.keys():
#         norm_e_scores[i]= e_scores.raw_emotion_scores[i]/norm_len
#         norm_e_scores_vect.append(norm_e_scores[i])
#     return norm_e_scores_vect, norm_e_scores
def emotion_score_article(article, emotional_labels=['disgust',
                                                    'trust',
                                                    'negative',
                                                    'sadness',
                                                    'anticipation',
                                                    'joy',
                                                    'positive',
                                                    'anger',
                                                    'fear',
                                                    'surprise']):
    if not isinstance(article, str):
        print(article)
    norm_len = len(article)
    e_scores = NRCLex(article)
    norm_e_scores = {}
    norm_e_scores_vect = []
    #for i in e_scores.raw_emotion_scores.keys():
    for i in emotional_labels:
        if i not in e_scores.raw_emotion_scores.keys():
            norm_e_scores[i]=0
            norm_e_scores_vect.append(0)
        else:
            norm_e_scores[i]= e_scores.raw_emotion_scores[i]/norm_len
            norm_e_scores_vect.append(norm_e_scores[i])
    return norm_e_scores_vect, norm_e_scores