import re
import string
import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords
import json

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

