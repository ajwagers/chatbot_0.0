# nltk_utils.py
# This file preps the data to be used by the model.  
# First, the sentence is split into words or characters (tokenized).
# Second, the word are reduced to their "stem" by chopping the end of the words off.  
#    This allows us to match roots of words rather than each different form of the word.
# Third, the tokenized sentence is translated into a vector of 0's and 1's based on a "bag of words".
# The next step is to feed this vector to the model and receive the results.

import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    examples:
    "How would you spend $100000000?"
    ["how","would","you","spend","$","100000000","?"]
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    stemming = find the root form of the word
    PorterStemmer is the flavor of stemming chosen
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag
