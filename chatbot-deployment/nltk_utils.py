import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

# Download the necessary resources for NLTK (if not already downloaded)
# nltk.download('punkt')

stemmer = PorterStemmer()


def tokenize(sentence, language='english'):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    if language == 'english':
        return nltk.word_tokenize(sentence)
    elif language == 'tamil':
        # For illustration purposes, let's assume a simple space-based tokenization for Tamil
        return sentence.split()
    else:
        raise ValueError("Unsupported language")


def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
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


def get_bow(sentence, language='english'):
    """
    Tokenize the input sentence and return the bag of words representation.
    """
    # Replace this list with your specific list of words
    your_list_of_words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]

    tokenized_sentence = tokenize(sentence, language=language)
    bow = bag_of_words(tokenized_sentence, your_list_of_words)

    return bow


# Example usage
sentence_english = "Hello, how are you?"
bow_english = get_bow(sentence_english, language='english')
print("Bag of Words (English):", bow_english)

sentence_tamil = "வணக்கம் எங்கள் குடும்பத்தில் வரவேற்கிறோம்"
bow_tamil = get_bow(sentence_tamil, language='tamil')
print("Bag of Words (Tamil):", bow_tamil)
