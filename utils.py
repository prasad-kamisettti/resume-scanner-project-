import nltk
import spacy
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load the spaCy English model
import en_core_web_sm
nlp = en_core_web_sm.load()

# Load stopwords from NLTK
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    # Make all text lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    #Break into words
    words = word_tokenize(text)

    #Remove stopwords and keep only words (ignore numbers/symbols)
    words = [word for word in words if word not in stop_words and word.isalpha()]

    #Lemmatize (get root words)
    doc = nlp(" ".join(words))
    lemmatized_words = [token.lemma_ for token in doc]

    #Return cleaned text
    return " ".join(lemmatized_words)
