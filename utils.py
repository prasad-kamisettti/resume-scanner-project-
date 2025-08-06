import nltk
import spacy
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Load stopwords from NLTK
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    # Step 1: Make all text lowercase
    text = text.lower()

    # Step 2: Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Step 3: Break into words
    words = word_tokenize(text)

    # Step 4: Remove stopwords and keep only words (ignore numbers/symbols)
    words = [word for word in words if word not in stop_words and word.isalpha()]

    # Step 5: Lemmatize (get root words)
    doc = nlp(" ".join(words))
    lemmatized_words = [token.lemma_ for token in doc]

    # Step 6: Return cleaned text
    return " ".join(lemmatized_words)
