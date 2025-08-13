import string
from functools import lru_cache

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


@lru_cache(maxsize=1)
def _ensure_nltk():
    """Ensure punkt + stopwords are available (quietly download if missing)."""
    for pkg, path in [("punkt", "tokenizers/punkt"), ("stopwords", "corpora/stopwords")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


@lru_cache(maxsize=1)
def _nlp():
    """Load (or download) spaCy model only once."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


def preprocess_text(text: str) -> str:
    """Lowercase → strip punctuation → tokenize → remove stopwords → lemmatize."""
    _ensure_nltk()
    nlp = _nlp()

    t = (text or "").lower().translate(str.maketrans("", "", string.punctuation))
    words = [w for w in word_tokenize(t) if w.isalpha()]
    sw = set(stopwords.words("english"))
    words = [w for w in words if w not in sw]

    doc = nlp(" ".join(words))
    return " ".join(tok.lemma_ for tok in doc)
