import string
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Téléchargement automatique des ressources NLTK
for res in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"tokenizers/{res}")
    except LookupError:
        try:
            nltk.data.find(f"corpora/{res}")
        except LookupError:
            nltk.download(res, quiet=True)

STOP_EN = set(stopwords.words("english"))
PUNCT = set(string.punctuation) | {"«", "»", "’", "''", "``", "“", "”"}
EXCEPT_UPPER = {"NLP", "DATA", "USA"}

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def load_text(path: str | Path) -> str:
    """Tâche 1 : Chargement"""
    return " ".join(Path(path).read_text(encoding="utf-8").split())

def split_sentences(text: str) -> list[str]:
    """Tâche 2 : Segmentation en phrases"""
    return sent_tokenize(text, language="english")

def tokenize_words(sentences: list[str]) -> list[list[str]]:
    """Tâche 3 : Tokenisation en mots"""
    return [word_tokenize(s, language="english") for s in sentences]

def drop_stopwords(tokenized: list[list[str]]) -> list[list[str]]:
    """Tâche 4 : Suppression des stopwords"""
    return [[w for w in sent if w.lower() not in STOP_EN] for sent in tokenized]

def drop_punctuation(tokenized: list[list[str]]) -> list[list[str]]:
    """Tâche 5 : Suppression de la ponctuation"""
    return [[w for w in sent if w not in PUNCT] for sent in tokenized]

def to_lower_with_exceptions(tokenized: list[list[str]]) -> list[list[str]]:
    """Tâche 6 : Passage en minuscules avec exceptions"""
    return [[w if w in EXCEPT_UPPER else w.lower() for w in sent] for sent in tokenized]

def lemmatize_tokens(tokenized: list[list[str]]) -> list[list[str]]:
    """Tâche 7 : Lemmatisation"""
    return [[lemmatizer.lemmatize(w) for w in sent] for sent in tokenized]

if __name__ == "__main__":
    fichier_en = Path("data/corpus_en.txt")
    if fichier_en.exists():
        texte = load_text(fichier_en)
        phrases = split_sentences(texte)
        mots = tokenize_words(phrases)
        sans_stop = drop_stopwords(mots)
        sans_punct = drop_punctuation(sans_stop)
        minuscules = to_lower_with_exceptions(sans_punct)
        resultat_final = lemmatize_tokens(minuscules)

        print("=== RÉSULTAT DU PRÉTRAITEMENT ===")
        for i, phrase in enumerate(resultat_final, 1):
            print(f"Phrase {i} : {phrase}")