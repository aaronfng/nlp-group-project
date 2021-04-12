import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

PORTER_STEMMER = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))


def preprocess_text(text):
    """ Preprocess one line of text. """
    # NLTK also separates out punctuation, but keeps them in the list.
    # e.g. ',' is a token.
    tokens = nltk.word_tokenize(text, language="english")

    tokens_new = []
    for token in tokens:
        if token in STOPWORDS:
            continue

        if not token.isalnum():
            # Only keep words containing alphanumeric characters
            # Tokens containing only punctuation, symbols etc. are removed
            continue

        # TODO: do further preprocessing if necessary,
        #       (if vocab size is too large) e.g. handle Unicode, lowercase

        # Stem word
        # TODO: map stemmed words to their original to recover them???
        token_new = PORTER_STEMMER.stem(token)
        tokens_new.append(token_new)

    return tokens_new
