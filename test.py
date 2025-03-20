import re

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords

# Contraction words
contractions = {
    "isn't": "is not", "aren't": "are not", "can't": "cannot", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
    "hasn't": "has not", "haven't": "have not", "hadn't": "had not", "isn't": "is not",
    "it's": "it is", "let's": "let us", "ma'am": "madam", "mightn't": "might not",
    "mustn't": "must not", "needn't": "need not", "needn't": "need not", "shan't": "shall not",
    "she'd": "she would", "she'll": "she will", "she's": "she is", "shouldn't": "should not",
    "that's": "that is", "there's": "there is", "they'd": "they would", "they'll": "they will",
    "they're": "they are", "they've": "they have", "wasn't": "was not", "weren't": "were not",
    "what's": "what is", "what'll": "what will", "what're": "what are", "what've": "what have",
    "where's": "where is", "where've": "where have", "who's": "who is", "who'll": "who will",
    "who're": "who are", "who've": "who have", "why's": "why is", "why're": "why are",
    "why've": "why have", "won't": "will not", "wouldn't": "would not", "you'd": "you would",
    "you'll": "you will", "you're": "you are", "you've": "you have"
}

# Important words
sentiment_important_words = {
    "not", "no", "very", "good", "bad", "excellent", "love", "hate", "great", "feel", "wish", "would", "should"
}

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to NOUN if unknown
    
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Expand contractions (assumes contractions dictionary is available)
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)

    # Remove punctuation using regular expressions
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Define English stopwords
    stop_words = set(stopwords.words('english'))

    # Customize the stopwords list for sentiment analysis (add negations or important words)
    stop_words = stop_words - sentiment_important_words  # Remove sentiment important words from stopwords

    # POS tagging
    pos_tags = pos_tag(tokens)

    # Apply lemmatization based on POS tags and filter out stopwords and single-letter words
    lemmatized_tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags
        if word.isalnum() and word not in stop_words and len(word) > 1  # Filter single-letter words
    ]

    # Join the lemmatized tokens back into a single string
    return ' '.join(lemmatized_tokens)

print(preprocess_text("I hate it"))