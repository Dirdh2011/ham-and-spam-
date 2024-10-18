import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

class NLPProcessor:
    def __init__(self):
        # Download necessary resources once during initialization
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab',quiet=True)
        nltk.download('averaged_perceptron_tagger_eng',quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('universal_tagset', quiet=True)
        
        # Initialize lemmatizer and stopwords set
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def get_wordnet_pos_universal(self, tag):
        """Map universal POS tags to WordNet POS for lemmatization"""
        if tag == 'ADJ':
            return wordnet.ADJ
        elif tag == 'VERB':
            return wordnet.VERB
        elif tag == 'NOUN':
            return wordnet.NOUN
        elif tag == 'ADV':
            return wordnet.ADV
        else:
            return None  # Return None for other POS tags
    @staticmethod
    def clean_text(text):
        """
        Use regular expressions to remove unwanted patterns like URLs, numbers, special characters, etc.
        """
        # Example regex to remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    def preprocess_text(self, paragraph):
        """Tokenize, remove stopwords, and apply lemmatization"""
        # Clean text using regex
        clean_paragraph = self.clean_text(paragraph)
        sentences = sent_tokenize(clean_paragraph)
        processed_sentences = []

        for sentence in sentences:
            words = word_tokenize(sentence)
            filtered_words = [word for word in words if word.lower() not in self.stop_words]
            pos_tagged = pos_tag(filtered_words, tagset='universal')

            lemmatized_words = []
            for word, tag in pos_tagged:
                wordnet_pos = self.get_wordnet_pos_universal(tag) or wordnet.NOUN
                lemmatized_word = self.lemmatizer.lemmatize(word, wordnet_pos)
                lemmatized_words.append(lemmatized_word)

            processed_sentences.append(' '.join(lemmatized_words))

        return ' '.join(processed_sentences)
    

