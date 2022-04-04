from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize 
import nltk
import re
import string
from nltk.corpus import stopwords
import nltk
import gensim
import pandas as pd
from sentence_transformers import SentenceTransformer

nltk.download('stopwords')
nltk.download('punkt')

#A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}


def clean_text(text, remove_stopwords = True):
    '''Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings
       instructions:
          news_data["Title"] = news_data["Title"].apply(lambda x: clean_text(x, remove_stopwords= True) )
    '''
    if type(text) == str:
        # Convert words to lower case
        text = text.lower()
        
        # Replace contractions with their longer forms 
        if True:
            text = text.split()
            new_text = []
            for word in text:
                if word in contractions:
                    new_text.append(contractions[word])
                else:
                    new_text.append(word)
            text = " ".join(new_text)
        
        #Format words and remove unwanted characters
        text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\<a href', ' ', text)
        text = re.sub(r'&amp;', '', text) 
        text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', '', text)
        text = re.sub(r'<br />', ' ', text)
        text = re.sub(r'\'', ' ', text)
        
        # Optionally, remove stop words
        if remove_stopwords:
            text = text.split()
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops]
            text = " ".join(text)

        return text

def count_words(count_dict, text):
    '''Count the number of occurrences of each word in a set of text'''
    for sentence in text:
      if type(sentence) == str:
        for word in sentence.split():
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1



def process_text(text):
    """Process text function.
    Input:
        text: a string containing a text
    Output:
        texts_clean: a list of words containing the stemming and tokenize text
    instructions:
        news['Title'] = news.Title.apply(process_text)
    """
    stemmer = PorterStemmer()

    # tokenize text
    text_tokens = word_tokenize(text)

    texts_clean = []
    for word in text_tokens:
          stem_word = stemmer.stem(word)  # stemming word
          texts_clean.append(stem_word)

    return texts_clean



""" Please uncomment this cell when you want use words_vectors function

!brew install wget
!wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
!gzip -d GoogleNews-vectors-negative300.bin.gz

"""
def words_vectores(news_dataset):
    """Preprocess for tfidf vectorize
    Input: 
        text: 
    Output:
        texts_clean: 

    instructions:
        
    """
    # Grab all the titles 
    article_titles = news_dataset['Title']
    # Create a list of strings, one for each title
    titles_list = [title for title in article_titles]

    # Collapse the list of strings into a single long string for processing
    big_title_string = ' '.join(titles_list)

    # Tokenize the string into words
    tokens = word_tokenize(big_title_string)

    # Remove non-alphabetic tokens, such as punctuation
    words = [word.lower() for word in tokens if word.isalpha()]

    # Filter out stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if not word in stop_words]

    # Load word2vec model (trained on an enormous Google corpus)
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True) 

    # Check dimension of word vectors
    model.vector_size

    # Filter the list of vectors to include only those that Word2Vec has a vector for
    vector_list = [model[word] for word in words if word in model.vocab]

    # Create a list of the words corresponding to these vectors
    words_filtered = [word for word in words if word in model.vocab]

    # Zip the words together with their vector representations
    word_vec_zip = zip(words_filtered, vector_list)

    # Cast to a dict so we can turn it into a DataFrame
    word_vec_dict = dict(word_vec_zip)
    df = pd.DataFrame.from_dict(word_vec_dict, orient='index')
    return df



def preprocess_tfidf(text):
    """Preprocess for tfidf vectorize
    Input:
        text: a string containing a text
    Output:
        texts_clean: a str of words containing the processed text

    instructions:
        news['Title'] = news.Title.apply(preprocess_tfidf)
    """

    # Replace contractions with their longer forms 
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)

    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    text = re.sub(r'\$\w*', '', text)
    # remove old style reabstract text "RT"
    text = re.sub(r'^RT[\s]+', '', text)
    # tokenize text
    text_tokens = word_tokenize(text)

    texts_clean = []
    for word in text_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # texts_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            texts_clean.append(stem_word)
  
    return " ".join(texts_clean)



"""

!pip install sentence-transformers


"""
def bert_embedding(text):
    """"worod embedding by bert
    Input:
        text: a string containing a text
    Output:
        texts_clean: sentence embedd

    instructions:
        news['Title'] = news.Title.apply(bert_embedding)
    """
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    str_emb = model.encode(text)
    return str_emb
