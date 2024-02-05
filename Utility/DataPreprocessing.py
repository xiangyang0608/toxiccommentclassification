import re
from collections import defaultdict
import unicodedata
import emoji
from unidecode import unidecode
from nltk.stem import SnowballStemmer

def get_clean_word_dict(cl_path):
    clean_word_dict = {}
    with open (cl_path, 'r', encoding = 'utf-8') as cl:
        for line in cl:
            line = line.strip('\n')
            typo, correct = line.split(',')
            clean_word_dict[typo] = correct
    return clean_word_dict

# Regex to remove all Non-Alpha Numeric and space
special_character_removal=re.compile(r'[^?!.,:a-z\d ]',re.IGNORECASE)

# regex to replace all numerics
replace_numbers=re.compile(r'\d+',re.IGNORECASE)
word_count_dict = defaultdict(int)
toxic_dict = {}

def clean_text(text, clean_word_dict, remove_stopwords=False, stem_words=False, count_null_words=True, clean_wiki_tokens=True):
    # Clean the text, with the option to remove stopwords and to stem words.
    # dirty words
    text = text.lower()
    text = re.sub(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)", "", text)
    text = re.sub(r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}", "", text)
    text = re.sub(r"[“”—’…’‘˚]«»▄·ˈ", "", text)
    
    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)
    # remove all the emojis
    text = emoji.demojize(text)
    text = re.sub(r':[a-z_]+:', ' ', text)
    # remove all the tones
    text = unidecode(text)
    
    if clean_wiki_tokens:
        # Drop the image
        text = re.sub(r"image:[a-zA-Z0-9]*\.jpg", " ", text)
        text = re.sub(r"image:[a-zA-Z0-9]*\.png", " ", text)
        text = re.sub(r"image:[a-zA-Z0-9]*\.gif", " ", text)
        text = re.sub(r"image:[a-zA-Z0-9]*\.bmp", " ", text)

        # Drop css
        text = re.sub(r"#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})", " ",text)
        text = re.sub(r"\{\|[^\}]*\|\}", " ", text)

        # Clean templates
        text = re.sub(r"\[?\[user:.*\]", " ", text)
        text = re.sub(r"\[?\[user:.*\|", " ", text)
        text = re.sub(r"\[?\[wikipedia:.*\]", " ", text)
        text = re.sub(r"\[?\[wikipedia:.*\|", " ", text)
        text = re.sub(r"\[?\[special:.*\]", " ", text)
        text = re.sub(r"\[?\[special:.*\|", " ", text)
        text = re.sub(r"\[?\[category:.*\]", " ", text)
        text = re.sub(r"\[?\[category:.*\|", " ", text)

    for typo, correct in clean_word_dict.items():
        text = re.sub(typo, " " + correct + " ", text)

    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"what’s", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\’s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"\’ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"can’t", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"n’t", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"i’m", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\’re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\’d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\’ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\!", " ! ", text)
    text = re.sub(r"\"", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"mslgbt", "lgbt", text)
    text = replace_numbers.sub(' ', text)
    
    if count_null_words:
        text = text.split()
        for t in text:
            word_count_dict[t] += 1
        text = " ".join(text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
        
    # replace with pattern
    # replace text starting with lg or contains lgbt with lgbt
    pattern = r'\blg\w*|\blgbt\b'
    text = re.sub(pattern, 'lgbt', text)
    return (text)

def get_clean_data(data, clean_word_dict):
    list_sentences = data["string"].fillna("no comment").values
    comments = [clean_text(text, clean_word_dict) for text in list_sentences]

    print("Cleaned")
    return comments