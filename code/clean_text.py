# import packages
import re
from string import punctuation
#from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# create function
def clean_text(text):
    """
    text: a string
    
    returns: modified initial string
    
    reference: https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568
    """

    # create objects for cleaning
    # replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    # remove_html_tags_re = re.compile('<.*?>')
    remove_newlines_re = re.compile('[\n]+')
    # remove_null_char_re = re.compile('\x00')
    # remove_misc_re = re.compile('\.\.+|==+|\B-+|\B_+|XXX+|MMM+')

    # clean the text
    text = text.lower()  # lowercase text
    # text = remove_html_tags_re.sub('', text)
    text = remove_newlines_re.sub(' ', text)
    # text = remove_null_char_re.sub(' ', text)
    # text = remove_misc_re.sub(' ', text)
    # text = replace_by_space_re.sub(' ', text)  # replace symbols by space in text
    text = bad_symbols_re.sub('', text)  # delete symbols from text
    text = re.sub(' +', ' ', text).strip()  # Function to remove multiple spaces
    text_tokens = word_tokenize(text)
    text_tokens = [w for w in text_tokens if w not in punctuation]
    # text_tokens = [w for w in text_tokens if w not in set(stopwords.words('english'))]  # delete stopwords from text
    clean = ' '.join(text_tokens)
    return clean


# create function
def clean_tokenize_text(text):
    """
    text: a string

    returns: modified initial string

    reference: https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568
    """

    # create objects for cleaning
    # replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    # remove_html_tags_re = re.compile('<.*?>')
    remove_newlines_re = re.compile('[\n]+')
    # remove_null_char_re = re.compile('\x00')
    # remove_misc_re = re.compile('\.\.+|==+|\B-+|\B_+|XXX+|MMM+')

    # clean the text
    text = text.lower()  # lowercase text
    # text = remove_html_tags_re.sub('', text)
    text = remove_newlines_re.sub(' ', text)
    # text = remove_null_char_re.sub(' ', text)
    # text = remove_misc_re.sub(' ', text)
    # text = replace_by_space_re.sub(' ', text)  # replace symbols by space in text
    text = bad_symbols_re.sub('', text)  # delete symbols from text
    text = re.sub(' +', ' ', text).strip()  # Function to remove multiple spaces
    text_tokens = word_tokenize(text)
    text_tokens = [w for w in text_tokens if w not in punctuation]
    # text_tokens = [w for w in text_tokens if w not in set(stopwords.words('english'))]  # delete stopwords from text
    return text_tokens
