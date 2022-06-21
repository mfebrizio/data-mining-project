# import dependencies
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


# define function
def tfidf_vectorize(df, var):
    """
    :param df: pandas dataframe object
    :param var: column name in pandas dataframe
    :return: tf-idf vectorized object
    """
    corpus = df.loc[:, var].tolist()

    vectorizer = TfidfVectorizer(analyzer='word', stop_words=stopwords.words('english'))
    processed_var = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    return processed_var, feature_names
