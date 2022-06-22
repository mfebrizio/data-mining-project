# import dependencies
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from wordcloud import WordCloud, ImageColorGenerator


# A function to generate the word cloud from text
def generate_basic_wordcloud(data, title):
    """
    data: a string of words
    title: title of the wordcloud

    returns: wordcloud figure

    reference: https://towardsdatascience.com/how-to-create-beautiful-word-clouds-in-python-cfcf85141214
    """

    cloud = WordCloud(width=400,
                      height=330,
                      max_words=150,
                      colormap='tab20c',
                      stopwords=stopwords,
                      collocations=True).generate_from_text(data)
    plt.figure(figsize=(10, 8))
    plt.imshow(cloud)
    plt.axis('off')
    plt.title(title, fontsize=13)
    plt.show()
