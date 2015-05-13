import logging
import os

import numpy
from lxml import etree
from gensim import corpora, models, similarities
from nltk import tokenize
from toolz import nth

logging.basicConfig(level='DEBUG')
logger = logging.getLogger(__name__)


class ArxivAbstracts(corpora.textcorpus.TextCorpus):
    """Generator over abstracts.

    Abstracts are lowercased and split on whitespace.

    Parameters
    ----------
    path : str
        The directory in which the XML files are stored.

    Returns
    -------
    description : list
        A list of strings (tokens).

    Notes
    -----
    Reads *all* the files in the directory.

    .. todo::

       A more intelligent tokenization scheme could be used and stop-words
       filtered.

    """
    def __init__(self, path):
        self.path = path
        super(ArxivAbstracts, self).__init__(input=True)

    def get_texts(self):
        for article_file in os.listdir(self.path):
            with open(os.path.join(self.path, article_file)) as article:
                tree = etree.parse(article)
                # ElementTree needs to namespace passed explicitly
                description_element = tree.find(
                    './/dc:description',
                    {'dc': 'http://purl.org/dc/elements/1.1/'}
                )
                yield tokenize.word_tokenize(description_element.text.lower())

    def __len__(self):
        return len(os.listdir(self.path))

# Perform LSI/LSA  on TF-IDF representation of abstracts
corpus = ArxivAbstracts('../harvest/data/')
corpus.dictionary.filter_extremes()
tfidf = models.tfidfmodel.TfidfModel(corpus, dictionary=corpus.dictionary)
lsi = models.lsimodel.LsiModel(tfidf[corpus], id2word=corpus.dictionary)

# Example of similarity query
index = similarities.MatrixSimilarity(lsi[tfidf[corpus]])
first_abstract = next(iter(corpus))
vec_lsi = lsi[first_abstract]
most_similar = numpy.argpartition(index[vec_lsi], -2)[-2]
print(' '.join(nth(most_similar, corpus.get_texts())))
