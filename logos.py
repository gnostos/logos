import logging
import os
from lxml import etree
from gensim import corpora, models
from nltk import tokenize

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

corpus = ArxivAbstracts('../harvest/data/')
corpus.dictionary.filter_extremes()
lsi = models.lsimodel.LsiModel(corpus, id2word=corpus.dictionary)
