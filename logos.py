from lxml import etree
from gensim import corpora
import os


def arxiv_abstracts(path):
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
    for article_file in os.listdir(path):
        with open(article_file) as article:
            tree = etree.parse(article)
            # ElementTree needs to namespace passed explicitly
            description_element = tree.find(
                './/dc:description',
                {'dc': 'http://purl.org/dc/elements/1.1/'}
            )
            yield description_element.text.lower().split()


dictionary = corpora.Dictionary(arxiv_abstracts('../harvest/data/'))
