"""
this is the script to build the doc2vec vector models using gensim. Here we use the standard reuters news data set for building the DOC2vec vectors. 
"""
from random import shuffle
import nltk
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import reuters
from nltk.tokenize import word_tokenize
import logging
logging.basicConfig(level=INFO)
logger=logging.getLogger(__name__)
#downloading the data set packaged in nltk 
logger.info('downloading reuters data from nltk')
nltk.download('reuters')
nltk.download('punkt')
logger.info('finished downloading data')

doc2vec_vectors_location = 'vectors/doc2vec-vectors.bin'	#set location to save the vector
doc2vec_dimensions = 300								#it can be 100, 200, 300 etc 

logger.info('loading reuters news articles and converting to TaggedDocuments')

taggedDocuments = [TaggedDocument(words=word_tokenize(reuters.raw(fileId)), tags=[i]) for i, fileId in enumerate(reuters.fileids())]
shuffle(taggedDocuments)
logger.info('finished creating tagged documents')
#creating doc2vec instance 
doc2vec = Doc2Vec(vector_size=doc2vec_dimensions, min_count=2, iter=10, workers=12)

logger.info('Building the word2vec model from the corpus'')
doc2vec.build_vocab(taggedDocuments)

doc2vec.train(taggedDocuments)
logger.info('finished building doc2vec vector')
doc2vec.save(doc2vec_vectors_location)
logger.info('vector saved to %s',doc2vec_vectors_location)