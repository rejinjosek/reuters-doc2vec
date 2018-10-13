'''
here we build a fully connected neural network with 4 dense layers. Here we infer the word vectors from the doc2vec model using 'reuters-doc2vec-train.py' 
'''
from random import shuffle
import nltk
import numpy
from gensim.models import Doc2Vec
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from nltk.corpus import reuters
from nltk.tokenize import word_tokenize
from sklearn.preprocessing.label import MultiLabelBinarizer
import logging 
logging.basicConfig(level=INFO)
logger=logging.getLogger(__name__)
#uncomment these two lines   if the corpus is not available
#nltk.download('reuters')
#nltk.download('punkt')
#logger.info('finished downloading corpus')

doc2vec_model_location = 'vectors/doc2vec-model.bin'
doc2vec_dimensions = 300
classifier_model_location = 'models/classifier-model.bin'

doc2vec = Doc2Vec.load(doc2vec_model_location)
logger.info('doc2vec vector loaded')
labelBinarizer = MultiLabelBinarizer()
labelBinarizer.fit([reuters.categories(fileId) for fileId in reuters.fileids()])
logger.info('finished Converting the categories to one hot encoded categories')
# Convert load the articles with their corresponding categories
train_articles = [{'raw': reuters.raw(fileId), 'categories': reuters.categories(fileId)} for fileId in reuters.fileids() if fileId.startswith('training/')]
test_articles = [{'raw': reuters.raw(fileId), 'categories': reuters.categories(fileId)} for fileId in reuters.fileids() if fileId.startswith('test/')]
shuffle(train_articles)
shuffle(test_articles)
logger.info('reuters data loaded and shuffled')
# Convert the articles to document vectors using the doc2vec model
train_data = [doc2vec.infer_vector(word_tokenize(article['raw'])) for article in train_articles]
test_data = [doc2vec.infer_vector(word_tokenize(article['raw'])) for article in test_articles]
train_labels = labelBinarizer.transform([article['categories'] for article in train_articles])
test_labels = labelBinarizer.transform([article['categories'] for article in test_articles])
train_data, test_data, train_labels, test_labels = numpy.asarray(train_data), numpy.asarray(test_data), numpy.asarray(train_labels), numpy.asarray(test_labels)
logger.info('completed vector inference from doc2vec')
# Initialize the neural network
model = Sequential()
model.add(Dense(input_dim=doc2vec_dimensions, output_dim=500, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(output_dim=1200, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(output_dim=400, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(output_dim=600, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(output_dim=train_labels.shape[1], activation='softmax'))
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Saves the model with highest score after each training cycle
checkpointer = ModelCheckpoint(filepath=classifier_model_location, verbose=1, save_best_only=True)
logger.info('neural network model compiled')
# Train the neural network
logger.info('starting training')
model.fit(train_data, train_labels, validation_data=(test_data, test_labels), batch_size=32, nb_epoch=15, callbacks=[checkpointer])
logger.info('model training finished')