# reuters-doc2vec
An experiment to use doc2vec as embedding for text classification with reuters news data
How to use :
1. We need to create the doc2vec vector representation of the documents in the corpus. So run ‘reuters-doc2vec-train.py’ first to build the vector. This will generate the vector file in the ‘vectors’ folder. 
2. Next is the classifier training. We use a fully connected neural network for classifying the news categories. Run ‘reuters-classifier-train.py’ to train a model for topic classification. 
3. After training the model we can use that trained model to predict the category of a unknown document. Run ‘reuters-classifier-predict.py’ for making prediction. 
A major difference while using the doc2vec representation is the absence of an embedding layer in our neural network. In the case of doc2vec  there is no pre-trained embedding vectors available. We need to train and build our own vector representation for the training of neural networks. Here we infer the vector representation of each document and that representation is given as the input to the neural network. For each prediction also the same inference should be done. 

