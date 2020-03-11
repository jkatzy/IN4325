import os
import gensim
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords
import smart_open
import pickle
import numpy as np


class Vectorizer:
    def __init__(self, file_name, train=True):
        self.data_file = file_name
        self.train = train
        self.model = None
        self.labels = []

        if os.path.isfile(self.data_file + ".model"):
            self.model = gensim.models.doc2vec.Doc2Vec.load(self.data_file + ".model")
            self.labels = pickle.load(open(self.data_file + "_labels.model", "rb"))

    def read_corpus(self):
        with smart_open.open(self.data_file) as f:
            f.readline()
            for i, line in enumerate(f):
                split = line.split('\t')
                tokens = gensim.utils.simple_preprocess(remove_stopwords(split[2]))
                tokens = gensim.utils.lemmatize(split[2])
                if not self.train:
                    yield tokens
                else:
                    self.labels.append(int(split[-1]))
                    yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

    def train_model(self):
        train_corpus = self.read_corpus()
        model = gensim.models.doc2vec.Doc2Vec(vector_size=100, min_count=1, epochs=50)
        model.build_vocab(train_corpus)
        model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
        return model

    def get_model(self):
        if not self.model:
            self.model = self.train_model()
            self.model.save(self.data_file + ".model")
            pickle.dump(self.labels, open(self.data_file + "_labels.model", "wb"))
        return self.model.docvecs.doctag_syn0, np.array(self.labels)
