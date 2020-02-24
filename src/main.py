import os
import gensim
#from gensim.models import Phrases
import smart_open

data_dir = '/home/nommoinn/IR/nlp/data/'
lee_train_file = os.path.join(data_dir, 'train.tsv')
lee_test_file = os.path.join(data_dir, 'test.tsv')
ratings = []


def read_corpus(fname, train=True):
    with smart_open.open(fname) as f:
        for i, line in enumerate(f):
            split = line.split('\t')
            tokens = gensim.utils.simple_preprocess(split[2])
            if not train:
                yield tokens
            else:
                ratings.append(int(split[-1]))
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


train_corpus = read_corpus(lee_train_file)
#test_corpus = list(read_corpus(lee_test_file, train=False))

model = gensim.models.doc2vec.Doc2Vec(vector_size=5, min_count=1, epochs=20)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

#vector = model.infer_vector(['only'])
vector = model.docvecs[0]
print(vector)


# bigram_transformer = Phrases(common_texts)
# model = Word2Vec(bigram_transformer[common_texts], min_count=1)

# path = get_tmpfile("word2vec.model")
# model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
# model.train([["hello", "world"]], total_examples=1, epochs=1)
#
# sentences = MySentences('/some/directory')  # a memory-friendly iterator
# model = gensim.models.Word2Vec(sentences)
