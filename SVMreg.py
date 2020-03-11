import numpy as np
import pandas as pd
from gensim.utils import lemmatize
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt
import collections
import seaborn as sn
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import os
import warnings

warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses


def conf_mat(y_true, y_pred, model_name):
    labels = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=labels)
    disp.plot(include_values=True,
              cmap=plt.cm.Reds, ax=None, xticks_rotation='horizontal',
              values_format=None)
    plt.show()
    #plt.savefig(model_name + "_conf_mat")


def tfidf(train_data, test_data):
    vectorizer = TfidfVectorizer(
        tokenizer=word_tokenize,
        preprocessor=lemmatizer.lemmatize)

    print("Training TFIDF")
    data_X = vectorizer.fit_transform(train_data.Phrase)
    test_words = vectorizer.transform(test_data.Phrase)

    return data_X, test_words


def doc2vec(train_data, test_data):
    SentimentDocument = collections.namedtuple('SentimentDocument', 'words tags sentiment')
    train = train_data.reset_index()
    test = test_data.reset_index()

    train_docs = [
        SentimentDocument(
            [lemmatizer.lemmatize(w) for w in row['Phrase'].lower().split() if w not in sw],
            [idx],
            row['Sentiment']
        ) for idx, row in train.iterrows()
    ]

    test_docs = [
        SentimentDocument(
            [lemmatizer.lemmatize(w) for w in row['Phrase'].lower().split() if w not in sw],
            [idx],
            row['Sentiment']
        ) for idx, row in test.iterrows()
    ]

    common_kwargs = dict(
        vector_size=200, epochs=50, min_count=2, workers=6
    )

    print("Training Doc2Vec")
    model = Doc2Vec(dm=0, **common_kwargs)
    model.build_vocab(train_docs)
    model.train(train_docs, total_examples=len(train_docs), epochs=model.epochs)

    # data_X = [model.docvecs[doc.tags[0]] for doc in train_docs]
    data_X = model.docvecs.doctag_syn0
    test_words = [model.infer_vector(doc.words) for doc in test_docs]

    return data_X, test_words


data = pd.read_csv('./train.tsv', sep='\t', header=0)

## Removes some class [optional]
data = data.query('Sentiment != 2')
data = data.reset_index()

## Selects only whole phrases [optional]
# data = data.drop_duplicates(['SentenceId']).groupby('SentenceId').head(1).reset_index()


# sw = list(stopwords.words('english'))
# sw.remove('no')
# sw.remove('not')
# sw.extend(['cinema', 'film', 'series', 'movie', 'one', 'like', 'story', 'plot', ''])
sw = ['cinema', 'film', 'series', 'movie', 'story', 'plot', '', 'the', 'of', 'an', 'a', 'she', 'he', 'to', 'our', 'it',
      'my', 'I']
lemmatizer = WordNetLemmatizer()

n_estimators = 5
max_iter = 10
# SVM OVA
s_svc = LinearSVC(
    C=3.0,
    class_weight='balanced',
    dual=True,
    fit_intercept=True,
    intercept_scaling=1,
    loss='squared_hinge',
    max_iter=max_iter,
    multi_class='ovr',
    penalty='l2',
    random_state=0,
    tol=1e-05,
    verbose=0
)

# SVM RBF OVA
rbf_svc = BaggingClassifier(
    SVC(
        C=6.0,
        kernel='rbf',
        max_iter=max_iter,
        probability=True,
        class_weight='balanced'
    ), max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs=-1)

# SVM LIN OVA
lin_svc = BaggingClassifier(
    LinearSVC(
        C=6.0,
        class_weight='balanced',
        loss='squared_hinge',
        max_iter=max_iter,
        penalty='l2',
        verbose=0
    ), max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs=-1)

# SVR
lin_svr = LinearSVR(
    C=0.75,
    epsilon=0.3,
    max_iter=max_iter
)

splitx = list(data.index)
splity = list(data.Sentiment.values)

# Split by sentence Id [optional]
# splitx = list(data.drop_duplicates(['SentenceId']).index)
# splity = list(data.drop_duplicates(['SentenceId']).groupby('SentenceId').head(1).Sentiment.values)

skf = StratifiedKFold(n_splits=5, shuffle=True)
models = {'Single SVC': s_svc, 'Lin SVC': lin_svc, 'RBF SVC': rbf_svc, 'SVR': lin_svr}
embeddings = {'tfidf': tfidf, 'doc2vec': doc2vec}

for n_e, embed in embeddings.items():
    for train_idx, test_idx in skf.split(splitx, splity):
        train = data[data['PhraseId'].isin(train_idx)]
        test = data[data['PhraseId'].isin(test_idx)]

        train_x, test_x = embed(train, test)

        for name, model in models.items():
            model.fit(X=train_x, y=train.Sentiment)

            train_pred = model.predict(train_x)
            test_pred = model.predict(test_x)

            if name == "SVR":
                train_acc = accuracy_score(np.round(train_pred), train.Sentiment)
                test_acc = accuracy_score(np.round(test_pred), test.Sentiment)
            else:
                train_acc = accuracy_score(train_pred, train.Sentiment)
                test_acc = accuracy_score(test_pred, test.Sentiment)
                conf_mat(test.Sentiment, test_pred, '_'.join([n_e, name]))
                #print(classification_report(test.Sentiment, test_pred))

            train_rmse = mean_squared_error(train.Sentiment, train_pred, squared=False)
            test_rmse = mean_squared_error(test.Sentiment, test_pred, squared=False)

            print("{}: Train Acc = {}, Test Acc = {}, Train rmse = {}, Test rmse = {}".format(name, train_acc, test_acc,
                                                                                              train_rmse, test_rmse))
