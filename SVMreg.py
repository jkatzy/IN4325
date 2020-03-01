import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

from sklearn.svm import SVR

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import NearestNeighbors

# Read the files into dataframes
df_train = pd.read_csv("train.tsv", sep="\t")
df_test = pd.read_csv("test.tsv", sep="\t")
df_traintest = pd.read_csv("traintest.tsv", sep="\t")


# Take only the whole sentence if we want to give sentiment per sentence/review
df_drop_train = df_train.drop_duplicates(['SentenceId']).groupby('SentenceId').head(1).reset_index()
df_drop_test = df_test.drop_duplicates(['SentenceId']).groupby('SentenceId').head(1).reset_index()

# SVM Regression
svr = SVR(C=1.0,
          epsilon=0.2,
          tol=1e-05)
# SVM OVA
svc = LinearSVC(C=1.0,
                class_weight='balanced',
                dual=True,
                fit_intercept=True,
                intercept_scaling=1,
                loss='squared_hinge',
                max_iter=10000,
                multi_class='ovr',
                penalty='l2',
                random_state=0,
                tol=1e-05,
                verbose=0
)
# Vectorizer to create tfidf feature vector: http://blog.christianperone.com/2011/09/machine-learning-text-feature-extraction-tf-idf-part-i/
vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)

# Select which set to use
SetChoice= "Combined"

# Combine the train and test set into one file such that tfidf vectorizer will give same feature vector length, required for when predicting
if SetChoice == "Combined":
    print("Using Combined")
    X_traintest= df_traintest.Phrase
    vectX_traintest = vectorizer.fit_transform(X_traintest)
    vectX = vectX_traintest[:156060]
    vectX_test = vectX_traintest[156061:]
    vectX = vectX[:20000] #CUT FOR REG, can remove
    vectX_test = vectX_test[:20000] #CUT FOR REG, can remove
    y = df_traintest[:156060].Sentiment
    y = y[:20000] #CUT FOR REG, can remove

# For when classifying all phrases
if SetChoice == "Phrases":
    print("Using Phrases")
    #cut to reduce run time when using svr
    df_cut_train = df_train[:20000]
    df_cut_test = df_test[:20000]
    X = df_cut_train.Phrase
    y = df_cut_train.Sentiment
    vectX = vectorizer.fit_transform(X)

#For When classifying only sentences/reviews
if SetChoice == "Sentences":
    print("Using Sentences")
    X = df_drop_train.Phrase
    y = df_drop_train.Sentiment
    vectX = vectorizer.fit_transform(X)

#K fold cross evaluation for SVM OVA
skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(vectX, y):
    svc.fit(vectX[train], y[train])
    train_score = svc.score(vectX[train], y[train])
    test_score = svc.score(vectX[test], y[test])
    print("SVM OVA: Train Score = {}, Test Score= {}".format(train_score, test_score))

#K fold cross evaluation for SVM Regression
skf_svr = StratifiedKFold(n_splits=3)
for train, test in skf_svr.split(vectX, y):
    svr.fit(vectX[train], y[train])
    train_score = svr.score(vectX[train], y[train])
    test_score = svr.score(vectX[test], y[test])
    print("SVM REG: Train Score = {}, Test Score= {}".format(train_score, test_score))

if SetChoice == "Combined":
    #predict test set with svc
    total_list = np.array(svc.predict(vectX_test))
    #np.set_printoptions(threshold=np.inf) # For printing entire array
    print(total_list)
    print(np.var(total_list))
    print(np.mean(total_list))

    #print(vectorizer.get_feature_names())

    #Naive Bayes classifier for PSP
    gnb = GaussianNB()
    gnbfit = gnb.fit(vectX.toarray(), y)
    prediction = gnbfit.predict(vectX_test.toarray())
    print(prediction)
    print(len(prediction))
    
 #TODO: Change vectX to vectX[test]
 #TODO : Figure out what alogrithm and leaf size to use
 #TODO : Fix dictionary appending of data
 #TODO : Distance function between labels
 #TODO : Summation of distance and similarity function

def knearest():
     nbrs = NearestNeighbors(algorithm='auto', leaf_size=30, n_neighbors=2, p=2,
     radius=1.0).fit(vectX)
     distances, indices = nbrs.kneighbors(vectX)
     print(distances)
     print(indices)
     return indices
     #nbrs.kneighbors_graph(vectX).toarray()

def getlabel(indices):
    ldict = dict()
    raw_labels = [(vectX[x], y[x]) for x in indices]
    
    for x in raw_labels:
        key = (0,1,2,3)
        ldict.setdefault(key,[]).append(raw_labels)
        
    print(ldict.items())   
    return ldict

def distance(labels):
    #dist = labels.get(1) - labels.get(3)
    #print(dist)


indices = knearest()
label_dictionary = getlabel(indices)
#distance(label_dictionary)