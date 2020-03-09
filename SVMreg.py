import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix as cm

from sklearn.svm import SVR

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors

import nltk 
#nltk.download('all')
from nltk.corpus import wordnet as wn

from nltk.corpus import stopwords, wordnet 
from nltk import word_tokenize, WordNetLemmatizer, sent_tokenize
from scipy import spatial

from nltk import pos_tag

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import plot_confusion_matrix


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
SetChoice= "Sentences"

# Combine the train and test set into one file such that tfidf vectorizer will give same feature vector length, required for when predicting
if SetChoice == "Combined":
    print("Using Combined")
    X_traintest= df_traintest.Phrase
    vectX_traintest = vectorizer.fit_transform(X_traintest)
    vectX = vectX_traintest[:15606]
    vectX_test = vectX_traintest[15606:]
    vectX = vectX[:2000] #CUT FOR REG, can remove
    vectX_test = vectX_test[:2000] #CUT FOR REG, can remove
    y = df_traintest[:15606].Sentiment
    y = y[:2000] #CUT FOR REG, can remove
    vectX = vectX[:2000] #CUT FOR REG, can remove
    vectX_test = vectX_test[:2000] #CUT FOR REG, can remove
    y = df_traintest[:15606].Sentiment
    y = y[:2000] #CUT FOR REG, can remove

# For when classifying all phrases
if SetChoice == "Phrases":
    print("Using Phrases")
    #cut to reduce run time when using svr
    df_cut_train = df_train[:250]
    df_cut_test = df_test[:250]
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

if SetChoice == "Sentences":
    #predict test set with svc
    total_list = np.array(svc.predict(vectX_test))
    #np.set_printoptions(threshold=np.inf) # For printing entire array
    print(total_list)
    y_true = np.array(y)
    
    print(np.var(total_list))
    print(np.mean(total_list))    
    #print(vectorizer.get_feature_names())

    #Naive Bayes classifier for PSP
    gnb = GaussianNB()
    gnbfit = gnb.fit(vectX.toarray(), y)
    prediction = gnbfit.predict(vectX_test.toarray())
    print(prediction)
    print(len(prediction))

def confusion_matrix(classifier, X, y_true, y_pred):
    
    my_cm = cm(y_true, y_pred, labels=[0.0,1.0,2.0,3.0,4.0])
    print(my_cm)
    
    labels=[0.0,1.0,2.0,3.0,4.0]
    np.set_printoptions(precision=2)    
    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X ,y_true,
                                 display_labels=labels,
                                 cmap=plt.cm.Reds,
                                 normalize=normalize)   
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)   
    plt.show()

#Generate confusion matrices     

cm_1 = confusion_matrix(svc, vectX, y_true, total_list)
print(cm_1)
    
 #TODO: Change vectX to vectX[test]
 #TODO : Figure out what alogrithm and leaf size to use
 #TODO : Fix dictionary appending of data
 #TODO : Distance function between labels
 #TODO : Summation of distance and similarity function

def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'
 
    if tag.startswith('V'):
        return 'v'
 
    if tag.startswith('J'):
        return 'a'
 
    if tag.startswith('R'):
        return 'r'
    return None

def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None


def sentence_similarity(sentence1, sentence2):

    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))
 
    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
 
    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]
  
    score, count = 0.0, 0
    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        # Check that the similarity could have been computed
            try:
                best_score = max([synset.path_similarity(ss) for ss in synsets2]) 
                score += best_score
                count += 1
                score /= count
                return score
            except:
                return None        


def knearest():
    
 nbrs = NearestNeighbors(algorithm='auto', leaf_size=30, n_neighbors=2, p=2,
 radius=1.0).fit(vectX)
 distances, indices = nbrs.kneighbors(vectX)
 
 print(distances)
 
 small_indices = indices[:20]
 return small_indices

def get_similarity(indices):
 w,h = 1, 20
 #h is the number of lists, w is the number of items
 sim = [[0 for x in range(w)] for y in range(h)] 

 #print (indices)
 #print(type(sim)) 
 X = df_drop_train.Phrase

 for k in range(h):
         for i in indices:  
             #Cosine similarity -. most values equate to 1
             #cosim = 1 - spatial.distance.cosine(int(y[i[0]]), int(y[i[1]]))
             score = sentence_similarity(X.iloc[i[0]],X.iloc[i[1]])            
             if score == None: 
                 score = 0
             #Similarity Wordnet 
             #sim[k] = [i[0], i[1], score]
             sim[k] = score
 print('done with sim')
 return sim

def getlabel(indices):
    #Size of label_dist = 200
    h = 20
    ldict = [(y[x]) for x in indices]
    label_dist = [0 for z in range(h)]

    for i in range(h):
        score = np.abs(ldict[i].iloc[0]-ldict[i].iloc[1])
        label_dist[i] = int(score)
        #label_dist = np.abs(ldict[0]-ldict[1])
        #print(label_dist)
    
    print('done with label_dist')
    return label_dist

#Final equation from paper Seeing Stars:... by Pang Lee
'''
def total_equation(sim, indices, label_dist):
    print('reached total_equation')
    
    w = 15
    output = [0 for x in range(w)]
    
    for x in indices:
        output = (y[x[0]]) 
        main = label_dist[(x)]
        main2 = sim[x]
        print(output)  
        print(main)
        print(main2)
      
    return output
    
#Function Calls
indices = knearest()
label_dist = getlabel(indices)
sim = get_similarity(indices)

output = total_equation(sim, indices, label_dist)

#similarity(label_dist)
#sim = distance(labels)
'''