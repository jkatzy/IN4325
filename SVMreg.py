import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix as cm
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
#nltk.download('all')
from nltk.corpus import stopwords, wordnet 
from nltk import word_tokenize, WordNetLemmatizer, sent_tokenize
from nltk.corpus import wordnet as wn
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report ,confusion_matrix as cm 
from nltk import pos_tag
import scipy as sc

# Read the files into dataframes
df_train = pd.read_csv("train.tsv", sep="\t")
df_test = pd.read_csv("test.tsv", sep="\t")
df_traintest = pd.read_csv("traintest.tsv", sep="\t")


# Take only the whole sentence if we want to give sentiment per sentence/review
df_drop_train = df_train.drop_duplicates(['SentenceId']).groupby('SentenceId').head(1).reset_index()#pd.read_csv("train_sentences.tsv", sep="\t") #
df_drop_test = df_test.drop_duplicates(['SentenceId']).groupby('SentenceId').head(1).reset_index()#pd.read_csv("test_sentences.tsv", sep="\t") #
df_drop_traintest= df_traintest.drop_duplicates(['SentenceId']).groupby('SentenceId').head(1).reset_index()#pd.read_csv("traintest_sentences.tsv", sep="\t") #

#df_drop_train.to_csv('train_sentences.tsv', index=False, sep = '\t')
#df_drop_test.to_csv('test_sentences.tsv', index=False, sep = '\t')

# SVM Regression
svr = LinearSVR(C=1.0,
          epsilon=0.2,
        max_iter=100000,
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
                             use_idf = True,
                             )
                             #stop_words='english')

# Select which set to use
SetChoice= "Combined"

#Match h to vectX_test 
h = 1000

# Combine the train and test set into one file such that tfidf vectorizer will give same feature vector length, required for when predicting
if SetChoice == "Combined":
    print("Using Combined")
    X_traintest= df_traintest.Phrase
    vectX_traintest = vectorizer.fit_transform(X_traintest)
    vectX = vectX_traintest[:156060]
    vectX_test = vectX_traintest[156060:]
    vectX = vectX[:h] #CUT FOR REG, can remove
    vectX_test = vectX_test[:h] #CUT FOR REG, can remove
    y = df_traintest[:156060].Sentiment
    y = y[:h] #CUT FOR REG, can remove


if SetChoice == "CombinedSentences":
    print("Using CombinedSentences")
    X_traintest= df_drop_traintest.Phrase
    #print(X_traintest)
    print(X_traintest[:8529]) #index doesnt match with sentence number because some sentence numbers are missing
    print(X_traintest[8529:])
    vectX_traintest = vectorizer.fit_transform(X_traintest)
    vectX = vectX_traintest[:8529]
    vectX_test = vectX_traintest[8529:]
    #vectX = vectX[:20000] #CUT FOR REG, can remove
    #vectX_test = vectX_test[:20000] #CUT FOR REG, can remove
    y = df_traintest[:8529].Sentiment
    #y = y[:20000] #CUT FOR REG, can remove
    print(vectorizer.get_feature_names())
    print("{} {} {} {}".format(np.shape(vectX_traintest),np.shape(vectX),np.shape(vectX_test),np.shape(y)) )

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

if SetChoice == "Combined" or SetChoice == "CombinedSentences":
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
    '''
    Mapping : The negative distance between the predicted(total_list) and actual labels(y) of vectX
    Set h to size of indices, label_dist, sim....
    ''' 
def get_mapping(h):
    mapping = [0 for z in range(h)]
    
    for i in range(h):
        mapping[i] = -1*(total_list[i] - y[i])
    #print('Mapping', mapping)
    
    return mapping

'''
Generate confusion matrices  
'''   
def confusion_matrix(classifier, vectX, y_true, y_pred):
    
    my_cm = cm(y_true, y_pred, labels=[0.0,1.0,2.0,3.0,4.0])
    print(my_cm)
    
    labels=[0.0,1.0,2.0,3.0,4.0]
    np.set_printoptions(precision=2)    
    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, vectX ,y_true,
                                 display_labels=labels,
                                 cmap=plt.cm.Reds,
                                 normalize=normalize)   
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)   
    plt.show()
    
'''
cm_1 -> confusion matrix for svc
'''
#cm_1 = confusion_matrix(svc, vectX, y_true, total_list)
#print(cm_1)

    
'''
Similarity function for metric labelling : 
#TODO : add ref to tutorial
'''

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

#Compare the similarity between two sentences using Wordnet
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
            
'''            
Finding the nearest neighbour of a phrase 
Returns indices or small_indices
Prints distances, indices of the nearest neighbour to x
'''
def knearest(h):
 nbrs = NearestNeighbors(algorithm='auto', leaf_size=30, n_neighbors=2, p=2,
 radius=1.0).fit(vectX_test)
 distances, indices = nbrs.kneighbors(vectX_test)
 
 #Prints the distances between datapoint x and its neighbour 
 #(including distance from x to x thus 0)
 
 print('Distances', distances)
 
 #Prints the indices of x and its neighbour
 print('Indices', indices)

 #Use small_indices to run all of the code faster
 #Make sure that the cut on the indices matches the size of sim, mapping, label_dist
 small_indices = indices[:h]
 return small_indices

''' Returns cosine similarity list : cosim
'''
def get_similarity_cosine(indices, h):
 #h is the number of lists, w is the number of items
 cosim = [0 for z in range(h)]

 for x in range(h):
     for i in indices:  
         #Cosine similarity -. most values equate to 1
         cosimscore = 1 - sc.spatial.distance.cosine(y[i[0]], y[i[1]]) 
         cosim[x] = cosimscore
 return cosim


'''
Returns wordnet similarity list : sim with dimensions w,h
Calls  sentence_similarity()
Uses X (sentence training dataframe)

def get_similarity_wordnet(indices, h):
 #h is the number of lists, w is the number of items
 #For every additional neighbour, increase item size w by 1
 w = 1
 #Initalize sim, X
 sim = [[0 for x in range(w)] for y in range(h)] 
 X = df_drop_train.Phrase

#For each i in indices, compute the similarity between 
#the sentence x and its neighbour (i[0],i[1]) and put this score into each list
#in sim.
 for k in range(h):
         for i in indices:  
             score = sentence_similarity(X.iloc[i[0]],X.iloc[i[1]])        
             #If score = None, means no similarity
             if score == None: 
                 score = 0
             sim[k] = score
 #Line print below for dealing with large datasets, checking how far the program has run
 print('Done with computing the sim')
 
 #Return sim
 return sim

'''

'''
Returns final output
ldict : the labels for the indices 
label_dist : Calculates the distance between label of x, and its neighbour
collective : Multiples label_dist by the similarity score for x and its neighbour (sim)
           : For more than 1 neighbors, please add summation function
           : alpha is hyperparameter. Can be tuned. 
mapping : Calculates the negative distance between the label assigned and (is initalized above)
final_score : The mapping + alpha(collective)

new_labels : Generate the new label of x, based on the final_score. Round this. 
'''
def getlabel(indices, sim, mapping, h):
    #Size of label_dist = 200
    ldict = [(y[x]) for x in indices]
    label_dist = [0 for z in range(h)]
    collective = [0 for z in range(h)]
    final_score = [0 for z in range(h)]
    new_labels = [0 for z in range(h)]
    alpha = 0.2
    
    for i in range(h):
        #Label distance for label_dist
        dist = np.abs(ldict[i].iloc[0]-ldict[i].iloc[1])    
        label_dist[i] = int(dist)
        # Get similarity score
        simscore = sim[i]
        #Calculate collective score
        collective[i] = (dist*alpha*simscore)
        #Get mapping
        map = mapping[i]
        #Calculate final score
        final_score[i] = map + collective[i]
        #Generate the new labels
        new_labels[i] = round(total_list[i] + final_score[i])
    
    #Print these, please comment out for large datasets
    #print('Collective:', collective)
    #print('Finale:', new_labels)
    #print('New Labels',new_labels)
    
    return new_labels

#TODO : new_labels confusion matrix
'''
Function Calls
'''
#H is the number of datapoints, set accordingly
mapping = get_mapping(h)
indices = knearest(h)
sim = get_similarity_cosine(indices, h)
#sim = get_similarity_wordnet(indices, h)
new_labels = getlabel(indices, sim, mapping,h)

my_cm = cm(y_true, total_list, labels=[0.0,1.0,2.0,3.0,4.0])
print('CM : No metric' , my_cm)

cm_metric_cosine = cm(y_true, new_labels, labels=[0.0,1.0,2.0,3.0,4.0])
print('CM Metric Cosine',cm_metric_cosine)

#print(classification_report(y_true, total_list))
#print(classification_report(y_true, new_labels))



#output = total_equation(sim, indices, label_dist)

#similarity(label_dist)
#sim = distance(labels)