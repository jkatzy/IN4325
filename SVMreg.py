from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

from doc2vec import Vectorizer

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

train_x, train_y = Vectorizer("train.tsv").get_model()
# test_x, test_y = Vectorizer("test.tsv", train=False).get_model()

# K fold cross evaluation for SVM OVA
#skf = StratifiedKFold(n_splits=5)

scores = cross_val_score(svc, train_x, train_y, cv=3)
print(scores)
# for train, test in skf.split(train_x, train_y):
#     svc.fit(train_x, train_y)
#     train_score = svc.score(train_x[train], train_y[train])
#     test_score = svc.score(train_x[test], train_y[test])
#     print("SVM OVA: Train Score = {}, Test Score= {}".format(train_score, test_score))

