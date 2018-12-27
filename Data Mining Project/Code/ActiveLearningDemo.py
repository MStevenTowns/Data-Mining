import numpy as np
import matplotlib.pyplot as plt
import csv
from copy import deepcopy
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from modAL.models import ActiveLearner, Committee
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import warnings
warnings.filterwarnings("ignore")
#http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
data=[]
lab=[]
realdata=[]
reallab=[]
posdict={"NotSpecified":0,"Standing":1,"Sitting":2,"Walking":3,"Running":4,"Climbing(up)":5,"Climbing(down)":6}

def readDataSet(datareq=0):
    global data,lab
    if(datareq==0):
        filename='../subject1/data/IntergratedData.csv'
    else:
        filename='../subject1/data/IntergratedDataDay2.csv'
    with open(filename, newline='') as csvfile:
        csvreader=csv.reader(csvfile,delimiter=',',quotechar='|')
        count=0
        for row in csvreader:
            if count==0:
                count=1
            else:
                #if(posdict[row[7]]!=0):
                    if(datareq==0):
                        data.append(row[1:7])
                        lab.append(posdict[row[7]])
                    else:
                        realdata.append(row[1:7])
                        reallab.append(posdict[row[7]])


def calPerformance(y_pred,y_real,title):
    
    cm = confusion_matrix(y_target=y_real.tolist(),y_predicted=y_pred.tolist(),binary=True,positive_label=1)
    TP=cm[0][0]
    FN=cm[0][1]
    FP=cm[1][0]
    TN=cm[1][1]

    #Confusion Matrix

    print(title+":")
    #print("Title:%s, accuracy:%f"%(title,accuracy_score(y_pred,y_real)))
    print(cm)
    Recall=TP/(FN+TP)
    TNR=FP/(TN+FP)
    Precision=TP/(FP+TP)
    # Overall accuracy
    Accuracy = (TP+TN)/(TP+FP+FN+TN)

    print("Recall:%f\t\tTNR:%f"%(Recall,TNR))
    print("Accuracy:%f\tPrecision:%f"%(Accuracy,Precision))
          
# visualizing the classes
readDataSet()
readDataSet(1)

print()
print()
X=np.array(deepcopy(data))
y=np.array(deepcopy(lab))
X_pool=X
y_pool=y
X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=0.95, random_state=20)
xtestlen=len(X_pool)
#clsf = KNeighborsClassifier(7)
#clsf=RandomForestClassifier(max_depth=5, n_estimators=5, max_features=1)
clsf=DecisionTreeClassifier()
clsf.fit(X_pool, y_pool)
pred=clsf.predict(np.array(realdata))

calPerformance(pred,np.array(reallab),"Decision Tree Classifier without Active Learning")
print("score:%f"%accuracy_score(pred,reallab))
print("How many labeled data we used:%d"%(xtestlen))
# initializing Committee members
n_members = 5
learner_list = list()

for member_idx in range(n_members):
    # initial training data
    n_initial = 7
    train_idx = np.random.choice(range(X_pool.shape[0]), size=n_initial, replace=False)
    X_train = X_pool[train_idx]
    y_train = y_pool[train_idx]

    # creating a reduced copy of the data with the known instances removed
    X_pool = np.delete(X_pool, train_idx, axis=0)
    y_pool = np.delete(y_pool, train_idx)

    # initializing learner
    learner = ActiveLearner(
        #estimator=RandomForestClassifier(max_depth=5, n_estimators=5, max_features=1),
        #estimator=KNeighborsClassifier(7),
        estimator=DecisionTreeClassifier(),
        X_training=X_train, y_training=y_train
    )
    learner_list.append(learner)

# assembling the committee
committee = Committee(learner_list=learner_list)
# visualizing the Committee's predictions per learner
print()
print()
# query by committee
n_queries = 10
for idx in range(n_queries):
        query_idx, query_instance = committee.query(X_pool)
        committee.teach(
            X=X_pool[query_idx].reshape(1, -1),
            y=y_pool[query_idx].reshape(1, )
        )
        # remove queried instance from pool
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx)

y_pred=committee.predict(np.array(realdata))
calPerformance(y_pred,np.array(reallab),"Decision Tree Classifier with Active Learning(10 query iterations)")
print("score:%f"%accuracy_score(y_pred,reallab))
print("How many labeled data we used:%d"%(xtestlen-len(X_pool)))
print()
print()
