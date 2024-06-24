from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score   
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import confusion_matrix
import numpy as np
from collections import defaultdict
try:
    from ezr_24Jun14.ezr import SOME, report
except ImportError as e:
    print("ImportError:", e)


def cross_validate(data, class_index, model,drop_names, preprocessor):
    data = merge_drop(data,drop_names)
    preprocessor(data)
    X, y = split(class_index,data)
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    unique_labels = y.unique()
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = build_model(X_train, y_train, model)
        res = predict(model, X_test)
        yield get_result(res,y_test,unique_labels)    
def preprocess(data):
    data.iloc[:,-1] = np.where(data.iloc[:,-1] > 0, "1", "0")
def run_models(data, models, drop_names=[], preprocessor = preprocess):
    results_precision = []
    results_recall = []
    results_false_alarm = []
    results_accuracy = []
    results_f1 = []
    for name, model in models.items():
        model_results = cross_validate(data,-1,model,drop_names, preprocessor = preprocessor)
        result_precision = []
        result_recall = []
        result_false_alarm =[] 
        result_accuracy = []
        result_f1 = []
        for precision, recall, false_alarm, accuracy, F1 in model_results:
            result_precision.append(round(precision,2))
            result_recall.append(round(recall,2))
            result_false_alarm.append(round(false_alarm,2))
            result_accuracy.append(round(accuracy,2))
            result_f1.append(round(F1,2))
        results_precision.append(SOME(inits=result_precision,txt=name))
        results_recall.append(SOME(inits=result_recall,txt=name))
        results_false_alarm.append(SOME(inits=result_false_alarm,txt=name))
        results_accuracy.append(SOME(inits=result_accuracy,txt=name))
        results_f1.append(SOME(inits=result_f1,txt=name))
    print("##Precision Report")
    report(results_precision)
    print("##Recall Report")
    report(results_recall)
    print("##False Alarm Report")
    report(results_false_alarm)
    
def get_train_test_split(train, test, class_index):
    train_data_merged = merge_drop(train)
    test_data_merged = merge_drop(test)
    X_train, y_train = split(class_index, train_data_merged)
    X_test ,y_test = split(class_index, test_data_merged)
    return X_train, X_test, y_train, y_test

def split(class_index, train_data_merged):
    X_train = train_data_merged.iloc[:,:class_index]
    y_train = train_data_merged.iloc[:,class_index]
    return X_train ,y_train

def merge_drop(train,drop_names):
    train_data_merged = pd.concat(train)
    for drop_name in drop_names:
        train_data_merged.drop('name',axis=1, inplace=True)
    return train_data_merged

def read_and_populate_dataset(path, project, version):
    return [pd.read_csv(path+'/'+project+'-'+v+'.csv') for v in version]


def build_model(X_train, y_train, model, **params):
    model_nb = model
    if params: model.set_params(**params)
    model_nb.fit(X_train, y_train)
    return model

def predict(model, X_test):
    return model.predict(X_test)

def stats(y_pred, y_test,model_name):
    print("-------CLASSIFICATION RESULTS FOR "+ model_name+'--------')
    print("Accuracy Score:", accuracy_score(y_pred, y_test))
    print(classification_report(y_pred, y_test))

def get_confusion_matrix(y_pred, y_true, lbls):
    return confusion_matrix(y_pred, y_true, labels = lbls)

def print_tabular(matrix, ind=['true:yes', 'true:no'], cols = ['pred:yes', 'pred:no']):
    print_matrix = pd.DataFrame(matrix, index=ind, 
                        columns=cols)
    print(print_matrix)

def get_confusion_mat_entries(confusion_matrix):
    true = np.diag(confusion_matrix)
    false = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix) 
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
    TP = true[0]
    TN = true[1]
    FP = false[0]
    FN = false[1]
    return TP, TN, FP, FN

def get_metrices(TP, TN, FP, FN):
    precision = TP / (TP+FP) if (TP+FP)>0 else 0
    recall = TP / (TP+FN) if (TP+FN)>0 else 0
    false_alarm = FP / (FP+TN) if (FP+TN)>0 else 0
    accuracy = (TP+TN) / (TP+FP+TN+FN)
    F1 = (2*precision*recall) / (precision+recall) if (precision+recall)>0 else 0
    return precision, recall, false_alarm, accuracy, F1

def get_result(res, y_test, labels):
    confusion_matrix = get_confusion_matrix(res, y_test, labels)
    TP, TN, FP, FN = get_confusion_mat_entries(confusion_matrix)
    return get_metrices(TP, TN, FP, FN)

data = read_and_populate_dataset('PROMISE-backup-master/PROMISE-backup-master/bug-data/jedit','jedit',['4.0','4.1','4.2','4.3','3.2'])
#run_models(data, drop_names=["name"])
#test_data = read_and_populate_dataset('PROMISE-backup-master/PROMISE-backup-master/bug-data/jedit','jedit',['4.0'])
"""
X_train, X_test, y_train, y_test = get_train_test_split(train_data,test_data,-1)
args = {"n_estimators":10}    
result = []

modelrf = build_model(X_train, y_train,RandomForestClassifier(),**args)
args = {"n_neighbors":10}
res = predict(modelrf, X_test)
print_classifier_result("Random Forest", res,y_test)    
modelknn = build_model(X_train, y_train,KNeighborsClassifier(),**args)
res = predict(modelknn, X_test)
print_classifier_result("KNN", res,y_test)    
modelNB = build_model(X_train, y_train,GaussianNB())
res = predict(modelNB, X_test)
precision, recall, false_alarm, accuracy, F1 = get_result("Naive Bayes", res,y_test)    
result.append(SOME([precision]*n,   txt="x1"))

#print(predict(modelrf, X_test))
#print(y_test)
#stats(predict(modelrf, X_test),y_test, 'RANDOM FOREST')
#stats(predict(modelknn, X_test),y_test,'KNN')
#stats(predict(modelNB, X_test),y_test,'NAIVE BAYES')
"""