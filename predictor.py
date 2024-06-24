from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score   
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import confusion_matrix
import numpy as np

def get_train_test_split(train, test, class_index):
    train_data_merged = pd.concat(train)
    train_data_merged.drop('name',axis=1, inplace=True)
    test_data_merged = pd.concat(test)
    test_data_merged.drop('name',axis=1, inplace=True)
    X_train = train_data_merged.iloc[:,:class_index]
    y_train = train_data_merged.iloc[:,class_index]
    y_train[y_train > 0] = 1
    y_train = y_train.astype('str')
    X_test = test_data_merged.iloc[:,:class_index] 
    y_test = test_data_merged.iloc[:,class_index]
    y_test[y_test > 0] = 1
    y_test = y_test.astype('str')
    return X_train, X_test, y_train, y_test

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
    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    false_alarm = FP / (FP+TN)
    accuracy = (TP+TN) / (TP+FP+TN+FN)
    F1 = (2*precision*recall) / (precision+recall)
    return precision, recall, false_alarm, accuracy, F1

def print_classifier_result(model, res, y_test):
    print("-------------Results for "+model+"--------------")
    confusion_matrix = get_confusion_matrix(res, y_test, ['1','0'])
    print_tabular(confusion_matrix)
    TP, TN, FP, FN = get_confusion_mat_entries(confusion_matrix)
    print_tabular([get_metrices(TP, TN, FP, FN)],ind = ['values'], cols = ['precision','recall','false alarm','accuracy','F-1 score'])

train_data = read_and_populate_dataset('PROMISE-backup-master/PROMISE-backup-master/bug-data/jedit','jedit',['4.1','4.2','4.3','3.2'])
test_data = read_and_populate_dataset('PROMISE-backup-master/PROMISE-backup-master/bug-data/jedit','jedit',['4.0'])

X_train, X_test, y_train, y_test = get_train_test_split(train_data,test_data,-1)
args = {"n_estimators":10}    

modelrf = build_model(X_train, y_train,RandomForestClassifier(),**args)
args = {"n_neighbors":10}
res = predict(modelrf, X_test)
print_classifier_result("Random Forest", res,y_test)    
modelknn = build_model(X_train, y_train,KNeighborsClassifier(),**args)
res = predict(modelknn, X_test)
print_classifier_result("KNN", res,y_test)    
modelNB = build_model(X_train, y_train,GaussianNB())
res = predict(modelNB, X_test)
print_classifier_result("Naive Bayes", res,y_test)    

#print(predict(modelrf, X_test))
#print(y_test)
#stats(predict(modelrf, X_test),y_test, 'RANDOM FOREST')
#stats(predict(modelknn, X_test),y_test,'KNN')
#stats(predict(modelNB, X_test),y_test,'NAIVE BAYES')
