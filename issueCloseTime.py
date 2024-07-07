from math import ceil
from sklearn.tree import DecisionTreeClassifier
from decisiontree import CustomDecisionTreeClassifier
from predictor import *
from scipy.io import arff
import os
from dendogram import dendogramclassifer
from randomforestdendogram import randomforestclassifer
def make_dataset(data,label_true = "1 day", label_false = "> 1 day", split_yval = b'1'):
    data.iloc[:,-1] = np.where(data.iloc[:,-1].astype(int) <= int(split_yval), label_true, label_false)
    return data
dir = 'featuresSelected'
arff_files = os.listdir(dir)
splits = [b'1',b'7',b'14',b'30',b'90']
labels_true_display = ["1 day","7 day","14 day", "30 day","90 day"]
#labels_false = ["> 1 day","> 7 day","> 14 day", "> 30 day","> 90 day"]

labels_true = ["1","1","1", "1","1"]
labels_false = ["0","0","0", "0","0"]

def preprocess(data):
    data = data.drop_duplicates()
    def rename_columns(df):
        new_columns = {}
        for i, col in enumerate(df.columns):
            if i == len(df.columns) - 1:  # Last column (Y)
                new_columns[col] = col + '!'
            elif pd.api.types.is_numeric_dtype(df[col]):
                new_columns[col] = col.capitalize()
            else:
                new_columns[col] = col.lower()
        return df.rename(columns=new_columns)
    data = rename_columns(data)
    X, y = split(-1,data)
    sm = SMOTE(random_state = 12345) 
    X, y = sm.fit_resample(X,  y.to_numpy())
    X['timeopen!'] = y
    return X
for filename in arff_files:
    if filename=="combined.arff":
        continue
    data, meta = arff.loadarff(dir+'/'+filename)
    df = pd.DataFrame(data)
    df.columns = meta.names()
    for threshold, label_true, label_true_display, label_false in zip(splits,labels_true, labels_true_display, labels_false):
        dfcopy = df.copy()
        dataset = make_dataset(dfcopy,label_true=label_true, label_false=label_false,split_yval=threshold)
        label = filename+" data:"+str(label_true_display)
        print("############Results for "+label+"########")
        #run_models([dataset], models = {"Decision Tree":DecisionTreeClassifier(criterion='entropy',min_samples_split=ceil(len(dataset)/25), min_samples_leaf=ceil(len(dataset)/25)),"Naive Bayes":GaussianNB()},preprocessor= lambda data: data)
        #run_models([dataset],models = {"dendogram":dendogramclassifer(),"Decision Tree":DecisionTreeClassifier(criterion='entropy',min_samples_split=ceil(len(dataset)/25), min_samples_leaf=ceil(len(dataset)/25)),"Naive Bayes":GaussianNB()},preprocessor= lambda data: data.drop_duplicates())
        #run_models([dataset],models = {"dendogram":dendogramclassifer(),"random forest":randomforestclassifer(),"Decision Tree":CustomDecisionTreeClassifier(criterion='entropy')},preprocessor= preprocess)
        #run_models(label, [dataset],models = {"dendogram":dendogramclassifer(),"random forest":randomforestclassifer(),"Decision Tree":CustomDecisionTreeClassifier(criterion='entropy')},preprocessor= preprocess)
        run_models(label, [dataset],models = {"dendogram":dendogramclassifer()},preprocessor= preprocess)
