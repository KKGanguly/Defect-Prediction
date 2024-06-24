from sklearn.tree import DecisionTreeClassifier
from predictor import *
from scipy.io import arff
import os

def make_dataset(data,label_true = "1 day", label_false = "> 1 day", split_yval = b'1'):
    data.iloc[:,-1] = np.where(data.iloc[:,-1] == split_yval, label_true, label_false)
    return data
dir = 'featuresSelected'
arff_files = os.listdir(dir)
splits = [b'1',b'7',b'14',b'30',b'90']
labels_true = ["1 day","7 day","14 day", "30 day","90 day"]
labels_false = ["> 1 day","> 7 day","> 14 day", "> 30 day","> 90 day"]

for filename in arff_files:
    if filename=="combined.arff":
        continue
    data, meta = arff.loadarff(dir+'/'+filename)
    df = pd.DataFrame(data)
    df.columns = meta.names()
    for split, label_true, label_false in zip(splits,labels_true, labels_false):
        dataset = make_dataset(df.copy(),label_true=label_true, label_false=label_false,split_yval=split)
        print("############Results for "+filename+" data:"+str(label_true)+"########")
        run_models([dataset], models = {"Decision Tree":DecisionTreeClassifier(),"Naive Bayes":GaussianNB()},preprocessor= lambda data: data)
