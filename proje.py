from sklearn.metrics import accuracy_score

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

import pandas as pd
import numpy as np

import os

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report

import joblib


class Model:
    def __init__(self):
        self.train_data = pd.DataFrame({'text': [], 'category': []})
        self.test_data = pd.DataFrame({'text': [], 'category': []})
                
    def get_trainData(self):
        return self.train_data
        
    def get_testData(self):
        return self.test_data

    def add_trainDataset(self, trainData_add):
        self.train_data = pd.concat([self.train_data, trainData_add], ignore_index = True, axis = 0)

    def add_testDataset(self, testData_add):
        self.test_data = pd.concat([self.test_data, testData_add], ignore_index = True, axis = 0)
        
    def get_n_classes(self, df):
        return df["category"].nunique()
    
    def preparetestDataX(self):
        X_test = self.test_data['text'].to_numpy()
        return X_test
     
    def preparetestDataY(self):
        y_test = self.test_data["category"].map({"history":0,
                                   "math": 1,
                                   "medicine": 2})
        return y_test

    def prepareDataY(self):
        y = self.train_data["category"].map({"history":0,
                               "math": 1,
                               "medicine": 2})

        n_classes = self.get_n_classes(self.train_data)
        print("number of classes: "+ str(n_classes))
        return y
        
    def prepareDataX(self):
        X = self.train_data['text'].to_numpy()      
        return X

    def trainig(self, path):
        X = self.prepareDataX()
        y = self.prepareDataY()
        X_test = self.preparetestDataX()
        y_test = self.preparetestDataY()
        print("Train data: " + str(self.train_data.shape))
        
        self.logreg = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', LogisticRegression(n_jobs=1, C=1e5)),
                       ])
        
        print("Olusan Modelin Basari Oranlari")
        self.logreg.fit(X, y)
         
        y_pred = self.logreg.predict(X_test)
        my_tags = ['history', 'math', 'medicine']
        print('accuracy %s' % accuracy_score(y_pred, y_test))
        print(classification_report(y_test, y_pred,target_names=my_tags))    
        
        joblib.dump(self.logreg, path)
    
    def load_model(self, path):
        self.logreg = joblib.load(path)
        
        
class Dataset:
    def __init__(self, dataPath):
        self.dataPath = dataPath
        self.page_data = []
        self.data_df = pd.DataFrame({'text': [], 'category': []})
        
    def pageByPage(self):        
        for page_layout in extract_pages(self.dataPath):
            extract_Data = ""
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    textData = element.get_text()
                    if len(textData) > 2:
                        textData = textData.replace("\n"," ")
                    extract_Data += textData
            self.page_data.append(extract_Data)
    
    def convertDf(self, label):
        df = pd.DataFrame({'text': [], 'category': []})
        for i in self.page_data:
            df_add = pd.DataFrame({'text': [i], 'category': [label]})
            df = pd.concat([df, df_add], ignore_index = True, axis = 0)
        self.data_df = pd.DataFrame(df, columns= ['text', 'category'])

    def set_dataset(self, label):
        self.pageByPage()
        self.convertDf(label)

    def get_dataset(self):
        return self.data_df


def addData(model, path, label, mode):
    os.chdir(path)
    for i in os.listdir():
        data = Dataset(i)
        data.set_dataset(label)
        if mode == 0:            
            model.add_trainDataset(data.get_dataset())
        else:
            model.add_testDataset(data.get_dataset())

# Kodun bulundugu ana klasor
path_main = os.path.dirname(os.path.abspath(__file__))  

# Test datalarinin bulundugu klasorler              
path_historyTest = path_main + r"/test veri kümesi/tarih"
path_mathTest = path_main + r"/test veri kümesi/matematik"
path_medTest = path_main + r"/test veri kümesi/tip"

# Egitim datalarinin bulundugu klasorler
path_historyTrain = path_main + r"/egitim/tarih"
path_mathTrain = path_main + r"/egitim/matematik"
path_medTrain = path_main + r"/egitim/tip"

# Modelin kaydedilecegi ve yuklenecegi adres
path_model = path_main + "/modelSave.joblib"


model = Model()

print("Tarih eğitim dataseti yükleniyor")
addData(model, path_historyTrain, "history", 0)
print("Matematik eğitim dataseti yükleniyor")
addData(model, path_mathTrain, "math", 0)
print("Tıp eğitim dataseti yükleniyor")
addData(model, path_medTrain, "medicine", 0)

print("Tarih test dataseti yükleniyor")
addData(model, path_historyTest, "history", 1)
print("Matematik test dataseti yükleniyor")
addData(model, path_mathTest, "math", 1)
print("Tıp test dataseti yükleniyor")
addData(model, path_medTest, "medicine", 1)

print("Eğitim yapılıyor")
model.trainig(path_model)
print("Eğitim bitti\n")

print("Model " + path_model + " adresine kaydedildi\n")
