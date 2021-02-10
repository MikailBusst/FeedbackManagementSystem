import pandas as pd
import numpy as np
from traits.api import *
from traitsui.api import *
from traitsui.menu import *
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

class Comment(HasTraits):
    """ Comment object """

    comment = Str()
    result = Str()
    analyse = Button()
    methods = Enum('Count Vectorisation Technique', 'Term Frequency-Inverse Document Frequency')

    def _analyse_changed(self):
        if self.methods == 'Count Vectorisation Technique':    
            if self.comment == '':
                self.result = "No comment was entered."
            else:
                ess_model = RandomForestClassifier()
                ess_model.fit(X_train_data_new_cv,Y_train_data)                             
                review=self.comment
                new_test_transform = count_vector.transform([review])        

                if ess_model.predict(new_test_transform):
                    self.result="Good"
                else :
                    self.result="Bad"
        elif self.methods == 'Term Frequency-Inverse Document Frequency':
            if self.comment == '':
                self.result = "No comment was entered."
            else:
                svm_model = SVC()
                svm_model.fit(X_train_data_new_tfidf,Y_train_data)                             
                review=self.comment
                new_test_transform = tfidf_vector.transform([review])        

                if svm_model.predict(new_test_transform):
                    self.result="Good"
                else :
                    self.result="Bad"

amazon_data = pd.read_csv("Amazon_Unlocked_Mobile.csv")
amazon_data = amazon_data.dropna(axis = 0)
amazon_data=amazon_data[["Reviews","Rating"]]
amazon_data_pos=amazon_data[amazon_data["Rating"].isin([4,5])]
amazon_data_neg=amazon_data[amazon_data["Rating"].isin([1,2])]
amazon_data_filtered=pd.concat([amazon_data_pos[:20000],amazon_data_neg[:20000]])
amazon_data_filtered["r"]=1
amazon_data_filtered["r"][amazon_data_filtered["Rating"].isin([1,2])]= 0

X_train_data,x_test_data,Y_train_data,y_test_data=train_test_split(amazon_data_filtered["Reviews"],amazon_data_filtered["r"],test_size=0.2)

tfidf_vector = TfidfVectorizer(stop_words="english")
tfidf_vector.fit(X_train_data)
X_train_data_new_tfidf=tfidf_vector.transform(X_train_data)
x_test_data_new_tfidf=tfidf_vector.transform(x_test_data)

count_vector=CountVectorizer(stop_words="english")
count_vector.fit(X_train_data)
X_train_data_new_cv=count_vector.transform(X_train_data)
x_test_data_new_cv=count_vector.transform(x_test_data)

predictions = dict()

CommentView = View(Item(name = "comment", springy=True, style='custom'), Item('result', show_label=False, style='readonly'), Item(name="methods", editor=EnumEditor(values={'Count Vectorisation Technique' : '1. Count Vectorisation Technique', 'Term Frequency-Inverse Document Frequency' : '2. Term Frequency-Inverse Document Frequency'})), Item(name="analyse", show_label=False, editor=ButtonEditor(label="Analyse")), buttons = [CancelButton], title="Insert Comment", resizable=True)

if  __name__ == "__main__":
   comment = Comment()
   comment.configure_traits(view=CommentView)
