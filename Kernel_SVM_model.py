#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset_train = pd.read_csv('train.csv')
x_train = dataset_train.iloc[:,[2,4,5,9]].values
y_train = dataset_train.iloc[:,1].values
dataset_test_x = pd.read_csv('test.csv')
x_test = dataset_test_x.iloc[:,[1,3,4,8]].values
dataset_test_y = pd.read_csv('gender_submission.csv')
y_test = dataset_test_y.iloc[:,1].values

#Cleaning the data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=0)
imputer = imputer.fit(x_train[:,[2]])
x_train[:,[2]] = imputer.transform(x_train[:,[2]])
imputer = imputer.fit(x_test[:,[2,3]])
x_test[:,[2,3]] = imputer.transform(x_test[:,[2,3]])

#Encoding categorical features
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x_train[:,1] = labelencoder_x.fit_transform(x_train[:,1])
x_test[:,1] = labelencoder_x.fit_transform(x_test[:,1])
onehotencoder_x = OneHotEncoder(categorical_features=[1])
x_train = onehotencoder_x.fit_transform(x_train).toarray()
x_test = onehotencoder_x.fit_transform(x_test).toarray()

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

#Fitting classifier to training set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(x_train,y_train)

#Predicting the test set results
y_pred = classifier.predict(x_test)

#Creating confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#visualizing the test set reults
import seaborn as sns
sns.set(style='darkgrid')
passing = sns.load_dataset('titanic')
ax = sns.countplot(x="class",data=passing)
g = sns.catplot(x="class", hue="who", col="survived",
                data=passing, kind="count",
                height=4, aspect=.7)

#Saving the predictions into a csv file
output = pd.read_csv('gender_Submission.csv')
output = final.drop(['Survived'],axis=1)
revision_1 = pd.DataFrame(data=y_pred)
revision_1.columns = ["Survived"]
revision_2 = pd.concat([output,revision_1],axis=1)
revision_2.to_csv('output.csv')
