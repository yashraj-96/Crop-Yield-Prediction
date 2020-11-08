
#Importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

    
#Importing the dataset
Yield= pd.read_csv('APY_13_14_to_17_18.csv')
Yield.head()

#Creating dataframes for respective csv files
df1 = pd.DataFrame(Yield)
df1.columns

#Treating the null/ NA values
df1.isnull()
print(df1.isnull().sum())

Dist_name = df1['District_name']
Dist_code = df1['District_code']
a1 = Dist_name.mode()
a2 = Dist_code.median()
df1.columns
df1['District_name'].fillna('Belagavi',inplace=True)
df1['District_code'].fillna('370',inplace=True)


#State_name has high correlation with State_code
#Crop_name has high correlation with crop_category
#Production needs to be dropped , highly skewed
df1.drop(['State_name','Production','Crop_name'],axis=1,inplace=True)


# Label Enconding the categorial columns
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
x1 = df1['Year']
x1 = LE.fit_transform(x1)
df1['Year']= x1

x2 = df1['Season']
x2 = LE.fit_transform(x2)
df1['Season']= x2

x3 = df1['Crop_category']
x3 = LE.fit_transform(x3)
df1['Crop_category']= x3

x4 = df1['District_name']
x4 = LE.fit_transform(x4)
df1['District_name']= x4

#Splitting the data into input and output variables
X = df1.iloc[:,0:8].values
Y = df1.iloc[:,8].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score    
model = RandomForestRegressor()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)  

#Model Evaluation
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
print(rmse)

###############################################################################################################
#Deploying the model
import pickle
# Saving model to disk
pickle.dump(model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
###########################################################################################################




































# Saving the model for Future Inferences

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")

# imports

from keras import model_from_json 

# opening and store file in a variable

json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()

# use Keras model_from_json to make a loaded model

loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("model.h5")
print("Loaded Model from disk")

# compile and evaluate loaded model

loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
























