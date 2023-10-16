import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score

import tuning_model
import mlflow


def process_data():
   data = pd.read_csv('https://drive.google.com/uc?id=1QsgW3apKJ8-PazRbQKTzkZrW76CZ--qQ&export=download')
   
   # retirando os valores null
   data = data[data['Gender'].notna()]
   data = data[data['Married'].notna()]
   data = data[data['LoanAmount'].notna()]
   data = data[data['Loan_Amount_Term'].notna()]

   # Preenchendo os vazios com 0
   data['Dependents'] = data['Dependents'].fillna('0')
   data['Credit_History'] = data['Credit_History'].fillna(0)

   # Preenchendo os vazios com No
   data['Self_Employed'] = data['Self_Employed'].fillna('No')
   
   # Trocando os valores para 0 e 1
   data.loc[data['Loan_Status'] == 'Y', 'Loan_Status'] = 1
   data.loc[data['Loan_Status'] == 'N', 'Loan_Status'] = 0
   
   data.loc[data['Married'] == 'Yes', 'Married'] = 1
   data.loc[data['Married'] == 'No', 'Married'] = 0
   
   data.loc[data['Self_Employed'] == 'Yes', 'Self_Employed'] = 1
   data.loc[data['Self_Employed'] == 'No', 'Self_Employed'] = 0
   
   data.loc[data['Education'] == 'Graduate', 'Education'] = 1
   data.loc[data['Education'] == 'Not Graduate', 'Education'] = 0

   data.loc[data['Gender'] == 'Male', 'Gender'] = 1
   data.loc[data['Gender'] == 'Female', 'Gender'] = 0

   data.loc[data['Property_Area'] == 'Rural', 'Property_Area'] = 0
   data.loc[data['Property_Area'] == 'Semiurban', 'Property_Area'] = 1
   data.loc[data['Property_Area'] == 'Urban', 'Property_Area'] = 2

   data.loc[data['Dependents'] == '3+', 'Dependents'] = 3
   
   # Convertendo os campos para int
   data['Dependents'] = data['Dependents'].astype(int)
   data['Gender'] = data['Gender'].astype(int)
   data['Married'] = data['Married'].astype(int)
   data['Education'] = data['Education'].astype(int)
   data['Self_Employed'] = data['Self_Employed'].astype(int)
   data['Property_Area'] = data['Property_Area'].astype(int)
   data['Loan_Status'] = data['Loan_Status'].astype(int)
   
   # Retirando a coluna de Loan_ID
   data.drop('Loan_ID', axis=1, inplace=True)
   
   X = data.drop('Loan_Status', axis=1)
   y = data['Loan_Status']
   
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
   return X_train, X_test, y_train, y_test


def create_model(X_train, y_train):
   model = tuning_model.get_best_model(X_train, y_train)
   return model


def train_model(model, X_train, y_train):
   with mlflow.start_run(run_name='experiment_01') as run:
      model.fit(X_train, y_train)


def predict_model(model, X_test):
   y_pred = model.predict(X_test)
   mse = mean_squared_error(y_test, y_pred)
   f1 = f1_score(y_test, y_pred)
   acc = accuracy_score(y_test, y_pred)

   result_array = [mse, f1, acc]

   return result_array

#y_test

#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)

#sns.heatmap(cm,annot=True)

#model.score(X_test, y_test)


if __name__ == '__main__':
   X_train, X_test, y_train, y_test = process_data()
   model = create_model(X_train, y_train)
   mlflow.config_mlflow()
   train_model(model, X_train, y_train)
