import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier

import tuning_model


def process_data(data):   
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
   if 'Loan_Status' in data.columns:
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

   if 'Loan_Status' in data.columns:
      data['Loan_Status'] = data['Loan_Status'].astype(int)

   # Retirando a coluna de Loan_ID
   data.drop('Loan_ID', axis=1, inplace=True)

   return data


def split_data(data):
   X = data.drop('Loan_Status', axis=1)
   y = data['Loan_Status']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   return X_train, X_test, y_train, y_test


def create_model(X_train, y_train):
   # Função para pegar o melhor modelo e hyperparametros (como a função demora em torno de 2hrs deixei já setado o melhor encontrado)
   #model = tuning_model.get_best_model(X_train, y_train)
   
   model = GradientBoostingClassifier(learning_rate=0.01, loss='exponential', max_depth=4,
                                    max_features='sqrt', min_samples_leaf=2, 
                                    min_samples_split=5, n_estimators=250,
                                    subsample=0.8)

   return model


def fit_model(model, X_train, y_train):
   model.fit(X_train, y_train)

def evaluate_model(model, X_test, y_test):
   y_pred = model.predict(X_test)
   return {
      'roc_auc': roc_auc_score(y_test, y_pred),
      'accuracy': accuracy_score(y_test, y_pred),
      'precision': precision_score(y_test, y_pred),
      'recall': recall_score(y_test, y_pred),
      'f1': f1_score(y_test, y_pred),
      'mse': mean_squared_error(y_test, y_pred)
      }

def pred_model_validation(model, data_val_df):
   y_pred_real = model.predict(data_val_df)
   data_val_df['Loan_Status_Predicted'] = y_pred_real
   data_val_df.to_csv('dados_teste_com_previsoes.csv', index=False)


if __name__ == '__main__':
   # dados de treinamento
   data = pd.read_csv('https://drive.google.com/uc?id=1QsgW3apKJ8-PazRbQKTzkZrW76CZ--qQ&export=download')

   # dados de validação
   data_val = pd.read_csv('https://drive.google.com/uc?id=18dY8nfISSjm0ODCDywqwGZxpj_YHl_vx&export=download')

   data_df = process_data(data)
   X_train, X_test, y_train, y_test = split_data(data_df)
   model = create_model(X_train, y_train)
   fit_model(model, X_train, y_train)
   result = evaluate_model(model, X_test, y_test)
   #print(result)

   data_val_df = process_data(data_val)
   pred_model_validation(model, data_val_df)