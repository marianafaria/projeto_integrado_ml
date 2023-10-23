import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, roc_auc_score, precision_score, recall_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier


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
   
   data.loc[(data['Married'] == 'Yes') | (data['Married'] == 'Sim'), 'Married'] = 1
   data.loc[(data['Married'] == 'No') | (data['Married'] == 'Não'), 'Married'] = 0
   
   data.loc[(data['Self_Employed'] == 'Yes') | (data['Self_Employed'] == 'Sim'), 'Self_Employed'] = 1
   data.loc[(data['Self_Employed'] == 'No') | (data['Self_Employed'] == 'Não'), 'Self_Employed'] = 0
   
   data.loc[(data['Education'] == 'Graduate') | (data['Education'] == 'Sim'), 'Education'] = 1
   data.loc[(data['Education'] == 'Not Graduate') | (data['Education'] == 'Não'), 'Education'] = 0

   data.loc[(data['Gender'] == 'Male') | (data['Gender'] == 'Masculino'), 'Gender'] = 1
   data.loc[(data['Gender'] == 'Female') | (data['Gender'] == 'Feminino'), 'Gender'] = 0

   data.loc[data['Property_Area'] == 'Rural', 'Property_Area'] = 0
   data.loc[(data['Property_Area'] == 'Semiurban') | (data['Property_Area'] == 'Semi-Urbano'), 'Property_Area'] = 1
   data.loc[(data['Property_Area'] == 'Urban') | (data['Property_Area'] == 'Urbano'), 'Property_Area'] = 2

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

   # Retirando a coluna de Loan_ID se ela existir
   if 'Loan_ID' in data.columns:
      data.drop('Loan_ID', axis=1, inplace=True)

   return data


def split_data(data):
   X = data.drop('Loan_Status', axis=1)
   y = data['Loan_Status']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   return X_train, X_test, y_train, y_test


def create_model(X_train, y_train):
   gb_model = GradientBoostingClassifier(learning_rate=0.01, loss='exponential', max_depth=3,
                                       max_features='sqrt', min_samples_leaf=2,
                                       min_samples_split=5, n_estimators=250,
                                       subsample=0.8)

   rf_model = RandomForestClassifier(n_estimators=50, max_depth=3,
                                    class_weight='balanced',
                                    criterion='entropy',
                                    max_features='sqrt',
                                    min_samples_leaf=2, min_samples_split=2,
                                    random_state=42)

   # Crie o ensemble de modelos
   model = VotingClassifier(estimators=[
      ('gb', gb_model),
      ('rf', rf_model)
      ], voting='hard')  # Use 'hard' para classificação, 'soft' se os modelos retornarem probabilidades

   return model


def fit_model(model, X_train, y_train):
   model.fit(X_train, y_train)

def evaluate_model(model, X_test, y_test):
   y_pred = model.predict(X_test)
   result = {
      'roc_auc': [roc_auc_score(y_test, y_pred)],
      'accuracy': [accuracy_score(y_test, y_pred)],
      'precision': [precision_score(y_test, y_pred)],
      'recall': [recall_score(y_test, y_pred)],
      'f1': [f1_score(y_test, y_pred)]
   }

   result_df = pd.DataFrame(data=result)
   result_df.to_csv('dados_metricas.csv', index=False)

   return result_df


def pred_model_validation(model, data_val_df):
   y_pred_real = model.predict(data_val_df)
   data_val_df['Loan_Status_Predicted'] = y_pred_real
   data_val_df.to_csv('dados_teste_com_previsoes.csv', index=False)

   return data_val_df['Loan_Status_Predicted']


def train_model(data):

   data_df = {
      'Gender': [data.Gender],
      'Married': [data.Married],
      'Dependents': [data.Dependents],
      'Education': [data.Education],
      'Self_Employed': [data.Self_Employed],
      'ApplicantIncome': [data.ApplicantIncome],
      'CoapplicantIncome': [data.CoapplicantIncome],
      'LoanAmount': [data.LoanAmount],
      'Loan_Amount_Term': [data.Loan_Amount_Term],
      'Credit_History': [data.Credit_History],
      'Property_Area': [data.Property_Area]      
   }

   data_df = pd.DataFrame(data_df)

   # dados de treino
   data_train = pd.read_csv('https://drive.google.com/uc?id=1QsgW3apKJ8-PazRbQKTzkZrW76CZ--qQ&export=download')

   # dados de validação
   #data_val = pd.read_csv('https://drive.google.com/uc?id=18dY8nfISSjm0ODCDywqwGZxpj_YHl_vx&export=download')

   data_train_df = process_data(data_train)
   X_train, X_test, y_train, y_test = split_data(data_train_df)
   model = create_model(X_train, y_train)
   fit_model(model, X_train, y_train)
   result = evaluate_model(model, X_test, y_test)
   #print(result)

   #data_val_df = process_data(data_val)
   #pred_model_validation(model, data_val_df)

   data_df_process = process_data(data_df)
   loanStatusPredicted = pred_model_validation(model, data_df_process)

   return loanStatusPredicted
