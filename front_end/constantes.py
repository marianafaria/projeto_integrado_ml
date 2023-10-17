# Módulos com valores constantes

# Import
import streamlit as st

# Spacer
def spacer(space_type, location=st):
	if 'title' in space_type:
		location.title('')
	if 'head' in space_type:
		location.header('')
	if 'subheader' in space_type:
		location.subheader('')
	if 'write' in space_type:
		location.write('')
	if 'text' in space_type:
		location.text('')

# Linha
def line(location=st):
	location.write('_____')

# Financiamento
Gender = 'Gender'
Married = 'Married'
Dependents = 'Dependents'
Education = 'Education'
Self_Employed = 'Self_Employed'
Property_Area = 'Property_Area'
ApplicantIncome = 'ApplicantIncome'
CoapplicantIncome = 'CoapplicantIncome'
LoanAmount = 'LoanAmount'
Loan_Amount_Term = 'Loan_Amount_Term'
Credit_History = 'Credit_History'
simular_btn = 'simular_btn'
