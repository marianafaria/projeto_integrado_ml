# Back-end

# Imports
import numpy as np
import streamlit as st

# Classe para o modelo de dados
class ModelData:
    # Construtor
	def __init__(self):

		self.model_dict = {}

		# Dados dos inputs para o modelo
		self.Gender = 'Gender'
		self.Married = 'Married'
		self.Dependents = 'Dependents'
		self.Education = 'Education'
		self.Self_Employed = 'Self_Employed'
		self.Property_Area = 'Property_Area'
		self.ApplicantIncome = 'ApplicantIncome'
		self.CoapplicantIncome = 'CoapplicantIncome'
		self.LoanAmount = 'LoanAmount'
		self.Loan_Amount_Term = 'Loan_Amount_Term'
		self.Credit_History = 'Credit_History'
		self.simular_btn = 'simular_btn'


	# Método para space
	def spacer(self, space_type, location=st):
		if 'title' in space_type:
			location.title('')
		if 'head' in space_type:
			location.header('')
		if 'subheader' in space_type:
			location.subheader('')

	# Método para linha
	def line(self, location=st):
		location.write('_____')
