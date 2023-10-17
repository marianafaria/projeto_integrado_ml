# Módulo de cálculo do financiamento

# Imports
import random
import numpy_financial as npf
import pandas as pd
import streamlit as st
import front_end.constantes as const

# Obtém os inputs
def get_financing_inputs(data, location=st):

	Gender = location.selectbox('Gênero', ['Feminino', 'Masculino'])
	data.Gender = Gender

	Married = location.selectbox('Você é Casado(a)?', ['Sim', 'Não'])
	data.Married = Married

	data.Dependents = location.number_input('Possui dependentes?',
		min_value = 0,
		max_value = 3,
		value = 1,
		step = 1,
		key = 'Dependents')

	Education = location.selectbox('Possui Graduação?', ['Sim', 'Não'])
	data.Education = Education

	Self_Employed = location.selectbox('Está empregado?', ['Sim', 'Não'])
	data.Self_Employed = Self_Employed

	Property_Area = location.selectbox('Qual o tipo da residência', ['Urbano', 'Semi-Urbano', 'Rural'])
	data.Property_Area = Property_Area

	data.ApplicantIncome = location.number_input('Renda do Candidato ($)',
		min_value = 1,
		max_value = 100_000_000_000,
		value = 1,
		step = 100,
		key = 'ApplicantIncome')

	data.CoapplicantIncome = location.number_input('Renda do co-requerente ($)',
		min_value = 0,
		max_value = 100_000_000_000,
		value = 0,
		step = 100,
		key = 'CoapplicantIncome')

	data.LoanAmount = location.number_input('Valor do Empréstimo ($)',
		min_value = 1,
		max_value = 100_000_000_000_000,
		value = 1,
		step = 100,
		key = 'LoanAmount')

	data.Loan_Amount_Term = location.number_input('Prazo do Empréstimo (em meses)',
		min_value = 1,
		max_value = 120_000_000,
		value = 360,
		step = 1,
		key = 'Loan_Amount_Term')

	data.Credit_History = random.choice([0, 1])
	data.simular_btn = None

	botao_clicado = False

	if location.button('Simular'):
		botao_clicado = True

	# Se o botão foi clicado, capture o valor do campo e adicione ao objeto "data"
	if botao_clicado:
		data.simular_btn = botao_clicado

	return data


# Função de cálculo do financiamento
def calc_financing(data, location=st):
	location.header('Financiamento 🏦')
	data = get_financing_inputs(data, location)
	const.line(location)
	return data
