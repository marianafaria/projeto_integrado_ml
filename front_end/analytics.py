# Módulo para os resultados

# Imports
import pandas as pd
import streamlit as st
import front_end.constantes as const

# Cálculo de métricas
def calc_display_basic_metrics(data, location=st):
	
	const.spacer('title')
        location.subheader(f'Valor do Empréstimo: R$ {data.LoanAmount}')
	location.subheader(f'Prazo do Empréstimo: {data.Loan_Amount_Term} Meses')
	location.subheader(f'Valor da Parcela : R$ {round(npf.pmt(0.1275/12,data.Loan_Amount_Term,-data.LoanAmount,0),2)}')
	location.subheader(f'Taxa de juros: {12.75} Ao Ano')

# Mostra os resultados
def display_all_results(data, location=st):
	location.title('Análises 📊')

	calc_display_basic_metrics(data, location)

	return data
