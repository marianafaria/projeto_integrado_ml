# Módulo para os resultados

# Imports
import pandas as pd
import streamlit as st
import front_end.constantes as const
import numpy_financial as npf
import locale

# Cálculo de métricas
def calc_display_basic_metrics(data, location=st):
	locale.setlocale(locale.LC_ALL, 'pt_BR.utf-8')
	const.spacer('title')
	formatted_amount = locale.format_string("%.2f", data.LoanAmount, grouping=True)
	location.subheader(f'Valor do Empréstimo: R$ {formatted_amount}')
	location.subheader(f'Prazo do Empréstimo: {data.Loan_Amount_Term} meses')
	formatted_parc = locale.format_string("%.2f", round(npf.pmt(0.1275/12,data.Loan_Amount_Term,-data.LoanAmount,0),2), grouping=True)
	location.subheader(f'Valor da Parcela : R$ {formatted_parc}')
	location.subheader(f'Taxa de juros: {12.75}% ao ano')


# Mostra os resultados
def display_all_results(data, location=st):
	location.title('Análises 📊')

	calc_display_basic_metrics(data, location)

	return data
