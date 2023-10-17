# Módulo para os resultados

# Imports
import pandas as pd
import streamlit as st
import front_end.constantes as const

# Cálculo de métricas
def calc_display_mensage(data, location=st):
	
	const.spacer('title')
	print(data)
	
	location.subheader('Prezado usuário, agradecemos por fornecer os seus dados para a solicitação de financiamento. Após uma análise minuciosa, lamentamos informar que, com as informações fornecidas, o financiamento não será aprovado neste momento.')


# Mostra os resultados
def display_menssage(data, location=st):
	location.title('Retorno 📋')

	calc_display_mensage(data, location)

	return data
