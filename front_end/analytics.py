# Módulo para os resultados

# Imports
import pandas as pd
import streamlit as st
import front_end.constantes as const

# Cálculo de métricas
def calc_display_basic_metrics(data, location=st):
	
	const.spacer('title')


# Mostra os resultados
def display_all_results(data, location=st):
	location.title('Análises 📊')

	calc_display_basic_metrics(data, location)

	return data
