
# Web Finance Simulator
import streamlit as st
from streamlit_option_menu import option_menu

# Imports
from datetime import datetime
import streamlit as st
import front_end.constantes as const
from front_end.financiamento import calc_financing
from front_end.analytics import display_all_results
from front_end.error import display_menssage
from back_end.data import ModelData

# Arquivo do modelo ML
from back_end.train import train_model

# Função de configuração da página
def page_config():
	st.set_page_config(page_title = 'Web Finance Simulator',
                    page_icon = '🚀',
                    layout = 'centered',
                    initial_sidebar_state = 'auto')

	st.title('Web Finance Simulator')
	const.spacer('title')
	const.spacer('title')
	st.sidebar.text('Web Finance Simulator: ')
	st.sidebar.text('Uma nova forma de simular')
	st.sidebar.text('o seu Financiamento')
	const.spacer('title', st.sidebar)

# Função para o rodapé
def page_footer():
	const.spacer('title', st.sidebar)
	const.spacer('title', st.sidebar)
	const.spacer('title', st.sidebar)
	st.sidebar.text(datetime.today().strftime('%B %d, %Y'))	
	st.sidebar.text('v1.0.0')
	st.sidebar.text('Projeto Integrado ML')

# Função main
def main():
	
	# Cria o modelo de dados
	data = ModelData()

	# Config
	page_config()

	# Executa os cálculos
	data = calc_financing(data)
	if data.simular_btn is not None:
		loanStatusPredicted = train_model(data)
		if (loanStatusPredicted == 1).all():
			data = display_all_results(data)
		else:
			data = display_menssage(data)
	# Rodapé
	page_footer()

# Executa a app
if __name__ == '__main__':
	main()
