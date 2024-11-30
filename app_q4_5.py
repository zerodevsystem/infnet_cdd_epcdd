import os
import streamlit as st
import requests
import json
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from collections import Counter
from dotenv import load_dotenv
import pandas as pd
import tiktoken
import numpy as np

# Configuração da página (deve ser a primeira chamada Streamlit)
st.set_page_config(page_title="Gerador de Texto AI e Categorização de Manchetes", page_icon="🤖", layout="wide")

# Agora, importe as funções após a configuração da página
from simpsons_analysis import analyze_simpsons_data
from simpsons_sentiment_analysis import analyze_simpsons_sentiments
from ollama_chat import ollama_chat 

# Carregar variáveis de ambiente
load_dotenv()

# Função para ler informações do projeto
def read_project_info():
    try:
        with open('project_info.txt', 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return "Informações do projeto não encontradas."

# Função para obter os modelos disponíveis localmente
def get_local_models():
    response = requests.get("http://localhost:11434/api/tags")
    models = []
    if response.status_code == 200:
        data = response.json()
        models = [model['name'] for model in data.get('models', [])]
    return models

# Função de chat com o modelo selecionado
def ollama_chat():
    st.title("Chat com LLM (Modelo Selecionado)")
    st.write("Interaja com a LLM servida pelo Ollama localmente na porta 11434.")

    # Obter modelos disponíveis
    models = get_local_models()
    
    # Selecionar modelo através de um menu dropdown
    selected_model = st.selectbox("Selecione o modelo:", models)
    
    # Sessão para manter o histórico do chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Campo de entrada do usuário
    user_input = st.text_input("Digite sua mensagem", placeholder="Escreva aqui...")

    if st.button("Enviar"):
        if user_input.strip():
            # Adicionar a entrada do usuário no histórico
            st.session_state.chat_history.append({"role": "user", "message": user_input})
            
            # Enviar a mensagem para o Ollama
            try:
                url = "http://localhost:11434/api/chat"
                headers = {"Content-Type": "application/json"}
                data = {
                    "model": selected_model,  # Usar o modelo selecionado
                    "messages": [{"role": "user", "content": user_input}],
                    "stream": False
                }
                
                with st.spinner("Gerando resposta..."):
                    response = requests.post(url, headers=headers, json=data)
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        llm_response = response_data.get('message', {}).get('content', 'Sem resposta da LLM.')
                        
                        # Adicionar a resposta no histórico
                        st.session_state.chat_history.append({"role": "assistant", "message": llm_response})
                        
                        # Exibir a resposta
                        st.markdown(f"**LLM:** {llm_response}")
                    else:
                        st.error(f"Erro na requisição: {response.status_code}")
                        st.error(response.text)
            except Exception as e:
                st.error(f"Erro ao se comunicar com o Ollama: {e}")
        else:
            st.warning("Digite uma mensagem antes de enviar.")

    # Exibir histórico do chat
    for entry in st.session_state.chat_history:
        if entry["role"] == "user":
            st.markdown(f"**Você:** {entry['message']}")
        elif entry["role"] == "assistant":
            st.markdown(f"**LLM:** {entry['message']}")

# Função para gerador de texto AI
def text_generation_app():
    st.header("Gerador de Texto AI")
    
    prompt = st.text_area("Digite seu prompt aqui:", height=100)

    if st.button("Gerar Texto"):
        if prompt:
            url = "http://localhost:11434/api/generate"
            headers = {"Content-Type": "application/json"}
            data = {"model": "llama3.2", "prompt": prompt, "stream": False}
            
            with st.spinner("Gerando texto..."):
                try:
                    response = requests.post(url, headers=headers, json=data)
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        generated_text = response_data.get('response', 'Erro na resposta da LLM.')
                        st.markdown(generated_text)
                    else:
                        st.error(f"Erro na requisição: {response.status_code}")
                        st.error(response.text)
                except requests.exceptions.RequestException as e:
                    st.error(f"Erro ao fazer a requisição: {e}")
        else:
            st.warning("Por favor, digite um prompt antes de gerar o texto.")

# Função para coletar manchetes
def get_headlines():
    url = "https://noticias.ufal.br/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    headlines = [h.text.strip() for h in soup.find_all('a', class_='titulo')]
    return headlines

# Função para categorizar manchetes
def categorize_headlines(headlines):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    
    prompt = """Categorize as seguintes manchetes como positiva, neutra ou negativa:

Exemplos:
1. "Estudantes da UFAL ganham prêmio nacional" - Positiva
2. "Universidade anuncia novos cursos para o próximo semestre" - Positiva
3. "Greve dos professores chega ao fim após acordo" - Neutra
4. "Aulas suspensas devido a problemas de infraestrutura" - Negativa

Agora, categorize estas manchetes:

"""
    prompt += "\n".join(headlines)

    data = {"model": "llama3.2", "prompt": prompt, "stream": False}
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        categorized = []
        for line in response.text.split('\n'):
            if ' - ' in line:
                headline, category = line.rsplit(' - ', 1)
                categorized.append({"headline": headline.strip(), "category": category.strip()})
        return categorized
    else:
        st.error(f"Erro na categorização: {response.status_code}")
        return []

# Função para criar gráfico
def create_chart(categorized):
    categories = [item['category'] for item in categorized]
    count = Counter(categories)
    
    fig, ax = plt.subplots()
    ax.bar(count.keys(), count.values())
    ax.set_title('Categorização das Manchetes da UFAL')
    ax.set_xlabel('Categorias')
    ax.set_ylabel('Quantidade de Manchetes')
    
    return fig, count

# Função para a aplicação de categorização de manchetes
def headline_categorization_app():
    st.header("Categorização de Manchetes da UFAL")
    
    if st.button("Coletar e Categorizar Manchetes"):
        with st.spinner("Coletando e categorizando manchetes..."):
            headlines = get_headlines()
            categorized = categorize_headlines(headlines)
            
            if categorized:
                fig, count = create_chart(categorized)
                st.pyplot(fig)
                
                st.subheader("Resultados da Categorização:")
                st.json(count)
                
                st.subheader("Manchetes Categorizadas:")
                st.json(categorized)
            else:
                st.error("Não foi possível categorizar as manchetes.")

# Título principal da aplicação
st.title("🤖 Gerador de Texto AI e Categorização de Manchetes & 🤖 Análise de Texto e Dados")

# Abas principais
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Apresentação", 
    "Gerador de Texto", 
    "Categorização de Manchetes", 
    "Análise The Simpsons", 
    "Análise de Sentimentos Simpsons",
    "Chat com Ollama"
])

with tab1:
    project_info = read_project_info()
    st.markdown(project_info)

with tab2:
    #text_generation_app()
    pass

with tab3:
    #headline_categorization_app()
    pass

with tab4:
    analyze_simpsons_data()
    pass

with tab5:
    st.subheader("Análise de Sentimentos dos Simpsons")
    distribution, accuracy, precision = analyze_simpsons_sentiments()
    st.write("Distribuição de Sentimentos:", distribution)
    st.write(f"Acurácia do Modelo: {accuracy:.2f}")
    st.write("Precisão por Classe:", precision)
    pass

with tab6:
    # ollama_chat() # desativado porqeue executo localmente no samsung_book
    pass

# # Barra lateral
# st.sidebar.header("Sobre")
# st.sidebar.info(
#     "Esta aplicação realiza análises de texto e dados, incluindo geração de texto, "
#     "categorização de manchetes e análise de episódios de The Simpsons."
# )

# # Footer
# st.sidebar.markdown("---")
# st.sidebar.markdown("Desenvolvido com ❤️ usando Streamlit")