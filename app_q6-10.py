import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from collections import Counter
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from openai import OpenAI

from simpsons_analysis import analyze_episode_summary, test_api_connection, analyze_episode
from summary_metrics import compare_summaries, analyze_convergence
from export import export_sentiment_analysis
from sentiment_visualization import main as sentiment_viz_main
from simpsons_sentiment_analysis import analyze_simpsons_sentiments, test_api_connection, export_to_csv
from sentiment_visualization import main as sentiment_viz_main

st.set_page_config(page_title="Gerador de Texto AI e Categorização de Manchetes", page_icon="🤖", layout="wide")

load_dotenv()

# variáveis de estado
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'episode_lines' not in st.session_state:
    st.session_state.episode_lines = None
if 'distribution' not in st.session_state:
    st.session_state.distribution = None
if 'num_calls' not in st.session_state:
    st.session_state.num_calls = 0

# Configura o cliente nvidia
client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

def read_project_info():
    try:
        with open('project_info.txt', 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return "Informações do projeto não encontradas."

# coletar manchetes
def get_headlines():
    url = "https://noticias.ufal.br/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    headlines = [h.text.strip() for h in soup.find_all('a', class_='titulo')]
    return headlines

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
    st.write("A funcionalidade de categorização está temporariamente indisponível.")
    
    if st.button("Coletar Manchetes"):
        with st.spinner("Coletando manchetes..."):
            headlines = get_headlines()
            st.subheader("Manchetes Coletadas:")
            for headline in headlines:
                st.write(headline)

# Título principal da aplicação
st.title("🤖 Gerador de Texto AI e Categorização de Manchetes & 🤖 Análise de Texto e Dados")

# Abas principais
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    "Apresentação", 
    "Gerador de Texto", 
    "Categorização de Manchetes", 
    "Análise The Simpsons", 
    "Análise de Sentimentos Simpsons",
    "Chat com Ollama",
    "Resumo do Episódio",
    "Resumo Detalhado do Episódio",
    "Comparação de Métricas",
    "Exportar Análise de Sentimento",
    "Visualização de Sentimentos"
])

with tab1:
    project_info = read_project_info()
    st.markdown(project_info)

with tab2:
    st.write("Gerador de Texto AI está temporariamente indisponível.")

with tab3:
    headline_categorization_app()

with tab4:
    st.write("Análise The Simpsons está temporariamente indisponível.")

# with tab5:
#     st.subheader("Análise de Sentimentos dos Simpsons")
    
#     if st.button("Testar Conexão com API"):
#         if test_api_connection():
#             st.success("Conexão com a API bem-sucedida!")
#         else:
#             st.error("Falha na conexão com a API. Verifique os logs para mais detalhes.")
    
#     if st.button("Iniciar Análise de Sentimentos"):
#         with st.spinner("Analisando sentimentos dos diálogos dos Simpsons..."):
#             progress_bar = st.progress(0)
            
#             def update_progress(progress):
#                 progress_bar.progress(progress)
            
#             episode_lines, distribution, accuracy, precision, num_calls = analyze_simpsons_sentiments(update_progress)
            
#             # remove a barra de progresso após a conclusão
#             progress_bar.empty()
        
#         st.success("Análise concluída!")
#         st.subheader("Número de chamadas ao LLM")
#         st.write(f"Foram necessárias {num_calls} chamadas ao LLM.")
#         st.subheader("Distribuição de Sentimentos")
#         fig, ax = plt.subplots()
#         distribution.plot(kind='bar', ax=ax)
#         plt.title("Distribuição de Sentimentos nas Falas")
#         plt.xlabel("Sentimento")
#         plt.ylabel("Proporção")
#         st.pyplot(fig)

#         st.subheader("Exemplo de Falas Classificadas")
#         if not episode_lines.empty and 'sentiment' in episode_lines.columns:
#             sample = episode_lines[['spoken_words', 'sentiment']].dropna().sample(10, random_state=42)
#             st.dataframe(sample.style.set_properties(**{'text-align': 'left'}))
#         else:
#             st.write("Não há dados de sentimento disponíveis.")


with tab5:
    st.subheader("Análise de Sentimentos dos Simpsons")
    
    if st.button("Testar Conexão com API"):
        if test_api_connection():
            st.success("Conexão com a API bem-sucedida!")
        else:
            st.error("Falha na conexão com a API. Verifique os logs para mais detalhes.")
    
    if not st.session_state.analysis_done:
        if st.button("Iniciar Análise de Sentimentos"):
            with st.spinner("Analisando sentimentos dos diálogos dos Simpsons..."):
                progress_bar = st.progress(0)
                
                def update_progress(progress):
                    progress_bar.progress(progress)
                
                st.session_state.episode_lines, st.session_state.distribution, _, _, st.session_state.num_calls = analyze_simpsons_sentiments(update_progress)
                
                progress_bar.empty()
            
            st.session_state.analysis_done = True

    if st.session_state.analysis_done:
        st.success("Análise concluída!")
        st.subheader("Número de chamadas ao LLM")
        st.write(f"Foram necessárias {st.session_state.num_calls} chamadas ao LLM.")
        
        st.subheader("Distribuição de Sentimentos")
        fig, ax = plt.subplots()
        st.session_state.distribution.plot(kind='bar', ax=ax)
        plt.title("Distribuição de Sentimentos nas Falas")
        plt.xlabel("Sentimento")
        plt.ylabel("Proporção")
        st.pyplot(fig)

        st.subheader("Exemplo de Falas Classificadas")
        if not st.session_state.episode_lines.empty and 'sentiment' in st.session_state.episode_lines.columns:
            sample_df = st.session_state.episode_lines[['spoken_words', 'sentiment']].dropna()
            if not sample_df.empty:
                sample_size = min(10, len(sample_df))
                sample = sample_df.sample(sample_size, random_state=42)
                st.dataframe(sample.style.set_properties(**{'text-align': 'left'}))
            else:
                st.write("Não há dados de sentimento disponíveis após a remoção de valores nulos.")
        else:
            st.write("Não há dados de sentimento disponíveis.")
        
        if st.button("Exportar Resultados para CSV"):
            csv_path = export_to_csv(st.session_state.episode_lines)
            st.success(f"Arquivo CSV salvo em: {csv_path}")
            
            with open(csv_path, "rb") as file:
                st.download_button(
                    label="Baixar arquivo CSV",
                    data=file,
                    file_name="simpsons_sentiment_analysis.csv",
                    mime="text/csv"
                )

with tab6:
    st.write("Chat com Ollama está temporariamente indisponível.")

with tab7:
    st.header("Resumo do Episódio dos Simpsons")
    
    if st.button("Testar Conexão com API", key="test_api_connection_button"):
        if test_api_connection():
            st.success("Conexão com a API bem-sucedida!")
        else:
            st.error("Falha na conexão com a API. Verifique os logs para mais detalhes.")
    
    if st.button("Analisar Episódio", key="analyze_episode_button"):
        with st.spinner("Gerando resumo do episódio... Isso pode levar alguns minutos."):
            summary, token_count = summarize_episode(92, 5)  # episódio 92 da temporada 5
        
        st.success("Resumo gerado com sucesso!")
        
        st.subheader(f"Resumo do Episódio 92 da Temporada 5")
        st.write(summary)
        st.write(f"Número de tokens no resumo: {token_count}")


with tab8:
    st.header("Resumo Detalhado do Episódio dos Simpsons")
    
    if st.button("Testar Conexão com API", key="test_api_connection_button_detailed"):
        if test_api_connection():
            st.success("Conexão com a API bem-sucedida!")
        else:
            st.error("Falha na conexão com a API. Verifique os logs para mais detalhes.")
    
    if st.button("Gerar Resumo Detalhado", key="analyze_episode_detailed_button"):
        with st.spinner("Gerando resumo detalhado do episódio... Isso pode levar alguns minutos."):
            final_summary, num_chunks, evaluation, chunk_summaries, chunk_evaluations = analyze_episode_summary(92, 5)
        
        st.success("Análise detalhada concluída!")
        
        st.subheader("Resumo Final do Episódio")
        st.write(final_summary)
        
        st.subheader("Detalhes da Análise")
        st.write(f"Número de chunks necessários: {num_chunks}")
        
        st.subheader("Avaliação do Resumo Final")
        st.write(evaluation)
        
        with st.expander("Ver Resumos e Avaliações dos Chunks"):
            for i, (summary, eval) in enumerate(zip(chunk_summaries, chunk_evaluations)):
                st.subheader(f"Chunk {i+1}")
                st.write("Resumo:")
                st.write(summary)
                st.write("Avaliação:")
                st.write(eval)

with tab9:
    st.header("Comparação de Métricas dos Resumos")
    
    if st.button("Gerar Comparação", key="compare_summaries_button"):
        with st.spinner("Gerando comparação dos resumos... Isso pode levar alguns minutos."):
            final_summary, num_chunks, _, chunk_summaries, _, reference_summary = analyze_episode_summary(92, 5)
            
            final_metrics, chunk_metrics = compare_summaries(reference_summary, final_summary, chunk_summaries)
            convergence_analysis, omitted_info = analyze_convergence(reference_summary, final_summary, chunk_summaries)
        
        st.success("Comparação concluída!")
        
        st.subheader("Métricas do Resumo Final")
        st.write(final_metrics)
        
        st.subheader("Métricas dos Chunks")
        for i, metrics in enumerate(chunk_metrics):
            st.write(f"Chunk {i+1}:")
            st.write(metrics)
        
        st.subheader("Análise de Convergência")
        st.write(convergence_analysis)
        
        st.subheader("Informações Omitidas")
        st.write(omitted_info)
        
        st.subheader("Resumos")
        st.write("Resumo de Referência (Exercício 7):")
        st.write(reference_summary)
        st.write("Resumo Final (Exercício 8):")
        st.write(final_summary)

with tab10:
    export_sentiment_analysis() 



with tab11:
    sentiment_viz_main()


    
# # Barra lateral
# st.sidebar.header("Sobre")
# st.sidebar.info(
#     "Esta aplicação realiza análises de texto e dados, incluindo geração de texto, "
#     "categorização de manchetes e análise de episódios de The Simpsons. "
#     "Algumas funcionalidades estão temporariamente indisponíveis."
# )

# # Footer
# st.sidebar.markdown("---")
# st.sidebar.markdown("Desenvolvido com ❤️ usando Streamlit")