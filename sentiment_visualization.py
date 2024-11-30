import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data
def read_csv(file):
    """Lê o arquivo CSV uploaded e armazena em cache."""
    return pd.read_csv(file)

def process_data(df, episode=None):
    """Processa os dados, contando os sentimentos, opcionalmente filtrando por episódio."""
    if episode:
        df = df[df['episode_id'] == episode]
    return df['sentiment'].value_counts()

def create_pie_chart(data, episode=None):
    """Cria um gráfico de pizza com os dados processados."""
    title = f"Proporção de Falas por Categoria de Sentimento"
    if episode:
        title += f" (Episódio {episode})"
    
    fig = px.pie(
        values=data.values,
        names=data.index,
        title=title,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def main():
    st.title("Visualização da Análise de Sentimento")

    uploaded_file = st.file_uploader("Escolha o arquivo CSV da análise de sentimento", type="csv")

    if uploaded_file is not None:
        try:
            df = read_csv(uploaded_file)
            
            # Verifica se há sentimentos não nulos
            df_with_sentiment = df[df['sentiment'].notna()]
            
            if df_with_sentiment.empty:
                st.warning("Não há dados de sentimento disponíveis no arquivo CSV.")
            else:
                # Adiciona seleção de episódio
                episodes = df['episode_id'].unique()
                selected_episode = st.selectbox("Selecione um episódio (opcional)", ["Todos"] + list(episodes))
                
                episode = selected_episode if selected_episode != "Todos" else None
                sentiment_counts = process_data(df_with_sentiment, episode)
                
                if not sentiment_counts.empty:
                    fig = create_pie_chart(sentiment_counts, episode)
                    st.plotly_chart(fig)
                else:
                    st.warning("Não há dados de sentimento para o episódio selecionado.")

            # Exibe dados brutos
            if st.checkbox("Mostrar dados brutos"):
                st.write(df)

        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")
    else:
        st.info("Por favor, faça o upload de um arquivo CSV para visualizar o gráfico.")

if __name__ == "__main__":
    main()