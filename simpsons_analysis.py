import pandas as pd
import tiktoken
import streamlit as st
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv


from openai import OpenAI
from summary_metrics import compare_summaries, analyze_convergence

# Carrega as variáveis de ambiente
load_dotenv(override=True)

# Função para obter variáveis de ambiente com validação
def get_env(key, default=None, required=False):
    value = os.getenv(key, default)
    if required and value is None:
        raise ValueError(f"A variável de ambiente '{key}' é obrigatória, mas não foi definida.")
    return value.strip() if isinstance(value, str) else value

# Configurações da API
OPENAI_BASE_URL = get_env('OPENAI_BASE_URL', required=True)
OPENAI_API_KEY = get_env('OPENAI_API_KEY', required=True)
OPENAI_MODEL = get_env('OPENAI_MODEL', required=True)
TEMPERATURE = float(get_env('TEMPERATURE', '0.5'))
TOP_P = float(get_env('TOP_P', '1.0'))
MAX_TOKENS = int(get_env('MAX_TOKENS', '1024'))

# Inicialização do cliente OpenAI
client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY
)

def load_simpsons_data():
    episodes = pd.read_csv('data/simpsons_episodes.csv', dtype={'id': 'int32', 'season': 'int32'})
    script_lines = pd.read_csv('data/simpsons_script_lines.csv', low_memory=False, dtype={'episode_id': 'int32'})
    combined_data = pd.merge(episodes, script_lines, left_on='id', right_on='episode_id')
    return combined_data

def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    if pd.isna(text):
        return 0
    return len(encoding.encode(str(text)))

def generate_text(prompt):
    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating text: {str(e)}")
        return "Erro ao gerar análise."

def analyze_simpsons_data():
    st.header("Análise dos Episódios de The Simpsons")
    
    data = load_simpsons_data()
    
    with st.expander("Análise de Tokens"):
        st.subheader("Análise de Tokens")
        data['tokens'] = data['spoken_words'].apply(count_tokens)
        
        # por episódio
        avg_tokens_per_episode = data.groupby('episode_id')['tokens'].sum().mean()
        max_tokens_episode = data.groupby('episode_id')['tokens'].sum().idxmax()
        max_tokens_episode_count = data.groupby('episode_id')['tokens'].sum().max()
        
        # por temporada
        season_data = data.groupby('season')['tokens'].sum()
        avg_tokens_per_season = season_data.mean()
        max_tokens_season = season_data.idxmax()
        max_tokens_season_count = season_data.max()
        
        st.write(f"Média de tokens por episódio: {avg_tokens_per_episode:.2f}")
        st.write(f"Média de tokens por temporada: {avg_tokens_per_season:.2f}")
        st.write(f"Episódio com mais tokens: {max_tokens_episode} ({max_tokens_episode_count} tokens)")
        st.write(f"Temporada com mais tokens: {max_tokens_season} ({max_tokens_season_count} tokens)")
    
    with st.expander("Gráfico de Tokens por Temporada"):
        fig, ax = plt.subplots(figsize=(12, 6))
        season_data.plot(kind='bar', ax=ax)
        ax.set_title('Tokens por Temporada')
        ax.set_xlabel('Temporada')
        ax.set_ylabel('Número de Tokens')
        st.pyplot(fig)
    
    with st.expander("IMDB e Audiência"):
        st.subheader("Análise de Avaliações IMDB e Audiência")
        
        # prompt Chaining para análise do IMDB
        imdb_prompt = f"""
        Analise as seguintes estatísticas de avaliações IMDB dos episódios de The Simpsons:
        
        Média: {data['imdb_rating'].mean():.2f}
        Mediana: {data['imdb_rating'].median():.2f}
        Mínimo: {data['imdb_rating'].min():.2f}
        Máximo: {data['imdb_rating'].max():.2f}
        Desvio Padrão: {data['imdb_rating'].std():.2f}
        
        Forneça uma análise descritiva dessas estatísticas em 3-4 frases.
        """
        
        imdb_analysis = generate_text(imdb_prompt)
        st.write("Análise das Avaliações IMDB:")
        st.write(imdb_analysis)
        
        # prompt Chaining para análise da audiência
        audience_prompt = f"""
        Analise as seguintes estatísticas de audiência dos episódios de The Simpsons:
        
        Média: {data['us_viewers_in_millions'].mean():.2f} milhões
        Mediana: {data['us_viewers_in_millions'].median():.2f} milhões
        Mínimo: {data['us_viewers_in_millions'].min():.2f} milhões
        Máximo: {data['us_viewers_in_millions'].max():.2f} milhões
        Desvio Padrão: {data['us_viewers_in_millions'].std():.2f} milhões
        
        Forneça uma análise descritiva dessas estatísticas em 3-4 frases.
        """
        
        audience_analysis = generate_text(audience_prompt)
        st.write("Análise da Audiência:")
        st.write(audience_analysis)

        

def summarize_episode(episode_id, season):
    data = load_simpsons_data()
    episode_data = data[(data['episode_id'] == episode_id) & (data['season'] == season)]
    
    if episode_data.empty:
        return "Episódio não encontrado.", 0
    
    episode_lines = episode_data['spoken_words'].dropna().tolist()
    episode_text = " ".join(episode_lines)
    
    prompt = f"""
    Leia as seguintes falas do episódio número {episode_id} da temporada {season} de The Simpsons e faça um resumo de aproximadamente 500 tokens, explicando o que acontece e como termina o episódio:

    {episode_text}

    Resumo (aproximadamente 500 tokens):
    """
    
    summary = generate_text(prompt)
    
    # Contar tokens do resumo
    token_count = count_tokens(summary)
    
    return summary, token_count

def analyze_episode(episode_id, season):
    summary, token_count = summarize_episode(episode_id, season)
    return summary, token_count

# Função para testar a conexão com a API
def test_api_connection():
    try:
        print(f"Attempting to connect to {OPENAI_BASE_URL}")
        print(f"Using model: {OPENAI_MODEL}")
        print(f"API Key (primeiros 5 caracteres): {OPENAI_API_KEY[:5]}...")
        
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": "Hello"}],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS
        )
        
        print("API connection successful!")
        print(f"Response: {completion.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"API connection failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {e.args}")
        return False

def create_chunks(lines, chunk_size=100, overlap=25):
    chunks = []
    for i in range(0, len(lines), chunk_size - overlap):
        chunk = lines[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

def summarize_chunk(chunk):
    prompt = f"""
    Resuma o seguinte trecho de diálogo do episódio dos Simpsons em aproximadamente 100 palavras:

    {' '.join(chunk)}

    Resumo:
    """
    return generate_text(prompt)

def summarize_episode_chunks(episode_id, season):
    data = load_simpsons_data()
    episode_lines = data[(data['episode_id'] == episode_id) & (data['season'] == season)]['spoken_words'].dropna().tolist()
    
    chunks = create_chunks(episode_lines)
    chunk_summaries = [summarize_chunk(chunk) for chunk in chunks]
    
    final_prompt = f"""
    Com base nos seguintes resumos de partes do episódio {episode_id} da temporada {season} dos Simpsons, crie um resumo final coerente de aproximadamente 500 palavras:

    {' '.join(chunk_summaries)}

    Resumo final:
    """
    
    final_summary = generate_text(final_prompt)
    
    return final_summary, len(chunks), chunk_summaries

def analyze_episode_summary(episode_id, season):
    final_summary, num_chunks, chunk_summaries = summarize_episode_chunks(episode_id, season)
    
    evaluation_prompt = f"""
    Avalie o seguinte resumo do episódio {episode_id} da temporada {season} dos Simpsons quanto à veracidade e coerência:

    {final_summary}

    Forneça uma análise detalhada sobre a qualidade do resumo, sua fidelidade ao conteúdo original e sua coerência narrativa.
    """
    
    evaluation = generate_text(evaluation_prompt)
    
    chunk_evaluations = []
    for i, chunk_summary in enumerate(chunk_summaries):
        chunk_eval_prompt = f"""
        Avalie o seguinte resumo de uma parte do episódio {episode_id} da temporada {season} dos Simpsons quanto à veracidade e coerência:

        {chunk_summary}

        Forneça uma breve análise sobre a qualidade deste resumo parcial.
        """
        chunk_evaluations.append(generate_text(chunk_eval_prompt))
    
    reference_summary = analyze_episode(episode_id, season)[0]  # Assumindo que esta função retorna o resumo simples
    
    return final_summary, num_chunks, evaluation, chunk_summaries, chunk_evaluations, reference_summary






if __name__ == "__main__":
    test_api_connection()