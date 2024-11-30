import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import json
from openai import OpenAI

import json
import logging

logging.basicConfig(level=logging.DEBUG)

load_dotenv()

OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL')
TEMPERATURE = float(os.getenv('TEMPERATURE'))
TOP_P = float(os.getenv('TOP_P'))
MAX_TOKENS = int(os.getenv('MAX_TOKENS'))

client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY
)

def load_simpsons_data():
    episodes = pd.read_csv('data/simpsons_episodes.csv')
    script_lines = pd.read_csv('data/simpsons_script_lines.csv', low_memory=False)
    combined_data = pd.merge(script_lines, episodes[['id', 'season']], left_on='episode_id', right_on='id', how='left')
    return combined_data


def classify_sentiment(texts, examples):
    prompt = f"""
    ### Instructions:
    You are an expert in human communication and marketing, specialized in sentiment analysis.
    You have to classify lines from the Simpsons show as negative, neutral and positive as defined below:

    - positive: happy, constructive, hopeful, joy and similar lines.
    - negative: sad, destructive, hopeless, aggressive and similar lines.
    - neutral: indifferent, objective, formal and lines classified neither as positive or negative.

    ### Examples:
    {examples}

    Given this information, respond in JSON with the classification of these other lines as positive,
    negative or neutral. The response should contain only the json with the classification, without any
    additional information.

    ### Lines:
    {texts}
    """
    
    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS
        )
        logging.debug(f"API Response: {completion.choices[0].message.content}")
        
        # remove crases e espaços em branco no início e no final
        content = completion.choices[0].message.content.strip().strip('`')
        
        # resposta como JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError as json_err:
            logging.error(f"JSON Decode Error: {json_err}")
            logging.error(f"Cleaned content: {content}")
            
            return eval(content)
    
    except Exception as e:
        logging.error(f"Error calling API: {str(e)}")
        return {}

def analyze_simpsons_sentiments(update_progress=None):
    data = load_simpsons_data()
    episode_lines = data[(data['episode_id'] == 92) & (data['season'] == 5)].copy()
    
    examples = """
    Positive:
    - "Woo-hoo! Life is like a box of doughnuts - sweet and full of surprises!"
    - "I'm king of the world! Top of the food chain, baby! No one can stop the Homer!"
    - "Hey Bart, my boy! You're looking as mischievous as ever. Ready to cause some trouble?"
    - "Mom, we don't need to pay for school. Springfield Elementary is free and full of adventure!"
    - "Ooh, a mystery box! This is gonna be more fun than a barrel of monkeys!"

    Neutral:
    - "Has anyone seen Maggie's pacifier?"
    - "Do we have to attend the school play?"
    - "Dad, can we please go to Itchy & Scratchy Land? I promise I'll be good forever!"
    - "The nuclear plant inspection is tomorrow, Homer."
    - "Behold, my latest invention: the Flanders-Be-Gone spray!"

    Negative:
    - "D'oh! I've reached rock bottom, and I'm still digging."
    - "Marge, our marriage is crumbling faster than the walls of this house. How did we end up here?"
    - "Poor Stampy the elephant. He met his end trying to pick up a peanut with his trunk."
    - "Bart, how could you sell your soul? That's the worst thing you've ever done!"
    - "Not so fast, Simpson. Your reign of terror over the power plant ends now."
    """
    
    batch_size = 20
    lines = episode_lines['spoken_words'].dropna().tolist()
    batches = [lines[i:i + batch_size] for i in range(0, len(lines), batch_size)]
    
    results = {}
    num_calls = 0
    
    for i, batch in enumerate(batches):
        text = '\n'.join(batch)
        batch_results = classify_sentiment(text, examples)
        if batch_results:
            if isinstance(batch_results, dict):
                results.update(batch_results)
            elif isinstance(batch_results, list):
                for item in batch_results:
                    if 'line' in item and ('classification' in item or 'sentiment' in item):
                        key = item.get('line') or item.get('Line')
                        value = item.get('classification') or item.get('sentiment') or item.get('Classification') or item.get('Sentiment')
                        results[key] = value
        num_calls += 1
        
        if update_progress:
            update_progress((i + 1) / len(batches))
    
    # novo DataFrame ds frases classificadas
    classified_lines = pd.DataFrame(list(results.items()), columns=['spoken_words', 'sentiment'])
    
    # join com o DataFrame original
    classified_episode_lines = pd.merge(episode_lines, classified_lines, on='spoken_words', how='left')
    
    distribution = classified_episode_lines['sentiment'].value_counts(normalize=True)
    
    return classified_episode_lines, distribution, None, None, num_calls
    
def test_api_connection():
    try:
        print(f"Attempting to connect to {OPENAI_BASE_URL}")
        print(f"Using model: {OPENAI_MODEL}")
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


def export_to_csv(df, filename="simpsons_sentiment_analysis.csv"):
    """
    Exporta o DataFrame para um arquivo CSV.
    
    :param df: DataFrame para exportar
    :param filename: Nome do arquivo CSV (padrão: "simpsons_sentiment_analysis.csv")
    :return: O caminho do arquivo CSV salvo
    """
    csv_path = os.path.join(os.getcwd(), filename)
    df.to_csv(csv_path, index=False)
    return csv_path


if __name__ == "__main__":
    test_api_connection()