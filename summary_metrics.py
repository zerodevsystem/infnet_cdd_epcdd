from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from googletrans import Translator
from deep_translator import GoogleTranslator

# Inicializar o tradutor e o Rouge
translator = Translator()
rouge = Rouge()

def translate_text(text, target_lang='en'):
    try:
        translator = GoogleTranslator(source='auto', target=target_lang)
        return translator.translate(text)
    except Exception as e:
        print(f"Erro na tradução: {e}")
        return text

def calculate_metrics(reference, hypothesis):
    # Calcular BLEU
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens)
    
    # Calcular ROUGE
    rouge_scores = rouge.get_scores(hypothesis, reference)[0]
    
    return {
        'bleu': bleu_score,
        'rouge-1': rouge_scores['rouge-1']['f'],
        'rouge-2': rouge_scores['rouge-2']['f'],
        'rouge-l': rouge_scores['rouge-l']['f']
    }

def compare_summaries(reference_summary, final_summary, chunk_summaries):
    final_metrics = calculate_metrics(reference_summary, final_summary)
    
    chunk_metrics = []
    for chunk_summary in chunk_summaries:
        chunk_metrics.append(calculate_metrics(reference_summary, chunk_summary))
    
    return final_metrics, chunk_metrics

def analyze_convergence(reference_summary, final_summary, chunk_summaries):
    final_metrics, chunk_metrics = compare_summaries(reference_summary, final_summary, chunk_summaries)
    
    # Analisar convergência
    convergence_analysis = "Os resumos parecem convergir em termos de conteúdo principal, mas há algumas diferenças notáveis:\n\n"
    
    # Comparar métricas do resumo final com a média das métricas dos chunks
    avg_chunk_metrics = {
        key: sum(chunk[key] for chunk in chunk_metrics) / len(chunk_metrics)
        for key in chunk_metrics[0]
    }
    
    for key in final_metrics:
        diff = final_metrics[key] - avg_chunk_metrics[key]
        convergence_analysis += f"- {key.upper()}: A diferença entre o resumo final e a média dos chunks é {diff:.4f}\n"
    
    # Analisar informações omitidas
    omitted_info = "Informações potencialmente omitidas entre os dois resumos:\n\n"
    
    # Aqui você pode implementar uma lógica mais sofisticada para detectar informações omitidas
    # Por exemplo, comparando entidades nomeadas ou frases-chave entre os resumos
    
    return convergence_analysis, omitted_info