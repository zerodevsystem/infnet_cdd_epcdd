import os
from flask import Flask, request, Response, jsonify
from openai import OpenAI
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

app = Flask(__name__)

# Configura o cliente OpenAI com as variáveis de ambiente
client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "$API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC")
)

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    prompt = data.get('prompt', "Write a limerick about the wonders of GPU computing.")
    
    def generate():
        try:
            completion = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct"),
                messages=[{"role": "user", "content": prompt}],
                temperature=float(os.getenv("TEMPERATURE", 0.5)),
                top_p=float(os.getenv("TOP_P", 1)),
                max_tokens=int(os.getenv("MAX_TOKENS", 1024)),
                stream=True
            )

            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            yield f"Error: {str(e)}"

    return Response(generate(), mimetype='text/plain')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=11434)