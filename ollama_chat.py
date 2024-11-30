import requests
import streamlit as st
import json

def ollama_chat():
    st.title("Chat com LLM (Modelo Selecionado)")
    st.write("Interaja com a LLM servida pelo Ollama localmente na porta 11434.")

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
                url = "http://localhost:11434/api/generate"  # Altere para o endpoint correto
                headers = {"Content-Type": "application/json"}
                data = {
                    "model": "llama3.2",  # Nome do modelo
                    "prompt": user_input,
                    "stream": False
                }
                
                with st.spinner("Gerando resposta..."):
                    response = requests.post(url, headers=headers, json=data)
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        llm_response = response_data.get('response', 'Sem resposta da LLM.')

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

# Adicione esta linha no final do seu script principal
if __name__ == "__main__":
    ollama_chat()