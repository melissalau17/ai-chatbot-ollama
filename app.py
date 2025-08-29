import streamlit as st
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import ChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import subprocess
import time

# --- Ollama Setup ---
def start_ollama_and_pull_models():
    """Starts the Ollama server and pulls all required models."""
    st.write("Starting Ollama server...")
    
    # Use the absolute path to the ollama binary
    subprocess.Popen(['/usr/local/bin/ollama', 'serve'])
    time.sleep(5)  # Give the server time to start
    
    # Pull all models at once
    models_to_pull = ["phi3:mini", "deepseek-coder:1.3b"]
    st.write(f"Pulling models: {', '.join(models_to_pull)}...")
    for model in models_to_pull:
        try:
            # Use the absolute path for pulling the model too
            subprocess.run(['/usr/local/bin/ollama', 'pull', model], check=True)
            st.success(f"Model '{model}' pulled successfully.")
        except subprocess.CalledProcessError as e:
            st.error(f"Failed to pull model '{model}': {e}")


# ----------------- Streamlit UI and Logic -----------------
st.set_page_config(layout="wide")
st.title("My Local Chatbot")

if "ollama_started" not in st.session_state:
    with st.spinner("Setting up the local LLM server... this may take a moment."):
        start_ollama_and_pull_models()
    st.session_state.ollama_started = True

# --- Sidebar Inputs ---
st.sidebar.header("Settings")
model_options = ["phi3:mini", "deepseek-coder:1.3b"]
MODEL = st.sidebar.selectbox("Choose a Model", model_options)

MAX_HISTORY = st.sidebar.number_input("Max History", min_value=1, max_value=10, value=2, step=1)
# ... rest of your sidebar code ...
CONTEXT_SIZE = st.sidebar.number_input("Context Size", min_value=1024, max_value=16384, value=8192, step=1024)
TEMPERATURE = st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
TOP_P = st.sidebar.slider("Top-p (nucleus sampling)", 0.0, 1.0, 0.9, 0.05)
TOP_K = st.sidebar.slider("Top-k", 0, 100, 40, 5)
MAX_TOKENS = st.sidebar.number_input("Max Tokens", min_value=256, max_value=16384, value=2048, step=256)

# ... (rest of the code is the same) ...

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
if "summary" not in st.session_state:
    st.session_state.summary = ""

def clear_memory():
    chat_history = ChatMessageHistory()
    st.session_state.memory = ConversationBufferMemory(chat_memory=chat_history)
    st.session_state.chat_history = []
    st.session_state.summary = ""

if st.sidebar.button("Clear Conversation History"):
    clear_memory()

llm = ChatOllama(
    model=MODEL,
    streaming=True,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    num_predict=MAX_TOKENS,
    base_url="http://localhost:11434"
)

summary_prompt_template = PromptTemplate(
    input_variables=["chat_history"],
    template="You are a summarizer. Summarize the following conversation to preserve key information and context. \n\n{chat_history}"
)

summary_chain = summary_prompt_template | llm

def get_summary(chat_history_str):
    return summary_chain.invoke({"chat_history": chat_history_str})

def summarize_chat():
    if not st.session_state.chat_history:
        return "No chat history to summarize."
    return get_summary(st.session_state.chat_history)

if st.sidebar.button("Summarize Chat"):
    with st.sidebar:
        st.markdown("**Chat Summary:**")
        summary = summarize_chat()
        st.success(summary)

prompt_template = PromptTemplate(
    input_variables=["summary", "history", "human_input"],
    template="""You are a helpful assistant.
    Current conversation summary:
    {summary}
    Conversation history:
    {history}
    User: {human_input}
    Assistant:"""
)

chain = prompt_template | llm

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def trim_memory():
    if len(st.session_state.chat_history) > MAX_HISTORY * 2:
        history_to_summarize = st.session_state.chat_history[:(len(st.session_state.chat_history) - MAX_HISTORY * 2)]
        history_str = ""
        for msg in history_to_summarize:
            history_str += f"{msg['role']}: {msg['content']}\n"
        new_summary = get_summary(history_str)
        st.session_state.summary += "\n" + new_summary
        st.session_state.chat_history = st.session_state.chat_history[(len(st.session_state.chat_history) - MAX_HISTORY * 2):]

if prompt := st.chat_input("Say something"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    trim_memory()
    formatted_history = ""
    for msg in st.session_state.chat_history:
        formatted_history += f"{msg['role']}: {msg['content']}\n"
    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""
        for chunk in chain.stream({
                "human_input": prompt,
                "history": formatted_history,
                "summary": st.session_state.summary
            }):
            full_response += chunk.content
            response_container.markdown(full_response)
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})