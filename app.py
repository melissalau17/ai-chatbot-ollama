import streamlit as st
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import ChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import os
import subprocess
from pyngrok import ngrok

# ----------------- Start-up Functions (Run only once) -----------------
# We use a simple environment variable here instead of google.colab.userdata
# For security, store your NGROK_AUTHTOKEN in your OS environment.
NGROK_AUTHTOKEN = os.environ.get("NGROK_AUTHTOKEN")

def start_ollama_and_ngrok():
    """Starts the Ollama server and sets up the ngrok tunnel."""
    if not NGROK_AUTHTOKEN:
        st.error("NGROK_AUTHTOKEN is not set as an environment variable.")
        st.stop()

    try:
        # Start Ollama server
        st.write("Starting Ollama server...")
        subprocess.Popen(['ollama', 'serve'])

        # Set up ngrok tunnel to Ollama
        st.write("Setting up ngrok tunnel...")
        ngrok.set_auth_token(NGROK_AUTHTOKEN)
        NGROK_PORT = '11434'
        tunnel = ngrok.connect(NGROK_PORT, host_header=f'localhost:{NGROK_PORT}')
        st.session_state.ollama_url = tunnel.public_url
        st.success(f"Ollama server is live at: {st.session_state.ollama_url}")

    except Exception as e:
        st.error(f"Failed to start Ollama or ngrok: {e}")
        st.stop()


# ----------------- Streamlit UI and Logic -----------------
st.set_page_config(layout="wide")
st.title("My Local Chatbot")

# --- ONE-TIME SETUP ---
# Use session state to run this block only once when the app first loads
if "ollama_url" not in st.session_state:
    with st.spinner("Setting up the local LLM server... this may take a moment."):
        start_ollama_and_ngrok()

# Sidebar Inputs
st.sidebar.header("Settings")
model_options = ["llama3:8b", "deepseek-r1:1.5b"]
MODEL = st.sidebar.selectbox("Choose a Model", model_options, index=0)

# Inputs for history + context size
MAX_HISTORY = st.sidebar.number_input("Max History", min_value=1, max_value=10, value=2, step=1)
# ... rest of your sidebar code ...
CONTEXT_SIZE = st.sidebar.number_input("Context Size", min_value=1024, max_value=16384, value=8192, step=1024)
TEMPERATURE = st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
TOP_P = st.sidebar.slider("Top-p (nucleus sampling)", 0.0, 1.0, 0.9, 0.05)
TOP_K = st.sidebar.slider("Top-k", 0, 100, 40, 5)
MAX_TOKENS = st.sidebar.number_input("Max Tokens", min_value=256, max_value=16384, value=2048, step=256)

def clear_memory():
    chat_history = ChatMessageHistory()
    st.session_state.memory = ConversationBufferMemory(chat_memory=chat_history)
    st.session_state.chat_history = []
    st.session_state.summary = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
if "summary" not in st.session_state:
    st.session_state.summary = ""

if st.sidebar.button("Clear Conversation History"):
    clear_memory()

# ----------------- LLM Instantiation (Corrected) -----------------
# The `llm` variable is now defined correctly and only once per app session
llm = ChatOllama(
    model=MODEL,
    streaming=True,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    num_predict=MAX_TOKENS,
    base_url=st.session_state.ollama_url
)

# ... The rest of your app logic is now correct and can be placed here ...
# Prompt Template for summarization
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

# Display Chat History 
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