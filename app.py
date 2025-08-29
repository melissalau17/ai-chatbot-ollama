import streamlit as st
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import ChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import os

# --- Model Definitions ---
# A dictionary mapping display names to model file paths and types
MODEL_MAP = {
    "Llama-2 (7B)": {
        "path": "/app/models/llama-2-7b-chat.gguf",
        "type": "llama"
    },
    "Deepseek-Coder (1.3B)": {
        "path": "/app/models/deepseek-coder-1.3b.gguf",
        "type": "deepseek"
    }
}

# ----------------- Streamlit UI and Logic -----------------
st.set_page_config(layout="wide")
st.title("My Local Chatbot")

# Sidebar Inputs
st.sidebar.header("Settings")
selected_model_name = st.sidebar.selectbox("Choose a Model", list(MODEL_MAP.keys()))

# Use st.cache_resource to load the LLM once
@st.cache_resource
def load_llm(model_name):
    st.write(f"Loading the '{model_name}' model... this may take a moment.")
    model_info = MODEL_MAP[model_name]
    llm = CTransformers(
        model=model_info["path"],
        model_type=model_info["type"],
        config={'max_new_tokens': 2048, 'temperature': 0.7}
    )
    return llm

# Load the selected model
llm = load_llm(selected_model_name)
st.success(f"Model '{selected_model_name}' loaded successfully!")

MAX_HISTORY = st.sidebar.number_input("Max History", min_value=1, max_value=10, value=2, step=1)
CONTEXT_SIZE = st.sidebar.number_input("Context Size", min_value=1024, max_value=16384, value=8192, step=1024)
TEMPERATURE = st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
TOP_P = st.sidebar.slider("Top-p (nucleus sampling)", 0.0, 1.0, 0.9, 0.05)
TOP_K = st.sidebar.slider("Top-k", 0, 100, 40, 5)
MAX_TOKENS = st.sidebar.number_input("Max Tokens", min_value=256, max_value=16384, value=2048, step=256)

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