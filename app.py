import streamlit as st
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import ChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

# Streamlit Setup
st.set_page_config(layout="wide")
st.title("My Local Chatbot")

# Sidebar Inputs
st.sidebar.header("Settings")

# Dropdown for model selection
model_options = ["llama3:8b", "deepseek-r1:1.5b"]
MODEL = st.sidebar.selectbox("Choose a Model", model_options, index=0)

# Inputs for history + context size
MAX_HISTORY = st.sidebar.number_input("Max History", min_value=1, max_value=10, value=2, step=1)
CONTEXT_SIZE = st.sidebar.number_input("Context Size", min_value=1024, max_value=16384, value=8192, step=1024)

# Advanced Parameters
st.sidebar.subheader("Model Parameters")
TEMPERATURE = st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
TOP_P = st.sidebar.slider("Top-p (nucleus sampling)", 0.0, 1.0, 0.9, 0.05)
TOP_K = st.sidebar.slider("Top-k", 0, 100, 40, 5)
MAX_TOKENS = st.sidebar.number_input("Max Tokens", min_value=256, max_value=16384, value=2048, step=256)

# Memory Controls 
def clear_memory():
    chat_history = ChatMessageHistory()
    st.session_state.memory = ConversationBufferMemory(chat_memory=chat_history)
    st.session_state.chat_history = []
    st.session_state.summary = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
# NEW: Initialize a summary variable
if "summary" not in st.session_state:
    st.session_state.summary = ""

# Button to clear memory manually
if st.sidebar.button("Clear Conversation History"):
    clear_memory()
# LangChain LLM Setup
llm = ChatOllama(
    model=MODEL,
    streaming=True,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    num_predict=MAX_TOKENS
)

# ---
# NEW: Summary Chain and Functions

# Prompt Template for summarization
summary_prompt_template = PromptTemplate(
    input_variables=["chat_history"],
    template="You are a summarizer. Summarize the following conversation to preserve key information and context. \n\n{chat_history}"
)

# Chain for summarization
summary_chain = summary_prompt_template | llm


def get_summary(chat_history_str):
    """Generates a summary of the conversation history."""
    return summary_chain.invoke({"chat_history": chat_history_str})

def summarize_chat():
    if not st.session_state.chat_history:
        return "No chat history to summarize."
    
    # Pass the full chat history list to the summarization function
    return get_summary(st.session_state.chat_history)


if st.sidebar.button("Summarize Chat"):
    with st.sidebar:
        st.markdown("**Chat Summary:**")
        summary = summarize_chat()
        st.success(summary)
# ---

# Main Prompt Template 
# Now includes a summary variable
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

# NEW & CORRECTED Trim Function
def trim_memory():
    # Trim the chat history to the MAX_HISTORY size
    if len(st.session_state.chat_history) > MAX_HISTORY * 2:
        # Get the history to be trimmed
        history_to_summarize = st.session_state.chat_history[:(len(st.session_state.chat_history) - MAX_HISTORY * 2)]
        
        # Format the history string for the summary prompt
        history_str = ""
        for msg in history_to_summarize:
            history_str += f"{msg['role']}: {msg['content']}\n"
        
        # Get a summary of the old messages and append to the existing summary
        new_summary = get_summary(history_str)
        st.session_state.summary += "\n" + new_summary
        
        # Remove the old messages from the chat history
        st.session_state.chat_history = st.session_state.chat_history[(len(st.session_state.chat_history) - MAX_HISTORY * 2):]


# Handle User Input 
if prompt := st.chat_input("Say something"):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # Call the updated trim_memory function
    trim_memory()

    # Format the current, non-summarized history for the prompt template
    formatted_history = ""
    for msg in st.session_state.chat_history:
        formatted_history += f"{msg['role']}: {msg['content']}\n"

    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""

        # Pass both 'human_input', 'history', and 'summary' to the chain
        for chunk in chain.stream({
                "human_input": prompt,
                "history": formatted_history,
                "summary": st.session_state.summary
            }):
            full_response += chunk.content
            response_container.markdown(full_response)
        
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})