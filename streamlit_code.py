import streamlit as st
import ollama
import os

# --- Helper functions ---

def load_file_as_context(filepath):
    """Load and label file content."""
    with open(filepath, 'r', encoding='utf-8') as f:
        filename = os.path.basename(filepath)
        content = f.read()
        return f"[FILE: {filename}]\n{content.strip()}\n"

def select_file_based_on_question(question):
    """Simple keyword matching to select a file."""
    question_lower = question.lower()
    if 'restaurant' in question_lower or 'resturent' in question_lower:
        return r'C:\Users\user\PycharmProjects\PythonProject\resturentdata.txt'
    elif 'pakistan' in question_lower:
        return r'C:\Users\user\PycharmProjects\PythonProject\pakistan.txt'
    else:
        return None

# --- Streamlit App ---

st.set_page_config(page_title="Document Chatbot", layout="centered")

st.title("📄 M.Ammar AI Chatbot (LLaMA 3.2 + Ollama)")
st.markdown("Ask a question about **Pakistan** or **restaurants**. The model will load the correct document for context.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input from user
user_input = st.chat_input("Ask something...")

if user_input:
    # Detect file based on question
    selected_file = select_file_based_on_question(user_input)

    messages = []

    if selected_file:
        file_content = load_file_as_context(selected_file)
        messages.append({
            'role': 'system',
            'content': (
                "You have access to the following document:\n\n" + file_content +
                "\nUse this document to answer the user's question."
            )
        })
    else:
        messages.append({
            'role': 'system',
            'content': (
                "The user has asked a question, but it does not match any known document keywords. "
                "Please let them know you need a more specific question."
            )
        })

    messages.append({'role': 'user', 'content': user_input})

    # Get response from Ollama
    try:
        response = ollama.chat(
            model='llama3.2',
            messages=messages
        )
        assistant_message = response['message']['content']
    except Exception as e:
        assistant_message = f"⚠️ Error: {str(e)}"

    # Save chat to session
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("assistant", assistant_message))

# Display chat history
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)
