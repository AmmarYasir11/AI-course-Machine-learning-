import ollama

def load_file_as_context(filepath):
    """Read the file content from a given path."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    # Array of context file paths
    context_files = [
        r'C:\Users\user\PycharmProjects\PythonProject\pakistan.txt',
        r'C:\Users\user\PycharmProjects\PythonProject\resturentdata.txt'
    ]

    # Combine contents from both files
    combined_context = ""
    for file_path in context_files:
        combined_context += load_file_as_context(file_path) + "\n\n"

    # Add the combined content to the system message
    messages = [
        {
            'role': 'system',
            'content': f"You are an assistant with access to the following combined context:\n\n{combined_context}"
        }
    ]

    # Chat loop
    while True:
        user_input = input("You: ")
        if user_input.lower() in ('exit', 'quit'):
            break

        messages.append({'role': 'user', 'content': user_input})

        response = ollama.chat(
            model='llama3.2',
            messages=messages
        )

        assistant_message = response['message']['content']
        print("Assistant:", assistant_message)

        messages.append({'role': 'assistant', 'content': assistant_message})


if __name__ == "__main__":
    main()
