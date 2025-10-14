import ollama

def load_file_as_context(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    # Updated with your file path
    context_file = r'C:\Users\user\PycharmProjects\PythonProject\pakistan.txt'
    context_text = load_file_as_context(context_file)

    # Initial message with the file's content
    messages = [
        {
            'role': 'system',
            'content': f"You are an assistant with access to the following context:\n\n{context_text}"
        }
    ]

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

        messages.append({'role': 'assistant', 'content': assistant_message})


if __name__ == "__main__":
    main()
