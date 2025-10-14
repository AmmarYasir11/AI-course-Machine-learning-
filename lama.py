import ollama

def main():
    print("Chat with LLaMA 3.2! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ('exit', 'quit'):
            print("Goodbye!")
            break

        response = ollama.chat(
            model='llama3.2',
            messages=[{'role': 'user', 'content': user_input}]
        )

        print("LLaMA 3.2:", response['message']['content'])

if __name__ == "__main__":
    main()

