import openai
import os

def read_csharp_code(file_path):
    with open(file_path, 'r') as file:
        return file.read()
        
def generate_unit_tests(csharp_code, api_key):
    # Set up the OpenAI API client, use input
    openai.api_key = api_key
    prompt = f"Generate C# unit tests for the following C# code:\n{csharp_code}\n\nUnit tests:\n"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
    )
    generated_tests = response.choices[0].text.strip()
    return generated_tests
if __name__ == "__main__":
    input_file_path = input("Enter C# code path:")  # Replace this with the path to your input C# file
    csharp_code = read_csharp_code(input_file_path)
    api_key = input("Enter OpenAI API key:")
    unit_tests = generate_unit_tests(csharp_code, api_key)
    print(unit_tests)