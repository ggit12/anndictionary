import os
import openai
from openai.error import AuthenticationError

def set_openai_api_key(api_key):
    bashrc_path = os.path.expanduser("~/.bashrc")
    export_command = f'\nexport OPENAI_API_KEY="{api_key}"\n'
    
    if not os.path.exists(bashrc_path):
        with open(bashrc_path, 'w') as f:
            f.write(export_command)
    else:
        with open(bashrc_path, 'a') as f:
            f.write(export_command)

    print(f"Added OPENAI_API_KEY to {bashrc_path}. Please restart your terminal or run 'source ~/.bashrc' to apply changes.")

def validate_api_key(api_key):
    openai.api_key = api_key
    try:
        # Attempt to list models to validate the key
        openai.Model.list()
        return True
    except AuthenticationError:
        return False

if __name__ == "__main__":
    api_key = input("Please enter your OpenAI API key: ").strip()
    if api_key:
        while not validate_api_key(api_key):
            print("Invalid API key. Please try again.")
            api_key = input("Please enter your OpenAI API key (leave empty to skip modification of bashrc): ").strip()
        set_openai_api_key(api_key)
    else:
        print("No API key entered. Skipping modification of .bashrc. Before using AI integration, you must set OPENAI_API_KEY environment variable.")
