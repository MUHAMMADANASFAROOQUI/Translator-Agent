from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# Get input from user
while True:
    target_language = input("Enter the target language: ").lower()
    language_list = ["english","urdu","french","german","spanish", "chinese", "japanese"]
    if target_language in language_list:
        break
    print("Input a Valid language")

sentence_to_translate = input("Enter the sentence to translate: ").strip()

# Setup Gemini API client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Define model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# Run config
config = RunConfig(
    model=model,
    tracing_disabled=True
)

# Translator Agent
writer = Agent(
    name='Translator Agent',
    ### System prompt or developer prompt
    instructions="""
        You are a language translator. First, identify the target language based on user input, 
        then translate the provided sentence into that language.
    """
)

# Combine user input into one formatted string
formatted_input = f"Language: {target_language}\nSentence: {sentence_to_translate}"

# Run the agent
response = Runner.run_sync(
    writer,
    ### User Input
    input=formatted_input,
    run_config=config
)

# Output the result
print("\nTranslated Output:")
print(response.final_output)
