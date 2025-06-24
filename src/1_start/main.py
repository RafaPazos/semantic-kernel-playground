# Import namespaces
import os
import asyncio
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions.kernel_arguments import KernelArguments

load_dotenv()  # Loads variables from .env into environment
api_key = os.environ.get("AZURE_KEY")
if api_key is None:
    raise ValueError("AZURE_KEY environment variable is not set.")

project_name = os.environ.get("AZURE_ENDPOINT_NAME")

endpoint = f"https://{project_name}.openai.azure.com/"
model_name = "gpt-4.1-mini"
deployment_name = "gpt-4.1-mini"
api_version = "2024-12-01-preview"

# Create a kernel with Azure OpenAI chat completion
kernel = Kernel()
chat_completion = AzureChatCompletion(
    api_key=api_key,
    endpoint=endpoint,
    deployment_name=deployment_name
)
kernel.add_service(chat_completion)

async def main() -> None:
    # Test the chat completion service
    response = await kernel.invoke_prompt(prompt="Give me a list of 10 breakfast foods with tomatoes and olive oil", arguments=KernelArguments())
    print("Assistant > " + str(response))

# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
asyncio.run(main())