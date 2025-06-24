# Import namespaces
import os
import asyncio
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.prompt_template import PromptTemplateConfig 
from semantic_kernel.prompt_template.kernel_prompt_template import KernelPromptTemplate
from semantic_kernel.prompt_template.handlebars_prompt_template import HandlebarsPromptTemplate

load_dotenv()  # Loads variables from .env into environment
api_key = os.environ.get("AZURE_KEY")
if api_key is None:
    raise ValueError("AZURE_KEY environment variable is not set.")

project_name = os.environ.get("AZURE_ENDPOINT_NAME")

endpoint = f"https://{project_name}.openai.azure.com/"
model_name = "gpt-4.1-mini"
deployment_name = "gpt-4.1-mini"
api_version = "2024-12-01-preview"

async def main() -> None:

    # Create a kernel with Azure OpenAI chat completion
    kernel = Kernel()
    chat_completion = AzureChatCompletion(
        api_key=api_key,
        endpoint=endpoint,
        deployment_name=deployment_name
    )
    kernel.add_service(chat_completion)

    # Define a semantic function (prompt) to generate a translation, here we use a prompt template
    prompt_template = "{{$input}}\n\nTranslate this into {{$target_lang}}:"

    # Define a function in code to translate text using the prompt
    translate_fn = kernel.add_function(
        prompt=prompt_template, 
        function_name="translator", 
        plugin_name="Translator",
        max_tokens=50
    )

    # Use the function
    text = """
    Semantic Kernel is a lightweight, open-source development kit that lets 
    you easily build AI agents and integrate the latest AI models into your C#, 
    Python, or Java codebase. It serves as an efficient middleware that enables 
    rapid delivery of enterprise-grade solutions.
    """

    summary = await kernel.invoke(translate_fn, input=text, target_lang="French")
    print(summary)

# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
asyncio.run(main())