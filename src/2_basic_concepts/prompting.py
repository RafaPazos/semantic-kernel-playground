# Import namespaces
import os
import asyncio
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.prompt_template.kernel_prompt_template import KernelPromptTemplate
from semantic_kernel.prompt_template import PromptTemplateConfig      # for prompt_config
from semantic_kernel.functions import kernel_function
from semantic_kernel.prompt_template.handlebars_prompt_template import HandlebarsPromptTemplate

# here we create the kernel
load_dotenv()  # Loads variables from .env into environment
api_key = os.environ.get("AZURE_KEY")
if api_key is None:
    raise ValueError("AZURE_KEY environment variable is not set.")

project_name = os.environ.get("AZURE_ENDPOINT_NAME")

endpoint = f"https://{project_name}.openai.azure.com/"
model_name = "gpt-4.1-mini"
deployment_name = "gpt-4.1-mini"
api_version = "2024-12-01-preview"

# Define the execution settings for the prompt, it uses handlebars template
handlebars_template = """
<message role="system">You are an AI assistant designed to help with image recognition tasks.</message>
<message role="user">
    <text>{{request}}</text>
    <image>{{imageData}}</image>
</message>
"""

# and these are the arguments for the prompt
arguments = {
    "request": "Describe this image:",
    "imageData": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAAXNSR0IArs4c6QAAACVJREFUKFNj/KTO/J+BCMA4iBUyQX1A0I10VAizCj1oMdyISyEAFoQbHwTcuS8AAAAASUVORK5CYII="
}

async def main() -> None:

    # Create a kernel with Azure OpenAI chat completion
    kernel = Kernel()
    chat_completion = AzureChatCompletion(
        api_key=api_key,
        endpoint=endpoint,
        deployment_name=deployment_name
    )
    kernel.add_service(chat_completion)

    # Create a history of the conversation
    history = ChatHistory()

    # Create the prompt template config and function
    prompt_config = PromptTemplateConfig(
        template=handlebars_template,
        template_format="handlebars",
        name="Vision_Chat_Prompt"
    )
    function = HandlebarsPromptTemplate(prompt_template_config=prompt_config)

    result = await kernel.invoke_prompt(
        prompt=prompt_config.template,
        request=arguments["request"],
        imageData=arguments["imageData"],
        service_id="chat_completion",
        template_format="handlebars"
    )
    print("Kernel result:", result)


# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
asyncio.run(main())