# Import namespaces
import os
import asyncio
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.agents import ChatCompletionAgent

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

    # Here we create a simple agent that uses the chat completion service, and assign to it our service.
    simple_agent = ChatCompletionAgent(
        service=chat_completion,
        name="ai_assistant",
        instructions="You are an AI assistant that helps users with their questions."
    )
    
    # This simple agent can handle messages and get responses from the service.
    response = await simple_agent.get_response(messages="What's the capital of France?")
    print("Agent:", response.content)

    # We can also create specialized agents that use the same service, but with different configurations. 
    # These configurations accept arguments that can be used to customize the agent's behavior.
    def generate_specialized_agent(expertise, tone, length):
        templated_agent = ChatCompletionAgent(
            service=chat_completion,
            name=f"{expertise}_assistant",
            instructions="""You are an AI assistant specializing in {{$expertise}}.
            Your tone should be {{$tone}} and your responses should be {{$length}} in length.
            """,
            arguments=KernelArguments(
                expertise=expertise,
                tone=tone,
                length=length
            ),
        )
        return templated_agent

    # We can generate specialized agents for different tasks, such as programming languages. We give these agents an expertise and even a character tone.
    python_agent = generate_specialized_agent("python_programming", "friendly, funny and snappy", "short")
    java_agent = generate_specialized_agent("java_programming", "sad, because it has been programming java for years and this is an awful experience", "long")

    # We can now use these agents to get some code examples.
    print("Python agent:\n", await python_agent.get_response(messages="Write me a hello world example"))

    print("Java agent:\n",await java_agent.get_response(messages="Write me a hello world example"))


# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
asyncio.run(main())