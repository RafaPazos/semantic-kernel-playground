# Import namespaces
import os
import asyncio
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.agents import ChatHistoryAgentThread
from semantic_kernel.contents import ChatHistorySummarizationReducer

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

    # 1. Create a specialized question-answering agent
    space_expert_agent = ChatCompletionAgent(
        service=chat_completion,
        name="SpaceExpert",
        instructions="""You are an expert in astronomy and space exploration.
        
        When answering questions:
        - Provide factual, scientifically accurate information
        - Include relevant dates, measurements, and statistics when applicable
        - Explain complex concepts in accessible language
        - Differentiate between established facts and theoretical or speculative ideas
        - When appropriate, mention recent developments or missions
        
        Focus on being educational and inspiring curiosity about space.
        """
    )

    # 2. Create a chat history with a relevant question and a summerizer

    # 2.1. Create a chat history summarization reducer to keep track of the conversation and reduce the tokens used summarizing the conversation so far.
    chat_history_with_reducer = ChatHistorySummarizationReducer(
        service=chat_completion,
        target_count=2,
        threshold_count=2,
    )
    chat_history_with_reducer.clear()

    # 2.2. Create a chat history agent thread to manage the conversation and store the chat history
    thread: ChatHistoryAgentThread = ChatHistoryAgentThread(chat_history=chat_history_with_reducer)
    user_messages = [
        "What are exoplanets and how do scientists detect them?",
        "What is the James Webb Space Telescope and what will it study?"
    ]

    # 3. Execute the agent and print its response
    for user_message in user_messages:
        print("*** User:", user_message)
        response = await space_expert_agent.get_response(messages=user_message, thread=thread)
        thread = response.thread
        print("*** Agent:", response.content)

# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
asyncio.run(main())