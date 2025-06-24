# Import namespaces
import os
import asyncio
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.prompt_template.kernel_prompt_template import KernelPromptTemplate
from typing import TypedDict, Annotated, List, Optional
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.function_choice_behavior import (
    FunctionChoiceBehavior,
)
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

# here we will work with Native functions, Semantic Kernel provides a way to define native functions in Python code using Plugins. 
# Plugins are Python classes that define a set of functions that can be invoked by the kernel. 
# Each function in a plugin is defined using the @kernel_function decorator, which allows the kernel to identify and execute the function when invoked. 
# The function signature should include type hints for the input arguments and return value, as well as any annotations that provide additional information about the function.

# We will implement a simple plugin that manages a list of lights and their states.

# Define a data model for a light
class LightModel(TypedDict):
    id: int
    name: str
    is_on: bool | None
    brightness: int | None
    hex: str | None

# Define a plugin to manage lights
class LightsPlugin:
    def __init__(self, lights: list[LightModel]):
        self.lights = lights
    
    # Define a function to get a list of lights
    @kernel_function(name="get_lights", description="Gets a list of lights and their current state.")
    async def get_lights(self) -> List[LightModel]:
        """Gets a list of lights and their current state."""
        return self.lights
    
    # Define a function to get the state of a particular light
    @kernel_function(name="get_state", description="Gets the state of a particular light.")
    async def get_state(
        self, id: Annotated[int, "The ID of the light"]
    ) -> Optional[LightModel]:
        """Gets the state of a particular light."""
        for light in self.lights:
            if light["id"] == id:
                return light
        return None
    
    # Define a function to change the state of a light
    @kernel_function
    async def change_state(
        self, id: Annotated[int, "The ID of the light"], new_state: LightModel
    ) -> Optional[LightModel]:
        """Changes the state of the light."""
        for light in self.lights:
            if light["id"] == id:
                light["is_on"] = new_state.get("is_on", light["is_on"])
                light["brightness"] = new_state.get("brightness", light["brightness"])
                light["hex"] = new_state.get("hex", light["hex"])
                return light
        return None


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

async def main() -> None:

    # Create a kernel with Azure OpenAI chat completion
    kernel = Kernel()
    chat_completion = AzureChatCompletion(
        api_key=api_key,
        endpoint=endpoint,
        deployment_name=deployment_name
    )
    kernel.add_service(chat_completion)

    # Create some lights
    lights = [
        {"id": 1, "name": "Table Lamp", "is_on": False, "brightness": 100, "hex": "FF0000"},
        {"id": 2, "name": "Porch light", "is_on": False, "brightness": 50, "hex": "00FF00"},
        {"id": 3, "name": "Chandelier", "is_on": True, "brightness": 75, "hex": "0000FF"},
    ]

    # Instantiate the plugin
    plugin = LightsPlugin(lights=lights)

    # Add the plugin to the kernel
    kernel.add_plugin(
        plugin=plugin,
        plugin_name="Lights",
    )

    # Enable planning
    execution_settings = AzureChatPromptExecutionSettings()
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    # Create a history of the conversation
    history = ChatHistory()

    async def manage_lights(message: str) -> None:
        history.add_user_message(message)
        
        # Print the user message
        print("User > " + message)

        # Get the response from the AI
        result = await chat_completion.get_chat_message_content(
            chat_history=history,
            settings=execution_settings,
            kernel=kernel,
        )

        # Print the results
        print("Assistant > " + str(result))

    await manage_lights(message="Please turn on the table lamp")
    await manage_lights(message="Please turn off the porch light")
    await manage_lights(message="Please show the status of the lights")
    

# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
asyncio.run(main())