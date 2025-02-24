import prompts as p
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
import constants as c
import toolbox
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from typing import List, Optional

# Load environment variables from .env file (ensures API keys and other configurations are available)
load_dotenv()


def initialize_agent() -> Optional[object]:
    """
    Initializes and returns a LangChain ReAct Agent with the Google Generative AI model.

    This function performs the following steps:
    1. Initializes a Google Generative AI chat model with specific parameters.
    2. Loads AI tools from the `toolbox` module.
    3. Creates an in-memory chat memory object.
    4. Constructs a ReAct agent using the AI model, tools, memory, and system prompt.

    Returns:
        object: The initialized ReAct agent if successful, or None in case of failure.
    """

    try:
        # Ensure the necessary model constant is available
        if not hasattr(c, "LLM_MODEL") or not isinstance(c.LLM_MODEL, str):
            raise ValueError("LLM_MODEL constant is missing or not a string in constants.py")

        # Initialize the Google Generative AI chat model with predefined parameters
        llm = ChatGoogleGenerativeAI(
            model=c.LLM_MODEL,  # Ensure this model exists in constants.py
            temperature=0,  # Deterministic output (no randomness in responses)
            timeout=180,  # Maximum wait time for API response (in seconds)
            max_retries=2  # Maximum retry attempts in case of failure
        )

        # Verify the existence of required tools in the toolbox module
        if not hasattr(toolbox, "adaptive_structured_query_tool"):
            raise AttributeError("adaptive_structured_query_tool is missing in toolbox.py")

        if not hasattr(toolbox, "adaptive_semantic_search_tool"):
            raise AttributeError("adaptive_semantic_search_tool is missing in toolbox.py")

        # Define the AI tools to be used by the agent
        toolkit: List[object] = [
            toolbox.adaptive_structured_query_tool,
            toolbox.adaptive_semantic_search_tool
        ]

        # Initialize in-memory chat memory for conversation tracking
        memory = MemorySaver()  # Note: This memory is not persistent between runs

        # Ensure the required system prompt exists in the prompts module
        if not hasattr(p, "react_agent_sys_prompt") or not isinstance(p.react_agent_sys_prompt, str):
            raise ValueError("react_agent_sys_prompt is missing or not a string in prompts.py")

        # Create a ReAct agent using the AI model, tools, and memory
        agent = create_react_agent(
            model=llm,
            tools=toolkit,
            checkpointer=memory,
            prompt=p.react_agent_sys_prompt
        )

        return agent  # Successfully created agent is returned

    except (ImportError, AttributeError, ValueError) as e:
        # Handle missing modules, attributes, or incorrect values
        print(f"Error initializing agent: {e}")
        return None  # Return None if initialization fails due to missing dependencies

    except Exception as e:
        # Catch all other unexpected errors
        print(f"Unexpected error during agent initialization: {e}")
        return None
