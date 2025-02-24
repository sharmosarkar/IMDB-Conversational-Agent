import streamlit as st
import ai_agent as brain
from langchain_core.messages import HumanMessage, AIMessage
import json
from typing import Dict, Union, Optional

USER = "user"
ASSISTANT = "assistant"

def process_msg_chunks(chunk: Dict, curr_step: int) -> Optional[Dict[str, str]]:
    """
    Processes message chunks from the Chain-of-Thought (CoT) in an AI-agent interaction.

    Args:
        chunk (Dict): The current message chunk from the AI agent.
        curr_step (int): The current step in the CoT sequence.

    Returns:
        Optional[Dict[str, str]]: A dictionary containing:
            - "ai_message" (str): The AI-generated message.
            - "cot" (str): The Chain-of-Thought formatted for display.
        Returns None if chunk does not contain valid AI or tool messages.
    """
    try:
        if "agent" in chunk:
            # Extract AI Message
            ai_message = chunk["agent"]["messages"][0]

            # Ensure ai_message has the required attributes
            if not isinstance(ai_message, AIMessage):
                raise ValueError("Invalid AIMessage format in agent chunk.")

            # Extract AI Message Content
            ai_message_content = ai_message.content if ai_message.content else "No Message Available"

            # Extract Tool Name
            tool_name = ai_message.additional_kwargs.get("function_call", {}).get("name", None)

            # Extract Tool Arguments
            tool_arguments_json = ai_message.additional_kwargs.get("function_call", {}).get("arguments", "{}")

            # Convert JSON string to dictionary
            try:
                tool_arguments = json.loads(tool_arguments_json)
            except json.JSONDecodeError:
                tool_arguments = {}

            # Chain-of-Thought formatted output
            template = """
            **:blue[Step {}:]**  
            **AI Thought:** {}  
            **Tool Name:** {}  
            **Tool Arguments:** {} 

            ....

            """
            cot_printable = template.format(curr_step, ai_message_content, tool_name, tool_arguments)

            return {
                "ai_message": ai_message_content,
                "cot": cot_printable
            }

        elif "tools" in chunk:
            # Extract Tool Message
            tool_message = chunk["tools"]["messages"][0]

            # Ensure tool_message has the required attributes
            if not hasattr(tool_message, "content") or not hasattr(tool_message, "name"):
                raise ValueError("Invalid ToolMessage format in tools chunk.")

            # Extract Tool Message Content
            tool_message_content = tool_message.content
            tool_name = tool_message.name

            # Chain-of-Thought formatted output
            template = """
            **:blue[Step {}:]**  (Tool Calling Step)
            **Tool Name:** {}  
            **Tool Response:** {}  

            ....

            """
            cot_printable = template.format(curr_step, tool_name, tool_message_content)

            return {
                "ai_message": "No Message Available",
                "cot": cot_printable
            }

        return None  # Return None if chunk doesn't match expected structure

    except (KeyError, IndexError, ValueError) as e:
        st.error(f"Error processing message chunk: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error in process_msg_chunks: {e}")
        return None


def render_chat_history() -> None:
    """
    Renders the chat history stored in Streamlit session state.

    Iterates over `st.session_state["chat_history"]` and displays user and assistant messages.
    If the assistant's message has Chain-of-Thought (CoT), it includes an expandable section.

    Returns:
        None
    """
    if "chat_history" in st.session_state:
        for item in st.session_state["chat_history"]:
            if item["role"] == USER:
                with st.chat_message(USER):
                    st.write(item["content"])
            elif item["role"] == ASSISTANT:
                with st.chat_message(ASSISTANT):
                    st.write(item["content"])
                    with st.expander("Chain-of-Thought (CoT)"):
                        st.write(item["cot"])


def main() -> None:
    """
    Main function to run the Streamlit chatbot application.

    - Initializes session state variables.
    - Captures user input via chat input.
    - Sends the user query to the AI agent and processes responses.
    - Renders chat history to display past interactions.

    Returns:
        None
    """
    try:
        # Initialize chat history if not already in session state
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # Initialize the AI agent if not already in session state
        if "agent" not in st.session_state:
            agent = brain.initialize_agent()
            if agent is None:
                st.error("Failed to initialize AI agent.")
                return
            st.session_state["agent"] = agent

        # Render the chat input to accept user input
        prompt: str = st.chat_input("Enter a prompt here")

        if prompt:
            # Append user input to chat history
            st.session_state["chat_history"].append({
                "role": USER, "content": prompt
            })

            # Retrieve the AI agent from session state
            agent = st.session_state["agent"]

            # Ensure agent has a valid stream method
            if not hasattr(agent, "stream"):
                st.error("AI agent does not support streaming.")
                return

            tracker = 1
            st.session_state["response_tracker"] = []  # To track intermediate AI responses

            try:
                for chunk in agent.stream(
                    {"messages": [HumanMessage(content=prompt)]},
                    {"configurable": {"thread_id": "1"}}
                ):
                    cot_details = process_msg_chunks(chunk, tracker)
                    if cot_details:
                        st.session_state["response_tracker"].append(cot_details)
                    tracker += 1

                if not st.session_state["response_tracker"]:
                    st.error("No valid AI response received.")
                    return

                # Retrieve final AI response
                ai_resp = st.session_state["response_tracker"][-1]["ai_message"]

                # Formulate CoT for display
                if len(st.session_state["response_tracker"]) > 1:
                    cot_display = "\n".join(
                        item["cot"] for item in st.session_state["response_tracker"] if item
                    )
                else:
                    cot_display = "Nothing to Display."

                # Append AI response and CoT to chat history
                st.session_state["chat_history"].append({
                    "role": ASSISTANT, "content": ai_resp, "cot": cot_display
                })

            except Exception as e:
                st.error(f"Error while processing AI response: {e}")

        # Render chat history
        render_chat_history()

    except Exception as e:
        st.error(f"Unexpected error in main function: {e}")


if __name__ == "__main__":
    main()
