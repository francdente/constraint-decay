import os
import uuid
from pathlib import Path

from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import MessageEvent
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.grep import GrepTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool

MAX_TURNS: int = 6


def _get_env_or_raise(key: str) -> str:
    var = os.getenv(key)
    if var is None:
        raise Exception(f"{key} is required!")
    return var


def main() -> None:
    api_key_value = _get_env_or_raise("LLM_API_KEY")
    model_name = _get_env_or_raise("LLM_MODEL")
    prompt = _get_env_or_raise("PROMPT")

    print("WARNING: missing system guidance !!")
    native_tc = os.getenv("NATIVE_TOOL_CALLING", "true").lower() != "false"

    llm = LLM(
        model=model_name,
        api_key=api_key_value,
        base_url=os.getenv("LLM_BASE_URL", None),
        temperature=0.0,
        native_tool_calling=native_tc,
    )

    agent = Agent(
        llm=llm,
        tools=[
            Tool(name=TerminalTool.name),
            Tool(name=FileEditorTool.name),
            Tool(name=TaskTrackerTool.name),
            Tool(name=GrepTool.name),
        ],
    )

    conversation_id = uuid.uuid4()
    trajectories_path = Path("/trajectories")
    repository_path = Path("/repository")

    conversation = Conversation(
        agent=agent,
        workspace=repository_path,
        persistence_dir=trajectories_path,
        conversation_id=conversation_id,
        max_iteration_per_run=200
    )
    conversation.send_message(prompt)

    turns = 0
    while turns < MAX_TURNS:
        turns += 1

        conversation.run()
        if conversation.state.execution_status != ConversationExecutionStatus.FINISHED:
            print(f"Exiting with status: {conversation.state.execution_status}")
            break

        last_event = conversation.state.events[-1]

        if isinstance(last_event, MessageEvent) and last_event.source == "agent":
            print("Agent sent a message, continuing...")
            conversation.send_message(
                "Continue with your task. When you are completely done and confident "
                "the task is finished, you MUST call the 'finish' tool to end the "
                "conversation. Do not send a message - use the finish tool instead. "
                "If you are not yet done, continue working on the task."
            )
        else:
            print("Agent completed task!")
            break


if __name__ == "__main__":
    main()
