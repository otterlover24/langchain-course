from dotenv import load_dotenv
from langchain.tools import tool, BaseTool
from langchain_core.tools import render_text_description
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.output_parsers import BaseOutputParser
from typing import Union, List
import re


def format_log_to_str(intermediate_steps: List[tuple]) -> str:
    """Format intermediate steps to string."""
    if not intermediate_steps:
        return ""
    thoughts = ""
    for action, observation in intermediate_steps:
        thoughts += f"{action.log}\nObservation: {observation}\nThought: "
    return thoughts


load_dotenv()


class ReActSingleInputOutputParser(BaseOutputParser):
    """Custom ReAct output parser that mimics the original functionality."""

    def parse(self, text: str):
        # Look for "Final Answer:" pattern
        if "Final Answer:" in text:
            final_answer = text.split("Final Answer:")[-1].strip()
            return AgentFinish(return_values={"output": final_answer}, log=text)

        # Look for Action: and Action Input: patterns
        action_match = re.search(r"Action: (.+)", text)
        action_input_match = re.search(r"Action Input: (.+)", text)

        if action_match and action_input_match:
            action = action_match.group(1).strip()
            action_input = action_input_match.group(1).strip()
            # Remove quotes if present
            action_input = action_input.strip("'\"")
            return AgentAction(tool=action, tool_input=action_input, log=text)

        # If no pattern matches, return the text as is
        return AgentFinish(return_values={"output": text}, log=text)


@tool
def get_text_length(text: str) -> int:
    """Returns the length of the input text in terms of number of characters."""
    print(f"enter get_text_length  with {text=}")
    text = text.strip()
    return len(text)


def find_tool_by_name(tools: List[BaseTool], tool_name: str) -> BaseTool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found.")


if __name__ == "__main__":
    # Your main application logic here
    # print("Hello, ReAct LangChain!")
    # print(f"Length of sample text: {get_text_length.invoke("Dog")}")
    tools: List[BaseTool] = [get_text_length]

    template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([tool.name for tool in tools]),
    )

    # llm = ChatOpenAI(temperature=0).bind(stop=["\nObservation:", "Observation:"])
    llm = ChatOpenAI(temperature=0)

    intermediate_steps = []

    agent = (
        {
            "input": lambda x: x.get("input", ""),
            "agent_scratchpad": lambda x: format_log_to_str(
                x.get("agent_scratchpad", [])
            ),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        input={"input": "What is the length of the word DOG?"},
        agent_scratchpad=intermediate_steps,
    )
    print(agent_step)

    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input

        observation = tool_to_use.invoke(tool_input)
        print(f"Observation: {observation}")
        intermediate_steps.append((agent_step, observation))

    print(
        f"Agent scratchpad so far:\n{format_log_to_str(intermediate_steps)}\n END OF SCRATCHPAD"
    )

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        input={"input": "What is the length of the word DOG?"},
        agent_scratchpad=intermediate_steps,
    )
    print(agent_step)
