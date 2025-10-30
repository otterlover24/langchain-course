from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.tools import render_text_description
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.output_parsers import BaseOutputParser
import re

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
    text = text.strip()
    return len(text)


if __name__ == "__main__":
    # Your main application logic here
    # print("Hello, ReAct LangChain!")
    # print(f"Length of sample text: {get_text_length.invoke("Dog")}")
    tools = [get_text_length]

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
Thought:
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([tool.name for tool in tools]),
    )

    llm = ChatOpenAI(temperature=0).bind(stop=["\nObservation:", "Observation:"])

    agent = (
        {"input": lambda x: x["input"]} | prompt | llm | ReActSingleInputOutputParser()
    )

    res = agent.invoke({"input": "What is the length of the word 'DOG'?"})
    print(res)
