"""
title: Sequential Thinking Pipeline
author: YannisTr
version: 1.0
description: A pipeline for enhancing LLM reasoning through structured sequential thinking
"""

import logging
from typing import Generator, Iterator
from pydantic import BaseModel, Field
from open_webui.apps.openai import main as openai
from open_webui.constants import TASKS
from open_webui.utils.misc import add_or_update_system_message, get_system_message, pop_system_message

name = "Sequential"

def setup_logger():
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.set_name(name)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger

logger = setup_logger()

class Pipeline:
    class Valves(BaseModel):
        max_steps: int = Field(
            default=5,
            description="Maximum number of thinking steps"
        )
        depth_level: str = Field(
            default="detailed",
            description="Depth of reasoning (basic/detailed/comprehensive)"
        )
        structured_output: bool = Field(
            default=True,
            description="Whether to format output in structured steps"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.thinking_prompt = self.get_thinking_prompt()

    def get_thinking_prompt(self):
        return """As an AI assistant, follow these steps for sequential thinking:

1. Initial Analysis
- Understand the core question/task
- Identify key components and constraints
- Map known and unknown elements

2. Step-by-Step Reasoning
- Break down complex problems into smaller parts
- Show your work clearly for each step
- Identify and state any assumptions made
- Note potential limitations or edge cases

3. Solution Development
- Build progressively on previous steps
- Consider multiple approaches when relevant
- Verify each step's logic
- Be explicit about your reasoning process

4. Verification & Conclusion
- Review the complete reasoning chain
- Validate assumptions and conclusions
- Present final answer clearly
- Note any remaining uncertainties

Maintain this structured approach for each response, documenting your thought process throughout."""

    def resolve_model(self, body: dict) -> str:
        model_id = body.get("model")
        without_pipe = ".".join(model_id.split(".")[1:])
        return without_pipe.replace(f"{name}-", "")

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__=None,
        __task__=None,
        __model__=None,
    ) -> str | Generator | Iterator:
        try:
            model = self.resolve_model(body)
            body["model"] = model
            system_message = get_system_message(body["messages"])

            if __task__ == TASKS.TITLE_GENERATION:
                content = await self.get_completion(model, body.get("messages"))
                return f"{name}: {content}"

            logger.debug(f"Pipe {name} received: {body}")

            if system_message is None:
                system_message = self.thinking_prompt
            else:
                system_message, body["messages"] = pop_system_message(body["messages"])
                system_message = f"{self.thinking_prompt}\n\nAdditional context:\n{system_message['content']}"

            body["messages"] = add_or_update_system_message(system_message, body["messages"])

            last_message = body["messages"][-1]["content"]
            structured_prompt = f"""
Task/Question: {last_message}

Please analyze this using the sequential thinking process:

1. Initial Analysis:
[Your initial understanding and breakdown]

2. Step-by-Step Reasoning:
[Show your detailed thought process]

3. Solution Development:
[Build and explain your solution]

4. Verification & Conclusion:
[Final review and answer]
"""
            body["messages"][-1]["content"] = structured_prompt

            return await openai.generate_chat_completion(body, user=__user__)

        except Exception as e:
            logger.error(f"Error in sequential thinking pipeline: {str(e)}")
            raise

    async def get_completion(self, model: str, messages):
        response = await openai.generate_chat_completion(
            {"model": model, "messages": messages, "stream": False}
        )
        return self.get_response_content(response)

    def get_response_content(self, response):
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            logger.error(
                f'ResponseError: unable to extract content from "{response[:100]}"'
            )
            return ""