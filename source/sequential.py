"""
title: Sequential Thinking Pipeline
author: YannisTr
version: 1.0
description: A pipeline for enhancing LLM reasoning through structured sequential thinking
"""

import logging
import json
from typing import Generator, Iterator, List, Union, Optional
from pydantic import BaseModel, Field
import aiohttp
import os

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
        openai_api_key: str = Field(
            default=os.getenv("OPENAI_API_KEY", ""),
            description="OpenAI API Key for completions"
        )
        openai_api_base: str = Field(
            default="https://api.openai.com/v1",
            description="OpenAI API Base URL"
        )
        model: str = Field(
            default="gpt-3.5-turbo",
            description="Default model to use"
        )
        temperature: float = Field(
            default=0.7,
            description="Temperature for completions"
        )

    def __init__(self):
        self.name = "Sequential Thinking Pipeline"
        self.valves = self.Valves()
        self.thinking_prompt = self._get_thinking_prompt()
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler()
            handler.set_name(self.name)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False
        return logger

    def _get_thinking_prompt(self):
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

    async def on_startup(self):
        self.logger.info(f"Starting {self.name}")
        # Verify API key is set
        if not self.valves.openai_api_key:
            self.logger.warning("OpenAI API key not set!")
        pass

    async def on_shutdown(self):
        self.logger.info(f"Shutting down {self.name}")
        pass

    def _prepare_messages(self, user_message: str, system_message: Optional[str] = None) -> List[dict]:
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append({"role": "system", "content": system_message})
        else:
            messages.append({"role": "system", "content": self.thinking_prompt})

        structured_prompt = f"""
Task/Question: {user_message}

Please analyze this using the sequential thinking process:

1. Initial Analysis:
[Provide your initial understanding and breakdown of the task/question]

2. Step-by-Step Reasoning:
[Show your detailed thought process, including assumptions and considerations]

3. Solution Development:
[Build and explain your solution, showing how each step connects]

4. Verification & Conclusion:
[Review your reasoning and present your final, validated answer]
"""
        messages.append({"role": "user", "content": structured_prompt})
        return messages

    async def _process_with_openai(self, messages: List[dict], model: Optional[str] = None) -> str:
        """Process messages with OpenAI API"""
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.valves.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model or self.valves.model,
                "messages": messages,
                "temperature": self.valves.temperature
            }

            async with session.post(
                f"{self.valves.openai_api_base}/chat/completions",
                headers=headers,
                json=data
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API error: {error_text}")
                
                result = await response.json()
                return result["choices"][0]["message"]["content"]

    async def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict
    ) -> Union[str, Generator, Iterator]:
        """Main pipeline processing method"""
        try:
            # Handle title generation requests
            if body.get("title", False):
                return self.name

            # Extract system message if present
            system_message = None
            if messages and messages[0].get("role") == "system":
                system_message = messages[0]["content"]

            # Prepare messages with structured thinking format
            processed_messages = self._prepare_messages(user_message, system_message)

            # Process with OpenAI
            self.logger.debug(f"Processing request with model: {model_id}")
            response = await self._process_with_openai(processed_messages, model_id)

            # Structure the response
            if self.valves.structured_output:
                try:
                    # Add any additional formatting if needed
                    return response
                except Exception as e:
                    self.logger.warning(f"Failed to structure output: {e}")
                    return response
            
            return response

        except Exception as e:
            self.logger.error(f"Error in sequential thinking pipeline: {str(e)}")
            raise

    def _verify_response(self, response: str) -> bool:
        """Verify that the response follows the sequential thinking structure"""
        required_sections = [
            "Initial Analysis",
            "Step-by-Step Reasoning",
            "Solution Development",
            "Verification & Conclusion"
        ]
        
        return all(section.lower() in response.lower() for section in required_sections)

    async def _retry_with_clarification(
        self,
        messages: List[dict],
        model: str,
        attempt: int = 1
    ) -> str:
        """Retry the request with additional clarification if needed"""
        if attempt > 3:
            raise Exception("Failed to get a properly structured response after 3 attempts")

        clarification = (
            "Please ensure your response follows the sequential thinking structure "
            "with all four sections clearly labeled: Initial Analysis, Step-by-Step "
            "Reasoning, Solution Development, and Verification & Conclusion."
        )
        
        messages.append({"role": "user", "content": clarification})
        response = await self._process_with_openai(messages, model)
        
        if self._verify_response(response):
            return response
            
        return await self._retry_with_clarification(messages, model, attempt + 1)