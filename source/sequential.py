"""
title: Sequential Thinking Pipeline
author: YannisTr
version: 1.0
description: A pipeline for enhancing LLM reasoning through structured sequential thinking
"""

import logging
import json
from typing import Generator, Iterator, List, Union, Optional, Any
from pydantic import BaseModel, Field
import aiohttp
import os
import asyncio

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
        self._session: Optional[aiohttp.ClientSession] = None
        
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
- Note any remaining uncertainties"""

    async def on_startup(self):
        self.logger.info(f"Starting {self.name}")
        if not self.valves.openai_api_key:
            self.logger.warning("OpenAI API key not set!")
        self._session = aiohttp.ClientSession()

    async def on_shutdown(self):
        self.logger.info(f"Shutting down {self.name}")
        if self._session:
            await self._session.close()

    def _prepare_messages(self, user_message: str, system_message: Optional[str] = None) -> List[dict]:
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        else:
            messages.append({"role": "system", "content": self.thinking_prompt})

        messages.append({"role": "user", "content": user_message})
        return messages

    async def _process_with_openai(self, messages: List[dict], model: Optional[str] = None) -> str:
        if not self._session:
            self._session = aiohttp.ClientSession()
            
        headers = {
            "Authorization": f"Bearer {self.valves.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model or self.valves.model,
            "messages": messages,
            "temperature": self.valves.temperature
        }

        try:
            async with self._session.post(
                f"{self.valves.openai_api_base}/chat/completions",
                headers=headers,
                json=data
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API error: {error_text}")
                
                result = await response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            self.logger.error(f"Error in OpenAI request: {str(e)}")
            raise

    def filter_inlet(self, messages: List[dict], body: dict) -> tuple[List[dict], dict]:
        return messages, body

    def filter_outlet(self, response: Any) -> Any:
        return response

    async def pipe(self, messages: List[dict], body: dict) -> Union[str, dict]:
        """Main pipeline processing method"""
        try:
            if body.get("title", False):
                return self.name

            user_message = messages[-1]["content"] if messages else ""
            model_id = body.get("model", self.valves.model)
            
            # Extract system message if present
            system_message = None
            if messages and messages[0].get("role") == "system":
                system_message = messages[0]["content"]

            # Prepare messages
            processed_messages = self._prepare_messages(user_message, system_message)
            
            # Process with OpenAI
            self.logger.debug(f"Processing request with model: {model_id}")
            response = await self._process_with_openai(processed_messages, model_id)

            if self.valves.structured_output:
                return response

            return response

        except Exception as e:
            self.logger.error(f"Error in pipeline: {str(e)}")
            raise

    def _verify_response(self, response: str) -> bool:
        required_sections = [
            "Initial Analysis",
            "Step-by-Step Reasoning",
            "Solution Development",
            "Verification & Conclusion"
        ]
        return all(section.lower() in response.lower() for section in required_sections)

    async def _retry_with_clarification(self, messages: List[dict], model: str, attempt: int = 1) -> str:
        if attempt > 3:
            raise Exception("Failed to get a properly structured response after 3 attempts")

        clarification = (
            "Please ensure your response follows the sequential thinking structure "
            "with all four sections clearly labeled."
        )
        
        messages.append({"role": "user", "content": clarification})
        response = await self._process_with_openai(messages, model)
        
        if self._verify_response(response):
            return response
            
        return await self._retry_with_clarification(messages, model, attempt + 1)