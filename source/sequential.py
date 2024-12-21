"""
title: Sequential Thinking Pipeline
author: YannisTr
version: 1.0
description: A pipeline for enhancing LLM reasoning through structured sequential thinking
"""

import logging
import asyncio
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
import aiohttp
import os
from fastapi import HTTPException

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
        self._session: Optional[aiohttp.ClientSession] = None
        self.logger = self._setup_logger()
        self.thinking_prompt = """As an AI assistant, follow these steps for sequential thinking:
1. Initial Analysis
2. Step-by-Step Reasoning
3. Solution Development
4. Verification & Conclusion"""

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

    async def on_startup(self):
        self.logger.info(f"Starting {self.name}")
        if not self.valves.openai_api_key:
            self.logger.warning("OpenAI API key not set!")
        self._session = aiohttp.ClientSession()

    async def on_shutdown(self):
        self.logger.info(f"Shutting down {self.name}")
        if self._session:
            await self._session.close()

    def pipe(self, messages: List[Dict], body: Dict) -> str:
        """Synchronous wrapper for async pipeline processing"""
        if body.get("title", False):
            return self.name

        # Create a task in the event loop to process the message
        loop = asyncio.get_event_loop()
        task = loop.create_task(self._async_process(messages, body))
        
        try:
            # Wait for the task to complete with a timeout
            return loop.run_until_complete(task)
        except Exception as e:
            self.logger.error(f"Error in pipeline: {str(e)}")
            raise

    async def _async_process(self, messages: List[Dict], body: Dict) -> str:
        """Asynchronous processing of messages"""
        try:
            # Get the model from body or use default
            model = body.get("model", self.valves.model)

            # Get the actual message
            user_message = messages[-1].get("content", "") if messages else ""

            # Get system message if present
            system_message = messages[0].get("content") if messages and messages[0].get("role") == "system" else None

            # Process the message
            return await self._process_message(user_message, model, system_message)

        except Exception as e:
            self.logger.error(f"Error in async processing: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _process_message(self, message: str, model: str, system_message: Optional[str] = None) -> str:
        """Process a single message through the OpenAI API"""
        if not message:
            return "No message provided"

        if not self._session:
            self._session = aiohttp.ClientSession()

        headers = {
            "Authorization": f"Bearer {self.valves.openai_api_key}",
            "Content-Type": "application/json"
        }

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        else:
            messages.append({"role": "system", "content": self.thinking_prompt})

        messages.append({"role": "user", "content": message})

        try:
            async with self._session.post(
                f"{self.valves.openai_api_base}/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": self.valves.temperature
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API error: {error_text}")
                
                result = await response.json()
                return result["choices"][0]["message"]["content"]

        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            raise

    def filter_inlet(self, messages: List[Dict], body: Dict) -> tuple[List[Dict], Dict]:
        """Pre-process messages and body before pipeline execution"""
        return messages, body

    def filter_outlet(self, response: Any) -> Any:
        """Post-process pipeline response"""
        return response