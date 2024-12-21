"""
title: Sequential Thinking Pipeline
author: YannisTr
version: 1.0
description: A pipeline for enhancing LLM reasoning through structured sequential thinking
"""

import logging
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
import aiohttp
import os
from fastapi import HTTPException

import asyncio
import threading
import weakref
from contextlib import asynccontextmanager
from threading import Lock
from typing import Optional, ClassVar, Dict, List, Union, Any

class Pipeline:
    _instance: ClassVar[Optional['Pipeline']] = None
    _instances: ClassVar[Dict[int, 'Pipeline']] = weakref.WeakValueDictionary()
    _lock = Lock()
    _session_lock = Lock()
    
    @classmethod
    async def get_instance(cls):
        """Get or create a Pipeline instance (for FastAPI dependency injection)"""
        task_id = id(asyncio.current_task())
        with cls._lock:
            instance = cls._instances.get(task_id)
            if instance is None or not instance.is_initialized:
                instance = cls()
                await instance.on_startup()
                cls._instances[task_id] = instance
            elif instance._session and instance._session.closed:
                await instance.on_startup()
            return instance

    @classmethod
    def get_sync_instance(cls):
        """Synchronous version of get_instance for non-async contexts"""
        thread_id = id(threading.current_thread())
        with cls._lock:
            instance = cls._instances.get(thread_id)
            if instance is None or not instance.is_initialized:
                instance = cls()
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(instance.on_startup())
                else:
                    loop.run_until_complete(instance.on_startup())
                cls._instances[thread_id] = instance
            elif instance._session and instance._session.closed:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(instance.on_startup())
                else:
                    loop.run_until_complete(instance.on_startup())
            return instance

    @classmethod
    @asynccontextmanager
    async def create(cls):
        """Factory method to properly initialize Pipeline with async context"""
        pipeline = cls()
        await pipeline.on_startup()
        try:
            yield pipeline
        finally:
            await pipeline.on_shutdown()

    def __del__(self):
        """Ensure cleanup of resources"""
        try:
            # Clean up session
            if self._session:
                try:
                    loop = asyncio.get_running_loop()
                    if not loop.is_closed():
                        loop.create_task(self.on_shutdown())
                except RuntimeError:
                    # No running event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self.on_shutdown())
                    finally:
                        loop.close()
                        asyncio.set_event_loop(None)
            
            # Clean up instance tracking
            task_id = None
            try:
                task_id = id(asyncio.current_task())
            except RuntimeError:
                thread_id = id(threading.current_thread())
                if thread_id in self._instances:
                    del self._instances[thread_id]
            
            if task_id and task_id in self._instances:
                del self._instances[task_id]
                
        except Exception as e:
            # Log but don't raise during cleanup
            if hasattr(self, 'logger'):
                self.logger.error(f"Error during cleanup: {str(e)}")

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
        self._initialized = False
        self.logger = self._setup_logger()
        self.thinking_prompt = """As an AI assistant, follow these steps for sequential thinking:
1. Initial Analysis
2. Step-by-Step Reasoning
3. Solution Development
4. Verification & Conclusion"""
        
    @property
    def is_initialized(self) -> bool:
        return self._initialized and self._session is not None


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
        if not self._session:
            self._session = aiohttp.ClientSession()
        self._initialized = True

    async def on_shutdown(self):
        self.logger.info(f"Shutting down {self.name}")
        if self._session:
            await self._session.close()

    async def pipe(self, user_message: Optional[str] = None, messages: Optional[List[Dict]] = None, body: Optional[Dict] = None, **kwargs) -> Union[str, Dict]:
        """
        Main pipeline processing method - must be used in async context
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not properly initialized. Use 'async with Pipeline.create()' to create a pipeline instance.")
            
        # Ensure session is available
        with self._session_lock:
            if not self._session or self._session.closed:
                self._session = aiohttp.ClientSession()
        try:
            # Handle title generation request
            if body and body.get("title", False):
                return self.name

            # Get messages from either direct parameter or body
            if messages is None and body:
                messages = body.get("messages", [])

            # Get model from body or use default
            model = body.get("model", self.valves.model) if body else self.valves.model

            # Get the actual user message
            actual_message = user_message
            if not actual_message and messages:
                actual_message = messages[-1].get("content", "") if messages else ""

            # Get system message if present
            system_message = None
            if messages and messages[0].get("role") == "system":
                system_message = messages[0].get("content")

            # Process the message
            response = await self._process_message(actual_message, model, system_message)
            
            return response

        except Exception as e:
            self.logger.error(f"Error in pipeline: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _process_message(self, message: str, model: str, system_message: Optional[str] = None) -> str:
        """Process a single message through the pipeline"""
        if not message:
            return "No message provided"

        with self._session_lock:
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

    async def filter_inlet(self, messages: List[dict], body: Dict) -> tuple[List[dict], Dict]:
        """Pre-process messages and body before pipeline execution"""
        # Ensure session is available for subsequent operations
        with self._session_lock:
            if not self._session or self._session.closed:
                self._session = aiohttp.ClientSession()
        return messages, body

    async def filter_outlet(self, response: Any) -> Any:
        """Post-process pipeline response"""
        # Check session state after processing
        if self._session and self._session.closed:
            with self._session_lock:
                self._session = aiohttp.ClientSession()
        return response
