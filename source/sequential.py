"""
title: Sequential Thinking Pipeline
author: YannisTr
version: 1.0
description: A pipeline for enhancing LLM reasoning through structured sequential thinking
"""

import logging
from typing import Dict, List, Optional, Union, Any, Callable
from pydantic import BaseModel, Field
import aiohttp
import os
from fastapi import HTTPException, Depends
from starlette.requests import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

import asyncio
import functools
from contextlib import asynccontextmanager
from typing import Optional, List, Union, Any, Callable, TypeVar, Awaitable

T = TypeVar('T')

def ensure_async(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """Decorator to ensure a coroutine is properly awaited in FastAPI context"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        # Get the current task
        current_task = asyncio.current_task()
        
        # Check if we're in a FastAPI request context
        request = None
        for arg in args:
            if isinstance(arg, Request):
                request = arg
                break
        if not request:
            for arg in kwargs.values():
                if isinstance(arg, Request):
                    request = arg
                    break
        
        # If we have a request, ensure proper pipeline instance
        if request and hasattr(request.state, 'pipeline'):
            pipeline = request.state.pipeline
            # Track the task
            if not hasattr(request.state, 'pipeline_tasks'):
                request.state.pipeline_tasks = set()
            request.state.pipeline_tasks.add(current_task)
            try:
                return await func(*args, **kwargs)
            finally:
                request.state.pipeline_tasks.remove(current_task)
        
        # No request context, just await normally
        return await func(*args, **kwargs)
    return wrapper

class Pipeline:
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

    @classmethod
    async def get_instance(cls, request: Request = None) -> 'Pipeline':
        """Get Pipeline instance for FastAPI dependency injection"""
        instance = cls()
        if not instance._initialized:
            await instance.on_startup()
        if request:
            # Store instance in request state for lifecycle management
            request.state.pipeline = instance
            # Ensure cleanup when request is done
            async def cleanup():
                await instance.on_shutdown()
            request.add_event_handler("shutdown", cleanup)
        return instance

    @classmethod
    def get_dependency(cls):
        """Get FastAPI dependency"""
        return Depends(cls.get_instance)

    @classmethod
    def get_middleware(cls) -> Callable:
        """Get FastAPI middleware for proper async context handling"""
        class PipelineMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
                # Create a new task-specific pipeline instance
                pipeline = await cls.get_instance(request)
                request.state.pipeline_task_id = id(asyncio.current_task())
                
                try:
                    # Ensure we're in an async context
                    asyncio.current_task()
                    response = await call_next(request)
                    return response
                except Exception as e:
                    # Log any errors but ensure cleanup
                    pipeline.logger.error(f"Error in middleware: {str(e)}")
                    raise
                finally:
                    try:
                        # Always ensure proper cleanup
                        if hasattr(request.state, 'pipeline'):
                            await request.state.pipeline.on_shutdown()
                            delattr(request.state, 'pipeline')
                    except Exception as e:
                        # Log cleanup errors but don't raise
                        pipeline.logger.error(f"Error during cleanup: {str(e)}")
        
        return PipelineMiddleware

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
        if self._session and not self._session.closed:
            try:
                loop = asyncio.get_running_loop()
                if not loop.is_closed():
                    loop.create_task(self.on_shutdown())
            except RuntimeError:
                pass  # No running event loop, session will be garbage collected

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

    @ensure_async
    async def pipe(self, user_message: Optional[str] = None, messages: Optional[List[Dict]] = None, body: Optional[Dict] = None, request: Request = None, **kwargs) -> Union[str, Dict]:
        """
        Main pipeline processing method - must be used in async context
        """
        # Get current task for tracking
        current_task = asyncio.current_task()
        task_id = id(current_task)

        try:
            if not self.is_initialized:
                raise RuntimeError("Pipeline not properly initialized. Use 'async with Pipeline.create()' to create a pipeline instance.")
                
            # Get instance from request state if available
            if request and hasattr(request.state, 'pipeline'):
                pipeline = request.state.pipeline
                if pipeline._session and not pipeline._session.closed:
                    self._session = pipeline._session
                # Track this task
                if not hasattr(request.state, 'pipeline_tasks'):
                    request.state.pipeline_tasks = {task_id}
                else:
                    request.state.pipeline_tasks.add(task_id)
            
            # Ensure session is available
            if not self._session or self._session.closed:
                self._session = aiohttp.ClientSession()
                if request:
                    request.state.pipeline = self

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
        finally:
            # Clean up task tracking if needed
            if request and hasattr(request.state, 'pipeline_tasks'):
                request.state.pipeline_tasks.discard(task_id)

    async def _process_message(self, message: str, model: str, system_message: Optional[str] = None) -> str:
        """Process a single message through the pipeline"""
        if not message:
            return "No message provided"

        if not self._session or self._session.closed:
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

    async def filter_inlet(self, messages: List[dict], body: Dict, request: Request = None) -> tuple[List[dict], Dict]:
        """Pre-process messages and body before pipeline execution"""
        # Get instance from request state if available
        if request and hasattr(request.state, 'pipeline'):
            pipeline = request.state.pipeline
            if pipeline._session and not pipeline._session.closed:
                self._session = pipeline._session
        
        # Ensure session is available
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()
            if request:
                request.state.pipeline = self
        return messages, body

    async def filter_outlet(self, response: Any, request: Request = None) -> Any:
        """Post-process pipeline response"""
        # Get instance from request state if available
        if request and hasattr(request.state, 'pipeline'):
            pipeline = request.state.pipeline
            if pipeline._session and not pipeline._session.closed:
                self._session = pipeline._session
        
        # Ensure session is available
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()
            if request:
                request.state.pipeline = self
        return response
