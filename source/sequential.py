"""
title: Sequential Thinking Pipeline
author: [Votre nom]
version: 1.0
description: A pipeline for enhancing LLM reasoning through structured sequential thinking
"""

import logging
from typing import Generator, Iterator
from pydantic import BaseModel, Field

# Configuration du logger
def setup_logger():
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.set_name("sequential_thinking")
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger

logger = setup_logger()

class Pipe:
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
- Break down complex problems
- Show your work clearly
- Identify assumptions and limitations

3. Solution Development
- Build on previous steps
- Consider alternatives
- Verify conclusions

4. Output Formatting
- Present conclusions clearly
- Summarize key points
- Note any uncertainties

Follow this process for every response, showing your work at each step."""

    def resolve_model(self, body: dict) -> str:
        model_id = body.get("model")
        return model_id.split("sequential_thinking-")[-1]

    async def pipe(self, body: dict, __user__: dict, __event_emitter__=None) -> str | Generator | Iterator:
        try:
            # Récupérer le message de l'utilisateur
            user_message = body["messages"][-1]["content"]
            
            # Ajouter le prompt de pensée séquentielle
            system_message = {
                "role": "system",
                "content": self.thinking_prompt
            }
            
            # Structurer le message pour le LLM
            structured_prompt = f"""
            Question/Task: {user_message}

            Think through this step-by-step:

            1. Initial Understanding:
            [Your initial analysis here]

            2. Breaking Down Components:
            [List key elements]

            3. Processing Steps:
            [Show your reasoning]

            4. Verification:
            [Check your work]

            5. Conclusion:
            [Final answer]

            Remember to show all your work and thinking process.
            """

            # Mettre à jour les messages
            body["messages"] = [system_message] + body["messages"][:-1] + [{
                "role": "user",
                "content": structured_prompt
            }]

            logger.debug(f"Processed prompt: {structured_prompt[:200]}...")

            # Retourner la réponse structurée
            return await self.generate_response(body)

        except Exception as e:
            logger.error(f"Error in sequential thinking pipeline: {str(e)}")
            raise

    async def generate_response(self, body: dict):
        # Ici, vous implementeriez la logique pour générer la réponse
        # Cela dépendrait de votre backend LLM (OpenAI, local, etc.)
        pass