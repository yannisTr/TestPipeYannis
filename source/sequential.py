"""
Title: Sequential Thinking Pipeline
Author: Assistant
Date: 2024-12-22
Version: 1.3
License: MIT
Description: A pipeline that combines structured thinking with step-by-step reasoning.
Requirements: pydantic>=2.0.0, open_webui>=0.1.0
"""

import logging
from typing import List, Dict, Any, Optional, Generator, Iterator
from pydantic import BaseModel, Field
import json

from open_webui.apps.openai import main as openai
from open_webui.constants import TASKS
from open_webui.utils.misc import add_or_update_system_message, get_system_message

class Pipeline:
    """Pipeline principale pour le raisonnement séquentiel"""
    
    __model__: str
    
    class Valves(BaseModel):
        temperature: float = Field(default=0.7)
        max_steps: int = Field(default=6)
        min_steps: int = Field(default=3)
        thinking_mode: bool = Field(default=True)
        stream_default: bool = Field(default=True)

    def __init__(self):
        self.name = "sequential-thinking"
        self.logger = logging.getLogger(self.name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.valves = self.Valves()
        self.file_handler = True
        
        # Prompt de base
        self.thinking_prompt = r"""<thinking_protocol>
Instructions pour le raisonnement structuré:
1. Analyser la requête
2. Générer les étapes appropriées
3. Raisonner méthodiquement
4. Format des étapes
5. Guidelines
</thinking_protocol>"""

    def pipes(self) -> List[Dict[str, str]]:
        """Liste des modèles disponibles"""
        openai.get_all_models()
        models = openai.app.state.MODELS
        return [
            {"id": "{}-{}".format(self.name, key), 
             "name": "{} {}".format(self.name, models[key]['name'])}
            for key in models
        ]

    def resolve_model(self, body: dict) -> str:
        model_id = body.get("model", "")
        if "." in model_id:
            without_pipe = ".".join(model_id.split(".")[1:])
            return without_pipe.replace("{}-".format(self.name), "")
        return model_id

    async def get_completion(self, model: str, messages: List[dict], stream: bool = False) -> Any:
        response = await openai.generate_chat_completion(
            {
                "model": model,
                "messages": messages,
                "stream": stream,
                "temperature": self.valves.temperature
            }
        )
        
        if not stream:
            try:
                return response["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                self.logger.error("ResponseError: unable to extract content")
                return ""
        return response

    async def create_thinking_steps(self, content: str, model: str) -> Any:
        prompt = r"""Analysez la requête et créez des étapes de raisonnement. Format JSON requis:
{
    "steps": [
        {
            "title": "Titre de l'étape",
            "description": "Description détaillée"
        }
    ]
}"""
        
        messages = [{"role": "user", "content": prompt + "\n\nRequête:\n" + content}]
        response = await self.get_completion(model, messages, stream=False)
        
        try:
            steps_data = json.loads(response)
            return steps_data
        except Exception as e:
            self.logger.error("Error parsing LLM response: " + str(e))
            return {
                "steps": [
                    {"title": "Analyse", "description": "Analyse de la requête"},
                    {"title": "Solution", "description": "Développement de la réponse"},
                    {"title": "Conclusion", "description": "Synthèse finale"}
                ]
            }

    async def execute_step(self, step: Dict[str, str], model: str, messages: List[dict]) -> str:
        prompt = r"""Étape: {title}
Description: {description}

Analysez cette étape et fournissez:
1. Votre raisonnement
2. Une conclusion""".format(**step)

        messages = messages + [{"role": "user", "content": prompt}]
        return await self.get_completion(model, messages, stream=False)

    async def process_thinking(self, thinking_steps: Dict[str, List[Dict[str, str]]], model: str, messages: List[dict]) -> str:
        self.logger.info("Processing thinking steps")
        results = []
        
        for step in thinking_steps["steps"]:
            step_result = await self.execute_step(step, model, messages)
            results.append("### " + step["title"] + "\n" + step_result)
            messages.append({"role": "assistant", "content": step_result})

        prompt = r"""Synthétisez les résultats et fournissez une conclusion finale."""
        messages.append({"role": "user", "content": prompt})
        conclusion = await self.get_completion(model, messages, stream=False)
        
        all_results = results + ["\n### Conclusion", conclusion]
        return "\n\n".join(all_results)

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__=None,
        __task__=None,
        __model__=None,
    ) -> str | Generator | Iterator:
        self.logger.info("Processing request with " + self.name)
        
        model = self.resolve_model(body)
        body["model"] = model
        
        if __task__ == TASKS.TITLE_GENERATION:
            content = await self.get_completion(model, body.get("messages"), stream=False)
            return self.name + ": " + content

        system_message = get_system_message(body["messages"])
        if system_message is None:
            system_message = {"role": "system", "content": self.thinking_prompt}
        elif len(system_message["content"]) < 500:
            system_message["content"] = self.thinking_prompt + "\n\n" + system_message["content"]
        
        body["messages"] = add_or_update_system_message(system_message, body["messages"])
        
        if self.valves.thinking_mode:
            content = body["messages"][-1]["content"]
            thinking_steps = await self.create_thinking_steps(content, model)
            return await self.process_thinking(
                thinking_steps,
                model,
                body["messages"][:-1]
            )
        else:
            if body.get("stream", self.valves.stream_default):
                return await openai.generate_chat_completion(body, user=__user__)
            else:
                return await self.get_completion(model, body["messages"], stream=False)