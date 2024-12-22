"""
Title: Sequential Thinking Pipeline
Author: Assistant
Date: 2024-12-22
Version: 1.4
License: MIT
Description:
    A pipeline that combines structured thinking with step-by-step reasoning.
    Uses dynamic step generation and chain-of-thought prompting.
Requirements:
    pydantic~=2.0.0
    openai>=1.0.0
    python-logging>=0.4.9.6
    typing-extensions>=4.5.0
"""

import logging
from typing import List, Dict, Any, Optional, Generator, Iterator
from pydantic import BaseModel, Field, ConfigDict
import json
from openai import OpenAI

# Configuration du logging
logger = logging.getLogger("sequential-thinking")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Modèles de données
class Step(BaseModel):
    """Représente une étape de raisonnement"""
    title: str = Field(..., description="Titre de l'étape")
    description: str = Field(..., description="Description détaillée")

    model_config = ConfigDict(
        extra='forbid',
    )

class ThinkingProcess(BaseModel):
    """Représente le processus de pensée"""
    steps: List[Step] = Field(default_factory=list)
    conclusion: Optional[str] = None

    model_config = ConfigDict(
        extra='forbid',
    )

class Pipeline:
    """Pipeline pour le raisonnement séquentiel"""
    
    class Valves(BaseModel):
        """Configuration de la pipeline"""
        OPENAI_API_KEY: str = Field(default="")
        MODEL_NAME: str = Field(default="gpt-4-turbo-preview")
        temperature: float = Field(default=0.7)
        max_steps: int = Field(default=6)
        min_steps: int = Field(default=3)
        thinking_mode: bool = Field(default=True)

        model_config = ConfigDict(
            extra='forbid',
        )

    def __init__(self):
        """Initialisation de la pipeline"""
        self.name = "sequential-thinking"
        self.valves = self.Valves()
        self.client = None
        self.thinking_prompt = r'''<thinking_protocol>
Instructions pour le raisonnement structuré:
1. Analyser la requête
2. Générer les étapes appropriées
3. Raisonner méthodiquement
4. Format des étapes
5. Guidelines
</thinking_protocol>'''

    async def on_startup(self):
        """Initialisation du client OpenAI"""
        self.client = OpenAI(
            api_key=self.valves.OPENAI_API_KEY,
        )

    async def on_shutdown(self):
        """Nettoyage lors de l'arrêt"""
        pass

    def get_models(self) -> List[Dict[str, str]]:
        """Liste des modèles disponibles"""
        return [
            {"id": f"{self.name}-gpt4", "name": f"{self.name} GPT-4 Turbo"},
            {"id": f"{self.name}-gpt3", "name": f"{self.name} GPT-3.5 Turbo"}
        ]

    def resolve_model(self, body: dict) -> str:
        """Résout l'ID du modèle"""
        model_id = body.get("model", "")
        if "-gpt4" in model_id:
            return "gpt-4-turbo-preview"
        return "gpt-3.5-turbo"

    async def create_thinking_steps(self, content: str, model: str) -> ThinkingProcess:
        """Génère les étapes de réflexion"""
        prompt = r"""Analysez la requête et créez des étapes de raisonnement. Format JSON:
{
    "steps": [
        {
            "title": "Titre de l'étape",
            "description": "Description détaillée"
        }
    ]
}"""
        response = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt + "\n\nRequête:\n" + content}],
            temperature=self.valves.temperature
        )
        
        try:
            steps_data = json.loads(response.choices[0].message.content)
            return ThinkingProcess(steps=[Step(**step) for step in steps_data["steps"]])
        except Exception as e:
            logger.error(f"Error parsing steps: {e}")
            return ThinkingProcess(steps=[
                Step(title="Analyse", description="Analyse de la requête"),
                Step(title="Solution", description="Développement de la réponse"),
                Step(title="Conclusion", description="Synthèse finale")
            ])

    async def execute_step(self, step: Step, model: str, messages: List[dict]) -> str:
        """Exécute une étape de réflexion"""
        prompt = r"""Étape: {title}
Description: {description}

Analysez cette étape et fournissez:
1. Votre raisonnement détaillé
2. Une conclusion""".format(title=step.title, description=step.description)

        response = await self.client.chat.completions.create(
            model=model,
            messages=messages + [{"role": "user", "content": prompt}],
            temperature=self.valves.temperature
        )
        return response.choices[0].message.content

    async def process_thinking(self, thinking_process: ThinkingProcess, model: str, messages: List[dict]) -> str:
        """Traite le processus de réflexion"""
        logger.info("Processing thinking steps")
        results = []
        
        for step in thinking_process.steps:
            step_result = await self.execute_step(step, model, messages)
            results.append(f"### {step.title}\n{step_result}")
            messages.append({"role": "assistant", "content": step_result})

        # Conclusion
        prompt = r"""Synthétisez les résultats et fournissez une conclusion finale."""
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages + [{"role": "user", "content": prompt}],
            temperature=self.valves.temperature
        )
        conclusion = response.choices[0].message.content
        
        return "\n\n".join(results + ["\n### Conclusion", conclusion])

    async def pipe(
        self,
        body: dict,
        messages: List[dict],
    ) -> str:
        """Point d'entrée principal"""
        logger.info(f"Processing request with {self.name}")
        
        model = self.resolve_model(body)
        messages = messages.copy()

        # Ajout du prompt de thinking
        messages.insert(0, {"role": "system", "content": self.thinking_prompt})
        
        if self.valves.thinking_mode:
            content = messages[-1]["content"]
            thinking_process = await self.create_thinking_steps(content, model)
            return await self.process_thinking(
                thinking_process,
                model,
                messages[:-1]
            )
        else:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=self.valves.temperature
            )
            return response.choices[0].message.content