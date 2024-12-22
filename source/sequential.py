"""
Title: Sequential Thinking Pipeline
Author: Assistant
Date: 2024-12-22
Version: 1.1
License: MIT
Description:
    A pipeline that combines structured thinking with step-by-step reasoning.
    Uses both chain-of-thought prompting and dynamic step generation to break down
    complex problems into manageable steps.
Requirements: 
    - pydantic>=2.0.0
    - open_webui>=0.1.0
    - python-logging>=0.4.9.6
    - typing-extensions>=4.5.0
    - python-json>=3.2

Features:
    - Dynamic step generation using LLM
    - Structured thinking process
    - Stream support
    - System message integration
    - Configurable step count and parameters
"""

import logging
from typing import List, Dict, Any, Optional, Generator, Iterator
from pydantic import BaseModel, Field
from datetime import datetime

from open_webui.apps.openai import main as openai
from open_webui.constants import TASKS
from open_webui.utils.misc import add_or_update_system_message, get_system_message, pop_system_message

# Configuration du logging
def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
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

# Modèles de données
class Step(BaseModel):
    """Représente une étape de raisonnement"""
    title: str = Field(..., description="Titre de l'étape")
    description: str = Field(..., description="Description détaillée de l'étape")
    reasoning: Optional[str] = Field(None, description="Raisonnement pour cette étape")
    result: Optional[str] = Field(None, description="Résultat de l'étape")

class ThinkingProcess(BaseModel):
    """Représente le processus de pensée complet"""
    steps: List[Step] = Field(default_factory=list)
    conclusion: Optional[str] = Field(None)

class Pipe:
    """Pipeline principale pour le raisonnement séquentiel"""
    
    __model__: str  # Requis pour la détection
    
    class Valves(BaseModel):
        """Configuration de la pipeline via Valves"""
        pass  # Les valves peuvent être configurées plus tard si nécessaire
    
    def __init__(self):
        """Initialisation de la pipeline"""
        self.name = "sequential-thinking"
        self.logger = setup_logger(self.name)
        
        # Paramètres de configuration
        self.max_steps = 6
        self.min_steps = 3
        self.temperature = 0.7
        self.thinking_mode = True
        self.stream_default = True
        
        # Flag pour la gestion des fichiers (optionnel)
        self.file_handler = True
        
        # Initialisation des valves
        self.valves = self.Valves()
        
        # Le prompt de base pour le thinking
        self.thinking_prompt = '''
<thinking_protocol>
Instructions pour le raisonnement structuré :

1. Analyser la requête
- Comprendre le contexte et les besoins
- Identifier les contraintes et objectifs

2. Générer les étapes appropriées
- Adapter le nombre d'étapes à la complexité
- Assurer une progression logique
- Garder chaque étape focalisée

3. Raisonner méthodiquement
- Analyser chaque étape en détail
- Justifier les décisions
- Maintenir la cohérence

4. Format des étapes
- Titre clair et concis
- Description détaillée des objectifs
- Raisonnement explicite
- Résultats vérifiables

5. Guidelines
- Rester factuel et précis
- Éviter les suppositions
- Être exhaustif dans l'analyse
- Adapter la profondeur à la complexité
</thinking_protocol>
'''

    def get_models(self) -> List[Dict[str, str]]:
        """Récupère la liste des modèles disponibles"""
        try:
            openai.get_all_models()
            models = openai.app.state.MODELS
            
            out = [
                {"id": f"{self.name}-{key}", "name": f"{self.name} {models[key]['name']}"}
                for key in models
            ]
            self.logger.debug(f"Available models: {out}")
            return out
        except Exception as e:
            self.logger.error(f"Error getting models: {e}")
            return []

    def resolve_model(self, body: dict) -> str:
        """Résout l'ID du modèle"""
        try:
            model_id = body.get("model", "")
            if "." in model_id:
                without_pipe = ".".join(model_id.split(".")[1:])
                return without_pipe.replace(f"{self.name}-", "")
            return model_id
        except Exception as e:
            self.logger.error(f"Error resolving model: {e}")
            return model_id

    async def get_completion(self, model: str, messages: List[dict], stream: bool = False) -> Any:
        """Obtient une completion du modèle (similaire à Pipeline 1)"""
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
                self.logger.error(f'ResponseError: unable to extract content')
                return ""
        return response

    async def create_thinking_steps(self, content: str, model: str) -> ThinkingProcess:
        """Génère dynamiquement les étapes de pensée via le LLM"""
        step_generation_prompt = f"""
Analysez la requête suivante et créez des étapes de raisonnement appropriées :

{content}

Générez entre {self.valves.min_steps} et {self.valves.max_steps} étapes de raisonnement pertinentes.
Chaque étape doit avoir :
1. Un titre court et descriptif
2. Une description détaillée de ce qui doit être analysé ou résolu

Format JSON requis :
{{
    "steps": [
        {{
            "title": "Titre de l'étape",
            "description": "Description détaillée"
        }},
        ...
    ]
}}
"""
        
        response = await self.get_completion(
            model,
            [{"role": "user", "content": step_generation_prompt}],
            stream=False
        )
        
        try:
            import json
            steps_data = json.loads(response)
            steps = [Step(**step_data) for step_data in steps_data["steps"]]
            return ThinkingProcess(steps=steps)
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            fallback_steps = [
                Step(
                    title="Analyse du problème",
                    description="Compréhension et décomposition de la requête"
                ),
                Step(
                    title="Élaboration de la solution",
                    description="Développement d'une réponse détaillée"
                ),
                Step(
                    title="Vérification et conclusion",
                    description="Validation et synthèse des résultats"
                )
            ]
            return ThinkingProcess(steps=fallback_steps)

    async def execute_step(self, step: Step, model: str, messages: List[dict]) -> str:
        """Exécute une étape de raisonnement"""
        step_prompt = f"""
Étape : {step.title}
Description : {step.description}

Analysez cette étape et fournissez :
1. Votre raisonnement détaillé
2. Une conclusion pour cette étape
"""
        messages = messages + [{"role": "user", "content": step_prompt}]
        return await self.get_completion(model, messages, stream=False)

    async def evaluate_need_for_additional_steps(self, current_results: List[str], messages: List[dict], model: str) -> Optional[List[Step]]:
        """Évalue si des étapes supplémentaires sont nécessaires"""
        evaluation_prompt = f"""
Analysez les étapes de raisonnement effectuées jusqu'à présent :

{'\n'.join(current_results)}

Déterminez si des étapes supplémentaires sont nécessaires pour une réponse complète.
Si oui, spécifiez les étapes additionnelles en format JSON comme suit :
{
    "needs_additional_steps": true/false,
    "steps": [
        {
            "title": "Titre de l'étape",
            "description": "Description détaillée"
        }
    ],
    "reasoning": "Explication de pourquoi ces étapes sont nécessaires"
}
"""
        try:
            response = await self.get_completion(model, messages + [{"role": "user", "content": evaluation_prompt}], stream=False)
            evaluation = json.loads(response)
            
            if evaluation.get("needs_additional_steps", False):
                self.logger.info(f"Additional steps needed: {evaluation.get('reasoning')}")
                return [Step(**step) for step in evaluation.get("steps", [])]
            return None
        except Exception as e:
            self.logger.error(f"Error evaluating need for additional steps: {e}")
            return None

    async def process_thinking(self, thinking_process: ThinkingProcess, model: str, messages: List[dict]) -> str:
        """Traite l'ensemble du processus de pensée avec ajout dynamique d'étapes"""
        self.logger.info("Processing thinking steps")
        
        results = []
        current_steps = thinking_process.steps.copy()
        processed_steps = 0
        
        while processed_steps < len(current_steps):
            # Traitement des étapes actuelles
            while processed_steps < len(current_steps):
                step = current_steps[processed_steps]
                step_result = await self.execute_step(step, model, messages)
                results.append(f"### {step.title}\n{step_result}")
                messages.append({"role": "assistant", "content": step_result})
                processed_steps += 1
            
            # Évaluation du besoin d'étapes supplémentaires
            additional_steps = await self.evaluate_need_for_additional_steps(results, messages, model)
            if additional_steps:
                self.logger.info(f"Adding {len(additional_steps)} new steps")
                current_steps.extend(additional_steps)

        # Conclusion finale
        conclusion_prompt = """
Analysez tous les résultats précédents et fournissez une conclusion finale qui :
1. Synthétise les points clés de chaque étape
2. Tire des conclusions générales
3. Répond à la question ou problématique initiale
4. Justifie la complétude de l'analyse
"""
        messages.append({"role": "user", "content": conclusion_prompt})
        conclusion = await self.get_completion(model, messages, stream=False)
        
        return "\n\n".join(results + ["\n### Conclusion", conclusion])

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__=None,
        __task__=None,
        __model__=None,
    ) -> str | Generator | Iterator:
        """Point d'entrée principal de la pipeline"""
        self.logger.info(f"Processing request with {self.name}")
        
        model = self.resolve_model(body)
        body["model"] = model
        
        # Gestion de la génération de titre
        if __task__ == TASKS.TITLE_GENERATION:
            content = await self.get_completion(model, body.get("messages"), stream=False)
            return f"{self.name}: {content}"

        # Gestion du message système
        system_message = get_system_message(body["messages"])
        if system_message is None:
            system_message = {"role": "system", "content": self.thinking_prompt}
        elif len(system_message["content"]) < 500:
            original_content = system_message["content"]
            system_message["content"] = f"{self.thinking_prompt}\n\n{original_content}"
        
        body["messages"] = add_or_update_system_message(system_message, body["messages"])
        
        # Mode thinking ou normal
        if self.valves.thinking_mode:
            content = body["messages"][-1]["content"]
            thinking_process = await self.create_thinking_steps(content, model)
            response = await self.process_thinking(
                thinking_process,
                model,
                body["messages"][:-1]
            )
            return response
        else:
            # Support du streaming si demandé
            if body.get("stream", self.valves.stream_default):
                return await openai.generate_chat_completion(body, user=__user__)
            else:
                response = await self.get_completion(
                    model,
                    body["messages"],
                    stream=False
                )
                return response