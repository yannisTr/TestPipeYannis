"""
title: Sequential Thinking Pipeline
author: Yannis
repo: https://github.com/Yannis/sequential-thinking-pipeline
version: 1.1
information: Pipeline avancé pour réflexion séquentielle dynamique dans Open WebUI.
"""

import logging
from typing import Generator, Iterator, Optional

from open_webui.apps.openai import main as openai
from open_webui.constants import TASKS
from open_webui.utils.misc import (
    add_or_update_system_message,
    get_system_message,
    pop_system_message
)
from pydantic import BaseModel

# Nom du pipeline
name = "SequentialThinking"

# Initialisation du logger
def setup_logger():
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

logger = setup_logger()

class SequentialThinkingPipe:
    """Classe principale du pipeline"""

    class Valves(BaseModel):
        """Configurations dynamiques du pipeline (si besoin)."""
        reflection_mode: Optional[str] = "auto"  # "auto", "manual", or "off"

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self) -> list[dict[str, str]]:
        """Récupère la liste des modèles disponibles."""
        openai.get_all_models()
        models = openai.app.state.MODELS

        return [
            {"id": f"{name}-{key}", "name": f"{name} {models[key]['name']}"}
            for key in models
        ]

    def get_thinking_prompt(self):
        """Prompt pour activer la réflexion séquentielle."""
        return """<sequential_thinking_protocol>
Claude suit un protocole de réflexion avancée :
1. Décompose chaque requête en étapes.
2. Génère plusieurs hypothèses ou solutions.
3. Révise son raisonnement à chaque étape.
4. Explore des alternatives tout en restant centré sur le problème principal.
5. Formule une réponse claire et structurée après réflexion complète.

Tous les détails de réflexion seront encapsulés dans un bloc `thinking` invisible pour l'utilisateur.
</sequential_thinking_protocol>"""

    def should_initiate_thinking(self, messages: list[dict]) -> bool:
        """Détermine si la réflexion séquentielle doit être activée."""
        user_message = messages[-1]["content"].lower()
        return "think:" in user_message or self.valves.reflection_mode == "auto"

    def adapt_thinking(self, thinking_block: str, new_message: str) -> str:
        """Adapte le raisonnement en fonction des nouveaux messages."""
        return (
            f"{thinking_block}\n\n--- Mise à jour : "
            f"Considérez également cette nouvelle requête : {new_message}."
        )

    def add_branch(self, thinking_block: str, branch_idea: str) -> str:
        """Ajoute une branche alternative au raisonnement."""
        return f"{thinking_block}\n\n--- Exploration alternative : {branch_idea} ---"

    def resolve_model(self, body: dict) -> str:
        """Récupère le modèle spécifique à utiliser."""
        model_id = body.get("model")
        without_pipe = ".".join(model_id.split(".")[1:])
        return without_pipe.replace(f"{name}-", "")

    async def get_completion(self, model: str, messages: list):
        """Récupère une réponse du modèle."""
        response = await openai.generate_chat_completion(
            {"model": model, "messages": messages, "stream": False}
        )
        return self.get_response_content(response)

    def get_response_content(self, response: dict) -> str:
        """Extrait le contenu principal de la réponse."""
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            logger.error(
                f"ResponseError: Unable to extract content from response: {response}"
            )
            return ""

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__=None,
        __task__=None,
        __model__=None,
    ) -> str | Generator | Iterator:
        """Pipeline principal."""
        model = self.resolve_model(body)
        body["model"] = model
        system_message = get_system_message(body["messages"])

        # Activation de la réflexion séquentielle si nécessaire
        if self.should_initiate_thinking(body["messages"]):
            thinking_protocol = self.get_thinking_prompt()
            system_message, body["messages"] = pop_system_message(body["messages"])
            system_message = f"{thinking_protocol}\n\n{system_message['content']}"
            body["messages"] = add_or_update_system_message(system_message, body["messages"])
            logger.debug("Réflexion séquentielle activée.")

        current_thinking = get_system_message(body["messages"])["content"]
        if __task__ == TASKS.TITLE_GENERATION:
            content = await self.get_completion(model, body.get("messages"))
            return f"{name}: {content}"

        # Mise à jour dynamique du raisonnement
        if len(body["messages"]) > 1:
            new_message = body["messages"][-1]["content"]
            current_thinking = self.adapt_thinking(current_thinking, new_message)
            body["messages"] = add_or_update_system_message(current_thinking, body["messages"])

        # Ajout d'une exploration alternative
        branch_idea = "Réfléchissez à un contre-exemple ou à une approche différente."
        current_thinking = self.add_branch(current_thinking, branch_idea)
        body["messages"] = add_or_update_system_message(current_thinking, body["messages"])

        # Génération finale
        return await openai.generate_chat_completion(body, user=__user__)
