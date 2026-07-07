"""
Rakshak offline knowledge agent.

A deterministic, fully offline question-answering agent over preprocessed
Rakshak Systems compliance reports. No LLM, no network, no RAG, no caching.

    from rakshak_agent import Agent
    agent = Agent.load("knowledge")
    print(agent.ask("How much was the RCM liability?").text)
"""

from .engine import Agent, Answer
from .knowledge import KnowledgeBase
from .smart_agent import SmartAgent, SmartAnswer
from .llm import (LLMClient, LLMResponse, MockLLMClient, ScriptedLLMClient,
                  DeepSeekClient, ModelCascade, ABSTAIN)
from .distillation import DistillationLogger

# The four flagship "virtual-CA" questions for the UI's preloaded section.
# Closed-loop: all resolve deterministically from the reports + clock ($0).
PRELOADED_QUESTIONS = [
    "What should I prioritise before the deadlines?",
    "Which statutory windows are closing?",
    "What should I do about Pinnacle Advisory?",
    "What is the notice position?",
]


def suggested_questions():
    """Preloaded questions for a frontend (closed-loop, zero-cost)."""
    return list(PRELOADED_QUESTIONS)

__all__ = [
    "Agent", "Answer", "KnowledgeBase",
    "SmartAgent", "SmartAnswer",
    "LLMClient", "LLMResponse", "MockLLMClient", "ScriptedLLMClient",
    "DeepSeekClient", "ModelCascade", "ABSTAIN",
    "DistillationLogger", "PRELOADED_QUESTIONS", "suggested_questions",
]
__version__ = "2.0.0"
