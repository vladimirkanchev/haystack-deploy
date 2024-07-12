"""Contain wrappers of prompt components of tne rag pipeline."""
from haystack.components.builders import PromptBuilder

from rag_system.prompts import PROMPT_TEMPLATE


def setup_prompt() -> PromptBuilder:
    """Render a prompt template."""
    return PromptBuilder(template=PROMPT_TEMPLATE)
