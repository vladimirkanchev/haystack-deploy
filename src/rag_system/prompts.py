"""Build teplate for chosen llm model."""
from haystack.components.builders import PromptBuilder

PROMPT_TEMPLATE = """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""

prompt_builder = PromptBuilder(template=PROMPT_TEMPLATE)
