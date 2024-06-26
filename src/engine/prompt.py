# Prompts templates

MAP_PROMPT = """
You are a helpful assistant that helps researchers summarize research papers.
Summarize the following chunks of text:

{docs}

Helpful answer:
"""

REDUCE_PROMPT = """The following is a set of summaries:

{docs}

Take these and distill it into a final, consolidated summary of the main themes, in less than {max_summary_tokens}
 tokens.
Helpful Answer:
"""
