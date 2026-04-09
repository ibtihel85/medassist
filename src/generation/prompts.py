

def routing_prompt(question: str) -> str:
    """structured output prompting"""
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are a medical AI routing agent. Analyse the question and return ONLY valid JSON.\n"
        "No explanation, no markdown, no preamble — just the JSON object.\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f'Question: "{question}"\n\n'
        "Return this JSON schema exactly:\n"
        "{\n"
        '  "needs_rewrite": <true if question is vague, uses pronouns like \'it\'/\'they\'/\'this\','
        " or is too short to retrieve well; false otherwise>,\n"
        '  "tool": <"quick_definition" if asking for a single medical/statistical term definition;'
        ' "literature_search" for all other questions>,\n'
        '  "reasoning": "<one sentence>"\n'
        "}\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )


def rewrite_prompt(question: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are a medical query optimisation expert. Rewrite vague questions into specific,\n"
        "natural language medical queries suitable for semantic search.\n"
        "IMPORTANT: Use plain English only. No boolean operators (AND, OR, NOT).\n"
        "No brackets, no quotes, no PubMed syntax. Just a clear, specific sentence.\n"
        "Return ONLY valid JSON, no other text.\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f'Vague question: "{question}"\n\n'
        "Return this JSON exactly:\n"
        "{\n"
        '  "rewritten_query": "<plain English, specific medical query — no AND/OR/brackets>",\n'
        '  "reasoning": "<what was vague and how you fixed it>"\n'
        "}\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )


def generation_prompt(question: str, context: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are MedAssist AI, a precise medical research assistant.\n"
        "Answer the question using ONLY the retrieved PubMed abstracts provided.\n"
        "If the abstracts do not contain enough information, set has_enough_info to false.\n"
        "Return ONLY valid JSON, no other text.\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"RETRIEVED ABSTRACTS:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Return this JSON schema exactly:\n"
        "{\n"
        '  "answer": "<factual answer, grounded strictly in the abstracts above>",\n'
        '  "key_findings": ["<finding 1>", "<finding 2>", "<finding 3>"],\n'
        '  "has_enough_info": <true if abstracts adequately address the question, false otherwise>\n'
        "}\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )


def fallback_rag_prompt(question: str, context: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are MedAssist AI. Answer using ONLY the context below.\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )
