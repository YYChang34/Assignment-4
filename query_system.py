"""Minimal KG query template for Assignment 4.

Keep these APIs unchanged for auto-test:
- generate_text(messages, max_new_tokens=220)
- get_relevant_articles(question)
- generate_answer(question, rule_results)

Keep Rule fields aligned with build_kg output:
rule_id, type, action, result, art_ref, reg_name
"""

import os
import re
from typing import Any

from neo4j import GraphDatabase
from dotenv import load_dotenv

from llm_loader import load_local_llm, get_tokenizer, get_raw_pipeline


# ========== 0) Initialization ==========
load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = (
    os.getenv("NEO4J_USER", "neo4j"),
    os.getenv("NEO4J_PASSWORD", "password"),
)

# Avoid local proxy settings interfering with model/Neo4j access.
for key in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
    if key in os.environ:
        del os.environ[key]


try:
    driver = GraphDatabase.driver(URI, auth=AUTH)
    driver.verify_connectivity()
except Exception as e:
    print(f"Neo4j connection warning: {e}")
    driver = None


# ========== Entity extraction helpers ==========

_STOPWORDS: set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "what", "how",
    "when", "where", "which", "who", "whom", "whose", "why", "if",
    "that", "this", "these", "those", "in", "on", "at", "by", "for",
    "with", "about", "of", "to", "from", "and", "or", "but", "not",
    "no", "i", "my", "me", "we", "our", "you", "your", "it", "its",
    "student", "students", "regulation", "regulations", "rule", "rules",
    "article", "articles", "ncu",
}

# Smaller stopword set used only for fulltext search term extraction.
# Keeps domain nouns like "student", "id", "penalty" that are useful for Neo4j queries.
_SEARCH_STOPWORDS: set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "what", "how",
    "when", "where", "which", "who", "whom", "whose", "why", "if",
    "that", "this", "these", "those", "in", "on", "at", "by", "for",
    "with", "about", "of", "to", "from", "and", "or", "but", "not",
    "no", "i", "my", "me", "we", "our", "you", "your", "it", "its",
    "many", "much", "more", "some", "any", "all", "each", "before",
    "after", "they", "them", "their", "he", "she", "his", "her",
}

_PENALTY_QKW: set[str] = {
    "penalty", "penalize", "punish", "consequence", "deduct", "deducted",
    "fine", "dismissed", "expel", "barred", "zero", "nullified", "disciplinary",
}
_REQUIREMENT_QKW: set[str] = {
    "require", "requirement", "minimum", "must", "mandatory", "credit",
    "credits", "semester", "semesters", "gpa", "grade", "qualify",
    "qualification", "eligible",
}
_PROCEDURE_QKW: set[str] = {
    "apply", "application", "process", "fee", "submit", "replace",
    "replacement", "procedure", "steps", "request", "deadline", "working day",
}

_EXAM_KW: set[str] = {
    "exam", "examination", "test", "quiz", "late", "absent", "cheat",
    "cheating", "proctor", "invigilator", "answer", "sheet", "sit", "minutes",
}
_ACADEMIC_KW: set[str] = {
    "course", "credit", "gpa", "grade", "academic", "graduation", "degree",
    "thesis", "dissertation", "major", "minor", "transfer", "withdraw",
}
_ADMIN_KW: set[str] = {
    "id", "identity", "card", "registration", "tuition", "scholarship",
    "dormitory", "suspension", "leave",
}


# ========== 1) Public API (query flow order) ==========
# Order: extract_entities -> build_typed_cypher -> get_relevant_articles -> generate_answer

def generate_text(messages: list[dict[str, str]], max_new_tokens: int = 220) -> str:
    """
    Call local HF model via chat template + raw pipeline.

    Interface:
    - Input:
      - messages: list[dict[str, str]] (chat messages with role/content)
      - max_new_tokens: int
    - Output:
      - str (model generated text, no JSON guarantee)
    """
    tok = get_tokenizer()
    pipe = get_raw_pipeline()
    if tok is None or pipe is None:
        load_local_llm()
        tok = get_tokenizer()
        pipe = get_raw_pipeline()
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return pipe(prompt, max_new_tokens=max_new_tokens)[0]["generated_text"].strip()


def extract_entities(question: str) -> dict[str, Any]:
    """Parse question into {question_type, subject_terms, aspect} via keyword matching."""
    q = question.lower()
    words = re.findall(r"\b\w+\b", q)

    if any(kw in q for kw in _PENALTY_QKW):
        question_type = "penalty"
    elif any(kw in q for kw in _REQUIREMENT_QKW):
        question_type = "requirement"
    elif any(kw in q for kw in _PROCEDURE_QKW):
        question_type = "procedure"
    else:
        question_type = "general"

    subject_terms = [w for w in words if w not in _STOPWORDS and len(w) >= 3]

    if any(kw in q for kw in _EXAM_KW):
        aspect = "exam"
    elif any(kw in q for kw in _ACADEMIC_KW):
        aspect = "academic"
    elif any(kw in q for kw in _ADMIN_KW):
        aspect = "admin"
    else:
        aspect = "general"

    return {
        "question_type": question_type,
        "subject_terms": subject_terms,
        "aspect": aspect,
    }


def build_typed_cypher(entities: dict[str, Any]) -> tuple[str, str]:
    """Return (typed_query, broad_query) Cypher strings using fulltext indexes."""
    question_type = entities.get("question_type", "general")

    cypher_typed = ""
    if question_type != "general":
        cypher_typed = """
        CALL db.index.fulltext.queryNodes('rule_idx', $search_terms)
        YIELD node, score
        WITH node, score
        WHERE node.type = $rule_type
        RETURN node.rule_id AS rule_id, node.type AS type,
               node.action AS action, node.result AS result,
               node.art_ref AS art_ref, node.reg_name AS reg_name,
               score
        ORDER BY score DESC
        LIMIT 10
        """

    cypher_broad = """
    CALL db.index.fulltext.queryNodes('rule_idx', $search_terms)
    YIELD node, score
    RETURN node.rule_id AS rule_id, node.type AS type,
           node.action AS action, node.result AS result,
           node.art_ref AS art_ref, node.reg_name AS reg_name,
           score
    ORDER BY score DESC
    LIMIT 10
    """

    return cypher_typed, cypher_broad


def _build_search_terms(subject_terms: list[str], question: str) -> str:
    # Use the full question's words minus search stopwords so domain nouns
    # like "student", "id", "card", "penalty" are preserved.
    words = re.findall(r"\b\w+\b", question.lower())
    terms = [w for w in words if w not in _SEARCH_STOPWORDS and len(w) >= 2]
    if not terms:
        terms = subject_terms[:8]
    # Escape Lucene special characters to avoid parse errors
    safe: list[str] = []
    for t in terms:
        cleaned = re.sub(r'[+\-&|!(){}\[\]^"~*?:\\/]', " ", t).strip()
        if cleaned:
            safe.append(cleaned)
    return " ".join(safe[:10]) if safe else question[:100]


def get_relevant_articles(question: str) -> list[dict[str, Any]]:
    """Run typed → broad → article_content_idx fallback retrieval; return merged rule dicts."""
    if driver is None:
        return []

    entities = extract_entities(question)
    typed_query, broad_query = build_typed_cypher(entities)

    subject_terms: list[str] = entities.get("subject_terms", [])
    question_type: str = entities.get("question_type", "general")
    search_terms = _build_search_terms(subject_terms, question)

    results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def _run(cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        if not cypher.strip():
            return []
        try:
            with driver.session() as session:
                return [dict(r) for r in session.run(cypher, **params)]
        except Exception:
            return []

    def _merge(records: list[dict[str, Any]]) -> None:
        for r in records:
            rid = r.get("rule_id", "")
            if rid and rid not in seen_ids:
                seen_ids.add(rid)
                results.append(r)

    base_params = {"search_terms": search_terms, "rule_type": question_type}

    if typed_query.strip():
        _merge(_run(typed_query, base_params))

    if len(results) < 3:
        _merge(_run(broad_query, base_params))

    if len(results) < 3:
        fallback = """
        CALL db.index.fulltext.queryNodes('article_content_idx', $search_terms)
        YIELD node AS a, score
        MATCH (a)-[:CONTAINS_RULE]->(ru:Rule)
        RETURN ru.rule_id AS rule_id, ru.type AS type,
               ru.action AS action, ru.result AS result,
               ru.art_ref AS art_ref, ru.reg_name AS reg_name,
               score
        ORDER BY score DESC
        LIMIT 5
        """
        _merge(_run(fallback, {"search_terms": search_terms}))

    return results[:5]


def generate_answer(question: str, rule_results: list[dict[str, Any]]) -> str:
    """Generate a grounded answer from retrieved rules using the local LLM."""
    if not rule_results:
        return "Insufficient rule evidence to answer this question."

    context_parts: list[str] = []
    for i, rule in enumerate(rule_results[:3], 1):
        art_ref = rule.get("art_ref", "?")
        reg_name = rule.get("reg_name", "")
        action = rule.get("action", "")
        result = rule.get("result", "")
        if action == result:
            context_parts.append(f"[{i}] Article {art_ref} ({reg_name}): {action}")
        else:
            context_parts.append(f"[{i}] Article {art_ref} ({reg_name}): {action} -> {result}")

    context = "\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an NCU regulation assistant. "
                "Answer ONLY based on the provided regulation excerpts below. "
                "Be direct and concise (1-3 sentences). "
                "Cite the article number. "
                "Do not add information not present in the excerpts."
            ),
        },
        {
            "role": "user",
            "content": f"Regulation excerpts:\n{context}\n\nQuestion: {question}",
        },
    ]

    raw = generate_text(messages, max_new_tokens=150)
    return raw.strip() or "Insufficient rule evidence to answer this question."


def main() -> None:
    """Interactive CLI (provided scaffold)."""
    if driver is None:
        return

    load_local_llm()

    print("=" * 50)
    print("NCU Regulation Assistant")
    print("=" * 50)
    print("Try: 'What is the penalty for forgetting student ID?'")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_q = input("\nUser: ").strip()
            if not user_q:
                continue
            if user_q.lower() in {"exit", "quit"}:
                print("Bye!")
                break

            results = get_relevant_articles(user_q)
            answer = generate_answer(user_q, results)
            print(f"Bot: {answer}")

        except KeyboardInterrupt:
            print("\nBye!")
            break
        except NotImplementedError as e:
            print(f"Not implemented: {e}")
            break
        except Exception as e:
            print(f"Error: {e}")

    driver.close()


if __name__ == "__main__":
    main()
