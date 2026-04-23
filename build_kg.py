"""KG builder for Assignment 4.

Keep this contract unchanged:
- Graph: (Regulation)-[:HAS_ARTICLE]->(Article)-[:CONTAINS_RULE]->(Rule)
- Article: number, content, reg_name, category
- Rule: rule_id, type, action, result, art_ref, reg_name
- Fulltext indexes: article_content_idx, rule_idx
- SQLite file: ncu_regulations.db
"""

import os
import re
import sqlite3
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase


# ========== 0) Initialization ==========
load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = (
    os.getenv("NEO4J_USER", "neo4j"),
    os.getenv("NEO4J_PASSWORD", "password"),
)

_PENALTY_KWS = {
    "deduct", "penalty", "zero", "dismiss", "expel", "disciplinary",
    "ntd", "barred", "prohibited", "not allow", "nullif", "disqualif",
}
_REQUIREMENT_KWS = {
    "shall", "must", "required", "minimum", "at least",
    "no fewer", "not fewer", "credits", "semesters",
}
_PROCEDURE_KWS = {
    "application", "apply", "submit", "working day",
    "replace", "replacement", "fee", "process", "may apply",
}


def _classify_type(content: str) -> str:
    c = content.lower()
    if any(kw in c for kw in _PENALTY_KWS):
        return "penalty"
    if any(kw in c for kw in _REQUIREMENT_KWS):
        return "requirement"
    if any(kw in c for kw in _PROCEDURE_KWS):
        return "procedure"
    return "general"


def _split_action_result(content: str) -> tuple[str, str]:
    """Split article content into (action, result) using legal sentence patterns."""
    for pattern in [
        r'^(.{15,}?(?:if|when|in\s+case|whoever|student[s]?\s+who)[^.;]{5,}[,;])\s*(.{5,})$',
        r'^(.{15,}?(?:late\s+(?:by|for)|after|beyond|more\s+than)[^.;]{5,}[,;])\s*(.{5,})$',
        r'^(.{15,}?(?:shall|must|will|may\s+not)[^.;]{5,})\s*[;]\s*(.{5,})$',
    ]:
        m = re.match(pattern, content, re.IGNORECASE | re.DOTALL)
        if m:
            a, r = m.group(1).strip(), m.group(2).strip()
            if len(a) > 15 and len(r) > 5:
                return a, r

    # Fall back to splitting at first sentence boundary past 40% of content
    threshold = max(30, int(len(content) * 0.4))
    for delim in [". ", "; "]:
        pos = content.find(delim, threshold)
        if 0 < pos < len(content) - 10:
            return content[: pos + 1].strip(), content[pos + 2 :].strip()

    return content, content


_CONDITION_RE = re.compile(
    r"^(?:student[s]?\s+who|whoever|any\s+student|if\s+a|when\s+a|in\s+case)",
    re.IGNORECASE,
)


def extract_entities(article_number: str, reg_name: str, content: str) -> dict[str, Any]:
    """Extract rule facts from article content using regex patterns."""
    content = content.strip()
    if not content:
        return {"rules": []}

    rule_type = _classify_type(content)

    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.;])\s+", content)
        if len(s.strip()) > 15
    ]
    if not sentences:
        return {"rules": []}

    # Group sentences: each sentence that starts a new condition begins a new rule group
    groups: list[str] = []
    current: list[str] = []
    for sent in sentences:
        if _CONDITION_RE.match(sent) and current:
            groups.append(" ".join(current))
            current = [sent]
        else:
            current.append(sent)
    if current:
        groups.append(" ".join(current))

    rules: list[dict[str, str]] = []
    seen: set[str] = set()
    for group in groups:
        action, result = _split_action_result(group)
        key = action[:80].lower()
        if key in seen or not action:
            continue
        seen.add(key)
        rules.append(
            {
                "type": rule_type,
                "action": action[:500],
                "result": result[:500],
                "art_ref": article_number,
                "reg_name": reg_name,
            }
        )

    return {"rules": rules}


def build_fallback_rules(article_number: str, content: str) -> list[dict[str, str]]:
    """Return a single catch-all rule using full article content."""
    return [
        {
            "type": _classify_type(content),
            "action": content[:400],
            "result": content[:400],
            "art_ref": article_number,
            "reg_name": "",
        }
    ]


# SQLite tables used:
# - regulations(reg_id, name, category)
# - articles(reg_id, article_number, content)


def build_graph() -> None:
    """Build KG from SQLite into Neo4j using the fixed assignment schema."""
    sql_conn = sqlite3.connect("ncu_regulations.db")
    cursor = sql_conn.cursor()
    driver = GraphDatabase.driver(URI, auth=AUTH)

    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

        # 1) Create Regulation nodes.
        cursor.execute("SELECT reg_id, name, category FROM regulations")
        regulations = cursor.fetchall()
        reg_map: dict[int, tuple[str, str]] = {}

        for reg_id, name, category in regulations:
            reg_map[reg_id] = (name, category)
            session.run(
                "MERGE (r:Regulation {id:$rid}) SET r.name=$name, r.category=$cat",
                rid=reg_id,
                name=name,
                cat=category,
            )

        # 2) Create Article nodes and HAS_ARTICLE relationships.
        cursor.execute("SELECT reg_id, article_number, content FROM articles")
        articles = cursor.fetchall()

        for reg_id, article_number, content in articles:
            reg_name, reg_category = reg_map.get(reg_id, ("Unknown", "Unknown"))
            session.run(
                """
                MATCH (r:Regulation {id: $rid})
                CREATE (a:Article {
                    number:   $num,
                    content:  $content,
                    reg_name: $reg_name,
                    category: $reg_category
                })
                MERGE (r)-[:HAS_ARTICLE]->(a)
                """,
                rid=reg_id,
                num=article_number,
                content=content,
                reg_name=reg_name,
                reg_category=reg_category,
            )

        # 3) Create fulltext index on Article content.
        session.run(
            """
            CREATE FULLTEXT INDEX article_content_idx IF NOT EXISTS
            FOR (a:Article) ON EACH [a.content]
            """
        )

        # 3b) Extract Rule nodes and link to Articles via CONTAINS_RULE.
        rule_counter = 0
        for reg_id, article_number, content in articles:
            reg_name, _ = reg_map.get(reg_id, ("Unknown", "Unknown"))

            extracted = extract_entities(article_number, reg_name, content)
            rules = extracted.get("rules", [])

            # Guarantee every article has at least one Rule node.
            if not rules:
                rules = [
                    {
                        "type": _classify_type(content),
                        "action": content[:400],
                        "result": content[:400],
                        "art_ref": article_number,
                        "reg_name": reg_name,
                    }
                ]

            for rule in rules:
                action = rule.get("action", "").strip()
                result = rule.get("result", "").strip()
                if not action:
                    continue

                rule_counter += 1
                rule_id = f"{reg_id}_{article_number.replace(' ', '_')}_{rule_counter}"

                session.run(
                    """
                    MATCH (a:Article {number: $num, reg_name: $reg_name})
                    CREATE (ru:Rule {
                        rule_id:  $rule_id,
                        type:     $type,
                        action:   $action,
                        result:   $result,
                        art_ref:  $art_ref,
                        reg_name: $reg_name
                    })
                    MERGE (a)-[:CONTAINS_RULE]->(ru)
                    """,
                    num=article_number,
                    reg_name=reg_name,
                    rule_id=rule_id,
                    type=rule.get("type", "general"),
                    action=action,
                    result=result,
                    art_ref=rule.get("art_ref", article_number),
                )

        print(f"[Rules] Created {rule_counter} Rule nodes.")

        # 4) Create fulltext index on Rule action and result fields.
        session.run(
            """
            CREATE FULLTEXT INDEX rule_idx IF NOT EXISTS
            FOR (r:Rule) ON EACH [r.action, r.result]
            """
        )

        # 5) Coverage audit.
        coverage = session.run(
            """
            MATCH (a:Article)
            OPTIONAL MATCH (a)-[:CONTAINS_RULE]->(r:Rule)
            WITH a, count(r) AS rule_count
            RETURN count(a) AS total_articles,
                   sum(CASE WHEN rule_count > 0 THEN 1 ELSE 0 END) AS covered_articles,
                   sum(CASE WHEN rule_count = 0 THEN 1 ELSE 0 END) AS uncovered_articles
            """
        ).single()

        total_articles = int((coverage or {}).get("total_articles", 0) or 0)
        covered_articles = int((coverage or {}).get("covered_articles", 0) or 0)
        uncovered_articles = int((coverage or {}).get("uncovered_articles", 0) or 0)

        print(
            f"[Coverage] covered={covered_articles}/{total_articles}, "
            f"uncovered={uncovered_articles}"
        )

    driver.close()
    sql_conn.close()


if __name__ == "__main__":
    build_graph()
