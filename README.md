# Assignment 4: KG-based QA for NCU Regulations

A question-answering system for National Central University (NCU) academic regulations, built on a Neo4j Knowledge Graph and a local HuggingFace language model. Given a natural-language question about university rules, the system retrieves the most relevant regulation articles from the graph and generates a grounded answer using the LLM.

---

## System Architecture

```
PDF Regulations
      |
  setup_data.py   -- pdfplumber + regex parsing
      |
  SQLite (ncu_regulations.db)
      |
  build_kg.py     -- Cypher node/relationship creation
      |
  Neo4j Knowledge Graph
      |
  query_system.py -- entity extraction + 3-tier retrieval + LLM generation
      |
  auto_test.py    -- LLM-as-Judge benchmark (20 questions)
```

### Knowledge Graph Schema

```
(Regulation)-[:HAS_ARTICLE]->(Article)-[:CONTAINS_RULE]->(Rule)
```

| Node | Key Properties |
|------|---------------|
| Regulation | reg_id, name, source_file |
| Article | article_id, article_number, content |
| Rule | rule_id, action, result, type (penalty/requirement/procedure/general) |

Two fulltext indexes are created during KG construction:

- `article_content_idx` — on `Article.content`, used as a broad fallback
- `rule_idx` — on `Rule.action` and `Rule.result`, used for typed and broad retrieval

### Retrieval Strategy (Three-Tier Fallback)

1. **Typed query** — filters by `Rule.type` inferred from the question, searches `rule_idx`
2. **Broad query** — same keywords, no type filter, wider recall on `rule_idx`
3. **Article fallback** — searches `article_content_idx` when the first two tiers return nothing

Results are deduplicated by `rule_id` and the top-3 rules are passed to the LLM.

---

## Prerequisites

- **Python 3.11** (strict requirement)
- **Docker Desktop** (to run Neo4j locally)
- Internet access on first run (to download the HuggingFace model; subsequent runs are fully offline)

---

## Environment Setup

### 1. Start Neo4j via Docker

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

| Port | Purpose |
|------|---------|
| 7474 | Neo4j Browser web UI |
| 7687 | Bolt protocol (Python driver connection) |

Verify the database is ready: open `http://localhost:7474` and log in with `neo4j` / `password`.

### 2. Create and Activate Virtual Environment

**macOS / Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## File Descriptions

| File | Description |
|------|-------------|
| `setup_data.py` | Parses raw PDF regulation files in `source/` using pdfplumber and regex; cleans text and stores structured data into `ncu_regulations.db` (SQLite) |
| `build_kg.py` | Reads from SQLite; creates Regulation, Article, and Rule nodes in Neo4j with Cypher; builds two fulltext indexes; classifies each Rule as penalty/requirement/procedure/general |
| `llm_loader.py` | Singleton loader for `Qwen/Qwen2.5-1.5B-Instruct`; downloads once to `./hf_model_cache/` and loads from local cache on subsequent runs; CPU/GPU auto-detection |
| `query_system.py` | Core Q&A logic: entity extraction, three-tier Cypher retrieval, context assembly, and LLM-based answer generation |
| `auto_test.py` | Runs 20 benchmark questions from `test_data.json`; uses the same LLM as an impartial judge (LLM-as-Judge pattern) to score each answer PASS/FAIL |
| `test_data.json` | 20 hand-crafted question/expected-answer pairs covering a range of NCU regulation topics |

---

## Execution Order

Ensure the Neo4j Docker container is running before starting.

```bash
# Step 1: Parse PDFs and populate SQLite
python setup_data.py

# Step 2: Build the Knowledge Graph in Neo4j
python build_kg.py

# Step 3 (optional): Test the chatbot interactively
python query_system.py

# Step 4: Run the automated benchmark
python auto_test.py
```

---

## LLM Configuration

The model is configured in `llm_loader.py`:

```python
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"   # ~3 GB, CPU-compatible
MODEL_CACHE_DIR = "./hf_model_cache"        # local cache directory
```

Alternative models (change `MODEL_ID` to switch):

| Model | Size | Context | Notes |
|-------|------|---------|-------|
| `Qwen/Qwen2.5-1.5B-Instruct` | ~3 GB | 32K | Default; fast on CPU |
| `Qwen/Qwen2.5-3B-Instruct` | ~6 GB | 32K | Better quality, slower |
| `microsoft/Phi-3.5-mini-instruct` | ~7 GB | 128K | Best for long documents |

---

## Evaluation Results

The automated benchmark covers 20 questions across topics including grading rules, leave of absence, retake policies, course withdrawal, and graduation requirements. Neo4j connected with **200 Rule nodes** across 6 regulation documents.

| Metric | Result |
|--------|--------|
| Total questions | 20 |
| Passed | 18 |
| Failed | 2 |
| **Accuracy** | **90.0%** |

### Per-Question Results

| Q | Question | Result |
|---|----------|--------|
| 1 | How many minutes late can a student be before they are barred from the exam? | PASS |
| 2 | Can I leave the exam room 30 minutes after it starts? | PASS |
| 3 | What is the penalty for forgetting my student ID? | PASS |
| 4 | What is the penalty for using electronic devices with communication capabilities during an exam? | PASS |
| 5 | What is the penalty for cheating, such as copying or passing notes, during an exam? | PASS |
| 6 | Is a student allowed to take the question paper out of the exam room? | PASS |
| 7 | What happens if a student threatens the invigilator? | PASS |
| 8 | What is the fee for replacing a lost EasyCard student ID? | PASS |
| 9 | What is the fee for replacing a lost Mifare (non-EasyCard) student ID? | PASS |
| 10 | How many working days does it take to get a new student ID after application? | PASS |
| 11 | What is the minimum total credits required for undergraduate graduation? | PASS |
| 12 | How many semesters of Physical Education (PE) are required for undergraduate students? | **FAIL** |
| 13 | Are Military Training credits counted towards graduation credits? | PASS |
| 14 | What is the standard duration of study for a bachelor's degree? | PASS |
| 15 | What is the maximum extension period for undergraduate study duration? | **FAIL** |
| 16 | What is the passing score for undergraduate students? | PASS |
| 17 | What is the passing score for graduate (Master/PhD) students? | PASS |
| 18 | Under what condition will an undergraduate student be dismissed (expelled) due to poor grades? | PASS |
| 19 | Can a student take a make-up exam for a failed semester grade? | PASS |
| 20 | What is the maximum duration for a leave of absence (suspension of schooling)? | PASS |

### Failure Analysis

**Q12 — Physical Education semester requirement**

- Question: How many semesters of Physical Education (PE) are required for undergraduate students?
- Bot answer: "a required PE course must be completed within one semester" (wrong)
- Root cause: The KG retrieval ranks Article 52 (PE retake rules) above the article encoding the semester count. The specific "4 semesters" fact is distributed across article prose and is not extracted as a standalone Rule node, so it does not surface in the top-3 results passed to the LLM.

**Q15 — Undergraduate study period extension**

- Question: What is the maximum extension period for undergraduate study duration?
- Bot answer: "the maximum extension period is one year" (wrong — correct answer is 2 years)
- Root cause: Article 57 (graduate extension = 1 year) scores higher than Article 13-1 (undergraduate extension = 2 years) because the query keywords match both. Without an explicit student-type filter in the Cypher query, the LLM receives both articles and picks the graduate rule.

---

## Screenshots

### Neo4j Browser — Complete Knowledge Graph (All Regulations, LIMIT 50)

![Complete KG](screenshots/完整三層圖.png)

### Neo4j Browser — NCU General Regulations Subgraph

![General Regulation Graph](screenshots/General%20Regulation.png)

### Neo4j Browser — Grading System Subgraph

![Grading System Graph](screenshots/Grading%20System.png)

### Auto Test Summary

![Auto Test Summary](screenshots/summary.png)
