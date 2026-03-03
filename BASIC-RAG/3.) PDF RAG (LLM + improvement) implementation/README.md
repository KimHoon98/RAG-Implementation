# Generating Long-Form Reports with RAG — Plan-and-Execute Pattern

A solution to the output truncation problem in LLM-based RAG pipelines, demonstrated through generating a multi-page sell-side equity research report from Samsung Electronics' 4Q 2025 earnings materials.

---

## The Problem

When asking an LLM to produce a long document in a single call — such as an 8-page financial report — the output is almost always truncated to roughly one page, regardless of how explicitly the length is requested.

**Example prompt that fails:**
```
삼성전자 2025년 Q4 분기실적발표에 대해 상세하게 모든 디테일을 놓치지 않고
financial sellside analyst report 형식으로 8장으로 요약해줘
```

**Actual output:** ~1 page

This is caused by two independent issues that compound each other:

| Root Cause | Explanation |
|---|---|
| **`max_new_tokens` hard limit** | The default value of `1024` corresponds to roughly 700–800 English words — far less than 8 pages. Even if you ask for more, the model physically cannot exceed this ceiling. |
| **Completion bias** | LLMs are trained to produce responses that feel naturally "finished." They tend to wrap up with a conclusion after sensing a logical endpoint, regardless of the requested length. Raising `max_new_tokens` alone does not fix this. |

---

## The Solution: Plan-and-Execute

Instead of asking the LLM to write the full report in one shot, we break the task into three stages that each fit comfortably within the token budget:

```
User Question
      ↓
[Step 1] PLANNING
  - LLM reads the question and generates a report outline (table of contents)
  - No RAG retrieval needed here — pure LLM reasoning
  - Output: numbered list of section titles
      ↓
[Step 2] SECTION-BY-SECTION GENERATION  (loop)
  - For each section title, run an independent RAG query
  - The section title is used as the retrieval search query
  - LLM writes only that one section (~1 page) per call
  - Output: detailed prose for each section
      ↓
[Step 3] COMBINE
  - Concatenate all sections in order in Python (no LLM tokens used)
  - Output: final complete report
```

This approach sidesteps both root causes: each individual LLM call is short enough to fit within `max_new_tokens`, and the model never needs to sustain focus across a long document.

---

## Implementation

### Stack

- **LLM:** `google/gemma-2-9b-it` via HuggingFace Inference API
- **Embeddings:** `BAAI/bge-m3` (HuggingFace, GPU-accelerated)
- **Vector Store:** FAISS
- **Framework:** LangChain
- **Observability:** LangSmith

### Source Documents

Two PDFs loaded from Samsung Electronics' 4Q 2025 IR release:
- `삼성_2025Q4_conference_eng_presentation.pdf` — slide deck (15 pages)
- `삼성_2025Q4_script_eng_AudioScript.pdf` — earnings call transcript (34 pages)

Total: 49 pages → 92 chunks after splitting

---

### Step 1: Environment & Document Loading

```python
from langchain_community.document_loaders import PyPDFLoader

docs = []
for path in file_paths:
    loader = PyPDFLoader(path)
    docs.extend(loader.load())
# Result: 49 pages loaded
```

### Step 2: Chunking & Vector Store

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
splits = text_splitter.split_documents(docs)
# Result: 92 chunks

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cuda'}
)
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

# k=8 (increased from default 4) for broader context per section query
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
```

> **Why `k=8`?** The default `k=4` retrieves too narrow a context window for section-level writing. Raising it to 8 ensures each section query captures enough relevant chunks from both the transcript and the slide deck.

### Step 3: LLM Setup

```python
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

llm_endpoint = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    max_new_tokens=2048,   # ← raised from 1024; this is the primary fix
    temperature=0.1,
    huggingfacehub_api_token=hf_token,
)
chat_llm = ChatHuggingFace(llm=llm_endpoint)
```

> **Why `max_new_tokens=2048`?** This allows each section to generate ~1,400 English words — enough for a substantive one-page section. The full report is assembled from multiple such calls, not a single one.

### Step 4: Planning Chain

The planning chain is a **pure LLM call with no retrieval**. The model only needs to reason about what a good report structure looks like, not retrieve facts.

```python
PLANNING_PROMPT = ChatPromptTemplate.from_template("""
You are a senior sell-side equity analyst. A client has made the following request:

REQUEST: {user_question}

Your task is ONLY to create a detailed report outline (table of contents).
Output a numbered list of section titles. No introduction, no explanation.

Section titles:
""")

planning_chain = PLANNING_PROMPT | chat_llm | StrOutputParser()
```

The output is then parsed into a clean list of section titles using `parse_sections()`, which handles varied numbering formats (`1.`, `1)`, `-`):

```python
def parse_sections(outline_text: str) -> list[str]:
    lines = outline_text.strip().split("\n")
    sections = []
    for line in lines:
        cleaned = re.sub(r'^[\d]+[.)\-]\s*', '', line.strip())
        if cleaned:
            sections.append(cleaned)
    return sections
```

**Example output from the planning step** (actual run):
```
1. Executive Summary & Investment Thesis
2. Consolidated Financial Results (4Q25 P&L)
3. Segment Performance Analysis (Semiconductors, Display, Mobile, Consumer Electronics)
4. Key Drivers of Performance (Demand Trends, Pricing, Costs)
5. Balance Sheet & Cash Flow Analysis
6. Capital Allocation & Shareholder Returns
7. Outlook & Risks
8. Valuation & Recommendation
```

### Step 5: Section Generation Chain

Each section is generated with its own dedicated RAG retrieval call. The section title is combined with the original question to form a targeted search query.

```python
SECTION_PROMPT = ChatPromptTemplate.from_template("""
You are a senior sell-side equity analyst writing a research report on Samsung Electronics 4Q 2025.
The overall report request is: {user_question}
You are now writing ONLY this specific section: {section_title}

Use the following source material:
---
{context}
---

INSTRUCTIONS:
- Write in full sentences and professional analyst prose. No bullet points.
- Include ALL specific numbers, financial figures, and percentages.
- Use financial terms: QoQ, YoY, bps, OP margin, ASP, etc.
- This section should fill roughly one page.
- Do NOT include a section title header. Do NOT write boilerplate intro/conclusion.

Section content:
""")

def generate_section(section_title: str, user_question: str) -> str:
    search_query = f"{section_title} Samsung Electronics 4Q 2025 earnings"
    context_docs = retriever.invoke(search_query)
    context_text = "\n\n".join([doc.page_content for doc in context_docs])

    section_chain = SECTION_PROMPT | chat_llm | StrOutputParser()
    return section_chain.invoke({
        "user_question": user_question,
        "section_title": section_title,
        "context": context_text
    })
```

### Step 6: Combine into Final Report

Assembly happens entirely in Python — no LLM tokens are consumed here.

```python
def generate_full_report(user_question: str) -> str:
    # Step 1: Plan
    outline_text = planning_chain.invoke({"user_question": user_question})
    sections = parse_sections(outline_text)

    # Step 2: Generate each section
    generated_sections = []
    for section_title in sections:
        content = generate_section(section_title, user_question)
        generated_sections.append((section_title, content))

    # Step 3: Combine
    report_parts = ["=" * 72, "SAMSUNG ELECTRONICS (005930 KS)", ...]
    for i, (title, content) in enumerate(generated_sections):
        report_parts.append(f"{i+1}. {title}")
        report_parts.append(content.strip())

    return "\n".join(report_parts)
```

**Actual output from a real run:**
```
[Step 1/3] 목차 생성 중...
파싱된 섹션 수: 8개

[Step 2/3] 섹션별 내용 생성 중...
  [1/8] Executive Summary & Investment Thesis      → 1,765자 생성
  [2/8] Consolidated Financial Results (4Q25 P&L)  → 1,074자 생성
  [3/8] Segment Performance Analysis               → 1,233자 생성
  [4/8] Key Drivers of Performance                 → 2,128자 생성
  [5/8] Balance Sheet & Cash Flow Analysis         → 1,553자 생성
  [6/8] Capital Allocation & Shareholder Returns   → 1,670자 생성
  [7/8] Outlook & Risks                            → 1,645자 생성
  [8/8] Valuation & Recommendation                 → 1,574자 생성

[Step 3/3] 최종 리포트 합성 중...
완료! 총 13,769자 / 87줄
```

---

## LangSmith Trace

LangSmith was used to track each chain invocation and inspect retrieved documents, prompt inputs, and model outputs at each step.

With the plan-and-execute pattern, a single `generate_full_report()` call produces **9 traces** in LangSmith (1 planning call + 8 section generation calls), making it straightforward to debug any individual section that underperforms without re-running the entire pipeline.

Each trace captures:
- The search query used for retrieval
- Which chunks were retrieved (and their source document + page)
- The full prompt sent to the model
- The raw model output before parsing

---

## Token Budget at a Glance

| Stage | LLM Calls | Tokens per Call | Notes |
|---|---|---|---|
| Step 1 — Planning | 1 | ~300–400 | No retrieval |
| Step 2 — Section generation | N (one per section) | ~800–1,500 | RAG per section |
| Step 3 — Combine | 0 | 0 | Python string join |
| **Total (8 sections)** | **9** | — | |

---

## Known Limitations

**HuggingFace free tier rate limits** — With 9 sequential API calls, you may hit the per-minute request limit on the free inference API. If you see a rate limit error, add a short delay between section calls:

```python
import time
content = generate_section(section_title, user_question)
time.sleep(5)  # add inside the loop in generate_full_report()
```

## Langsmith traces:
LangSmith Traces in order
https://smith.langchain.com/public/20b4a36a-e8e7-4db3-9435-6db59bf004d6/r
https://smith.langchain.com/public/6b3ae726-2ddd-4cfc-abf5-c1dc38825af4/r
https://smith.langchain.com/public/46eafc53-74c3-4595-8e94-fe89ac599418/r
https://smith.langchain.com/public/141a88da-0bec-4a27-ac09-62817fde6755/r
https://smith.langchain.com/public/d626d16d-289b-4555-83e9-296a35c05780/r
https://smith.langchain.com/public/1d0212bd-ec02-4472-9e49-5ae225ab3aef/r
https://smith.langchain.com/public/a1ad9cc8-a626-40eb-951e-92b492655878/r
https://smith.langchain.com/public/c394bd45-1028-460b-84e1-3e43e79e0f6e/r
https://smith.langchain.com/public/efb94108-a4be-460e-8b53-7693da38150b/r
https://smith.langchain.com/public/7828756c-08a5-4da8-ac75-e2e92b399f8f/r
https://smith.langchain.com/public/469fa813-7b69-4924-a0ab-ab9672a22b73/r
https://smith.langchain.com/public/120fc734-761a-47d7-89c0-e131b1986db0/r
https://smith.langchain.com/public/1931208a-54ca-431c-a697-bc9dba06b50e/r
https://smith.langchain.com/public/72d82356-6f5c-4dd3-8687-30e963f10830/r
https://smith.langchain.com/public/1352b700-4589-4fef-81a0-915a50b0651b/r
https://smith.langchain.com/public/408feabf-2839-47e2-a71e-dc497e6efd0f/r
https://smith.langchain.com/public/84eabd1b-b862-49dd-964c-9945d92e8b80/r
**Section quality depends on chunk coverage** — If a section topic is spread thinly across the source documents, the retriever may not surface enough relevant content. Increasing `k` further or expanding `chunk_size` can help in those cases.
