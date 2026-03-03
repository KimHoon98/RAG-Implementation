# 📊 삼성전자 2025 Q4 실적발표 RAG 챗봇

삼성전자 2025년 4분기 실적발표 PDF 문서를 기반으로, **완전 무료 오픈소스 모델만** 사용해 구축한 RAG(Retrieval-Augmented Generation) 챗봇입니다.

---

## 🏗️ 전체 아키텍처

```
PDF 문서 (2개)
    ↓ PyPDFLoader
문서 로드 (49 pages)
    ↓ RecursiveCharacterTextSplitter
텍스트 청크 분할 (92 chunks)
    ↓ HuggingFaceEmbeddings (BAAI/bge-m3)
벡터 임베딩 생성 (CUDA GPU 활용)
    ↓ FAISS VectorStore
벡터 저장 및 인덱싱
    ↓ Retriever
질문과 유사한 청크 검색
    ↓ ChatPromptTemplate
프롬프트 조합 (context + question)
    ↓ HuggingFaceEndpoint (gemma-2-9b-it)
LLM 추론
    ↓ StrOutputParser
최종 답변 출력
```

---

## 🛠️ 기술 스택

| 구성 요소 | 사용 기술 | 비용 |
|-----------|-----------|------|
| **문서 로더** | `PyPDFLoader` (langchain-community) | 무료 |
| **텍스트 분할** | `RecursiveCharacterTextSplitter` | 무료 |
| **임베딩 모델** | `BAAI/bge-m3` (HuggingFace) | 무료 |
| **벡터 DB** | `FAISS` (Facebook AI) | 무료 |
| **LLM** | `google/gemma-2-9b-it` (HuggingFace Inference API) | 무료 |
| **프레임워크** | `LangChain` | 무료 |
| **GPU 가속** | NVIDIA CUDA (RTX 5060 Ti) | - |

---

## 📂 입력 문서

- `삼성_2025Q4_conference_eng_presentation.pdf` (15 pages)
- `삼성_2025Q4_script_eng_AudioScript.pdf` (34 pages)
- **총 로드 페이지 수: 49 pages**

---

## ⚙️ 코드 Flow (단계별 설명)

### Step 1 — 환경 설정 및 라이브러리 import

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
```

`.env` 파일에 HuggingFace API 토큰을 저장하고 `load_dotenv()`로 로드합니다.

---

### Step 2 — PDF 문서 로드

```python
loader = PyPDFLoader(path)
docs.extend(loader.load())
# 총 49페이지 로드
```

프레젠테이션 자료와 어닝스콜 오디오 스크립트, 두 개의 PDF를 함께 로드해 더 풍부한 컨텍스트를 확보합니다.

---

### Step 3 — 텍스트 분할 (Chunking)

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,   # 청크 당 최대 1000자
    chunk_overlap=50   # 문맥 유지를 위해 50자 오버랩
)
splits = text_splitter.split_documents(docs)
# 결과: 92개 청크
```

`chunk_overlap=50`으로 청크 경계에서 문맥이 잘리는 것을 방지합니다.

---

### Step 4 — 벡터 임베딩 및 FAISS 인덱싱

```python
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cuda'}  # GPU 가속
)

vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()
```

- **임베딩 모델**: `BAAI/bge-m3` — 로컬 다운로드 후 GPU에서 실행, API 비용 없음
- **벡터 DB**: FAISS — 인메모리 저장, 빠른 유사도 검색

---

### Step 5 — LLM 연결

```python
llm_endpoint = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    max_new_tokens=1024,
    temperature=0.1,
    huggingfacehub_api_token=hf_token,
)
chat_llm = ChatHuggingFace(llm=llm_endpoint)
```

HuggingFace Inference API의 **무료 티어**를 활용합니다. `gemma-2-9b-it`은 사전에 [HuggingFace 모델 페이지](https://huggingface.co/google/gemma-2-9b-it)에서 라이선스 동의가 필요합니다.

프롬프트는 삼성전자 실적발표 전문가 페르소나로 설정해 수치 기반의 정확한 답변을 유도합니다:

```
당신은 삼성전자 실적발표 전문 AI 어시스턴트입니다.
제공된 자료를 바탕으로 수치를 정확히 포함하여 상세히 답변하세요.
```

---

### Step 6 — RAG 체인 구성

```python
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | chat_llm
    | StrOutputParser()
)
```

LangChain의 LCEL(LangChain Expression Language) 파이프라인으로 Retriever → Prompt → LLM → Parser를 한 줄로 연결합니다.

---

### Step 7 — 질의응답 실행

```python
response = rag_chain.invoke("2025 4Q highlights에 대해 최대한 상세하게 알려줘")
```

---

## 💬 테스트 질문 및 답변 예시

**Q: 2025 4Q highlights에 대해 최대한 상세하게 알려줘**
> 매출 93.8조 KRW, 영업이익 20.1조 KRW, 연간 매출 333.6조 KRW, 영업이익 43.6조 KRW 등 주요 재무 지표를 상세히 답변

**Q: HBM4 개발 현황이랑 2026년 HBM 매출 전망 알려줘**
> HBM4 최종 자격 검증 단계 진입, 11.7 Gbps 최고 성능 Bin 대량 생산 중, 2026년 HBM 매출 전년 대비 3배 이상 증가 전망

**Q: 2025년 4분기에 DS사업부는 좋아졌는데 DX사업부는 왜 나빠졌어?**
> 신규 스마트폰 출시 효과 소멸 및 미국 관세로 인한 가전제품 부진으로 DX 매출 8% QoQ 감소 설명

---

## 📦 설치 방법

```bash
pip install langchain langchain-community langchain-huggingface
pip install faiss-gpu  # GPU 버전 (CPU: faiss-cpu)
pip install pypdf python-dotenv
```

`.env` 파일 생성:
```
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

> ⚠️ HuggingFace 토큰 발급 계정과 모델 라이선스 동의 계정이 **반드시 동일**해야 합니다.

---

## 💡 무료로 만든 핵심 포인트

1. **임베딩**: `BAAI/bge-m3`를 로컬 GPU에서 실행 → OpenAI Embedding API 비용 0원
2. **LLM**: HuggingFace Inference API 무료 티어 활용 → OpenAI GPT API 비용 0원
3. **벡터 DB**: FAISS 인메모리 → Pinecone/Weaviate 같은 유료 벡터 DB 비용 0원
