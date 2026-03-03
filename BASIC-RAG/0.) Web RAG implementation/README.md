# LangChain & HuggingFace를 활용한 RAG 챗봇

## 1. 코드 구현 방법

### Step 1 — 문서 로드

LangChain의 `WebBaseLoader`를 사용하여 네이버 뉴스 기사를 웹에서 불러옵니다. BeautifulSoup을 활용해 기사 본문과 제목에 해당하는 HTML `<div>` 요소만 파싱하고, 내비게이션이나 광고 등 불필요한 내용은 제거합니다.

```python
loader = WebBaseLoader(
    web_paths=("https://n.news.naver.com/article/437/0000378416",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "div",
            attrs={"class": ["newsct_article _article_body", "media_end_head_title"]},
        )
    ),
)
docs = loader.load()
```

### Step 2 — 청크 분할

문맥이 경계에서 손실되지 않도록 텍스트를 겹치는 청크로 분할합니다. 청크 크기는 1000자, 겹침(overlap)은 50자로 설정합니다.

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
splits = text_splitter.split_documents(docs)
```

### Step 3 — 임베딩 및 인덱싱

HuggingFace의 `all-MiniLM-L6-v2` 모델로 각 청크를 임베딩합니다. 이 모델은 가볍지만 의미 유사도 검색에 효과적인 문장 변환 모델입니다. 생성된 벡터는 FAISS 인메모리 벡터 저장소에 저장되고, 이를 기반으로 검색기(retriever)를 생성합니다.

```python
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()
```

### Step 4 — LLM 설정

`google/gemma-2-9b-it` 모델을 HuggingFace Inference Endpoints를 통해 불러옵니다. API 토큰은 보안을 위해 `.env` 파일에서 읽어오며, LangChain의 채팅 모델 인터페이스에 맞게 `ChatHuggingFace`로 래핑합니다.

```python
llm_endpoint = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    max_new_tokens=512,
    temperature=0.1,
    huggingfacehub_api_token=hf_token
)
chat_llm = ChatHuggingFace(llm=llm_endpoint)
```

### Step 5 — 프롬프트 정의

검색된 문맥에서만 답변하도록 모델을 엄격하게 제한하고, 답변은 반드시 한국어로 작성하도록 프롬프트를 설계합니다. 이를 통해 환각을 최소화합니다.

```python
template = """당신은 질문-답변을 도와주는 AI 어시스턴트입니다.
아래의 제공된 문맥(Context)을 활용해서만 질문에 답하세요.
답을 모른다면 모른다고 말하고, 직접적인 답이 문맥에 없다면 문맥을 바탕으로 추론하지 마세요.
답변은 반드시 한국어로 작성하세요.

#Context:
{context}

#Question:
{question}

#Answer:"""

prompt = ChatPromptTemplate.from_template(template)
```

### Step 6 — RAG 체인 구성 및 실행

전체 파이프라인을 LangChain LCEL 체인으로 구성합니다. 검색기가 자동으로 관련 청크를 가져오고, 이를 질문과 함께 프롬프트에 주입한 뒤 LLM을 거쳐 최종적으로 문자열로 파싱됩니다.

```python
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | chat_llm
    | StrOutputParser()
)

question = "부영그룹은 출산 직원에게 얼마의 지원을 제공하나요?"
response = rag_chain.invoke(question)
# → "부영그룹은 출산 직원에게 1억원을 지원합니다."
```
