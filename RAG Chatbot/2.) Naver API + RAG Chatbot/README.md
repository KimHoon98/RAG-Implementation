# rag_engine.py 코드 설명

---

## 전체 구조

`rag_engine.py`는 두 개의 클래스로 구성됩니다.

```
NaverSearchClient       # 네이버 API 호출만 담당
RAGManager              # 검색 결과를 가공하고 LLM 체인을 조립
    ├── _history_search_query()   # 대화 히스토리 기반 검색어 재작성
    ├── _fetch_context()          # 네이버 API 호출 + 컨텍스트 생성
    └── get_chain()               # LangChain RAG 체인 반환
```

---

## NaverSearchClient

### 인증 방식

```python
self.headers = {
    "X-Naver-Client-Id": self.client_id,
    "X-Naver-Client-Secret": self.client_secret,
}
```

네이버 API는 요청마다 HTTP 헤더에 Client ID와 Secret을 실어 보내 인증합니다.
`requests.get()` 호출 시 이 `headers`를 매번 함께 전달합니다.
키 값은 `.env` 파일에서 `os.getenv()`로 불러오며, 둘 중 하나라도 없으면 `ValueError`를 즉시 발생시켜 런타임 에러를 방지합니다.

---

### `search_local()` — 식당 기본 정보

```python
def search_local(self, query: str, display: int = 5) -> list[dict]:
    params = {"query": query, "display": display, "sort": "comment"}
    response = requests.get(
        f"{self.BASE_URL}/local",
        headers=self.headers,
        params=params,
        timeout=5,
    )
    response.raise_for_status()
    return response.json().get("items", [])
```

**파라미터 설명**

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| `sort` | `"comment"` | 리뷰(댓글) 수가 많은 순으로 정렬. 검증된 맛집을 상단에 노출 |
| `display` | 기본값 `5` | 가져올 결과 개수. `config.yaml`의 `local_display` 값으로 주입됨 |
| `timeout` | `5` | 5초 내 응답 없으면 요청 포기 |

**반환 데이터 주요 필드**

| 필드 | 설명 |
|------|------|
| `title` | 식당 이름 (HTML 태그 포함) |
| `roadAddress` | 도로명 주소 |
| `address` | 지번 주소 |
| `category` | 업종 분류 (예: `음식점 > 한식 > 찌개,전골`) |
| `description` | 식당 소개 |

`response.raise_for_status()`는 HTTP 에러 코드(401, 404 등)를 받으면 즉시 예외를 발생시켜, 빈 응답을 정상으로 오인하는 상황을 방지합니다.

---

### `search_blog()` — 블로그 리뷰

```python
def search_blog(self, query: str, display: int = 5) -> list[dict]:
    params = {"query": f"{query} 리뷰 맛집", "display": display, "sort": "sim"}
```

`search_local()`과 다른 두 가지 차이점이 있습니다.

**1. 검색어 자동 보강**
```python
"query": f"{query} 리뷰 맛집"
```
사용자가 `"서울역 근처"`를 입력해도 `"서울역 근처 리뷰 맛집"`으로 변환해 더 정확한 리뷰 포스팅을 탐색합니다.

**2. 정렬 기준 변경**
- `search_local`: `sort="comment"` (리뷰 많은 순)
- `search_blog`: `sort="sim"` (연관도/정확도 높은 순)

블로그는 양보다 질이 중요하기 때문에, 리뷰 수보다 검색어와 얼마나 관련이 높은지를 기준으로 정렬합니다.

---

## RAGManager

### `__init__()` — 설정값 로드

```python
def __init__(self, config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        self.settings = config["rag_settings"]

    self.model_name = self.settings["model_name"]
    naver_cfg = self.settings.get("naver_search", {})
    self.local_display = naver_cfg.get("local_display", 5)
    self.blog_display = naver_cfg.get("blog_display", 5)
```

`config.yaml`에서 모델명과 검색 결과 수를 런타임에 주입받습니다.
`naver_cfg.get("local_display", 5)`처럼 기본값을 지정해, `config.yaml`에 해당 키가 없어도 앱이 중단되지 않습니다.

---

### `_history_search_query()` — 문맥 기반 검색어 재작성

```python
def _history_search_query(self, current_question: str, chat_history: list) -> str:
    if not chat_history:
        return current_question

    llm = ChatOpenAI(model=self.model_name, temperature=0.1)
    history_text = "\n".join(
        f"{'사용자' if isinstance(m, HumanMessage) else '어시스턴트'}: {m.content}"
        for m in chat_history[-6:]
    )
    ...
    result = llm.invoke(history_prompt)
    return result.content.strip()
```

**왜 필요한가?**

LLM은 대화 문맥을 이해하지만, 네이버 API는 그렇지 않습니다.

```
사용자: "강남역 맛집 알려줘"
AI:     "A식당, B식당이 있습니다."
사용자: "거기 메뉴는 뭐야?"  →  네이버 검색어: "거기 메뉴"  ← 결과 없음
```

이 메서드는 히스토리를 LLM에 넘겨 `"거기"`가 `"A식당"`임을 파악하고, 네이버에 실제로 검색할 수 있는 구체적인 키워드로 재작성합니다.

**핵심 설계 결정**

- `temperature=0.1`: 창의성을 낮춰 검색어를 일관되게 생성
- `chat_history[-6:]`: 최근 6개 메시지만 참조해 토큰 비용과 문맥 혼란을 최소화
- `if not chat_history: return current_question`: 첫 질문은 히스토리가 없으므로 LLM 호출 없이 바로 반환해 불필요한 API 비용 절약

---

### `_fetch_context()` — 컨텍스트 생성

```python
def _fetch_context(self, search_query: str) -> tuple[str, dict]:
    context_parts = []
    raw_data = {"local": [], "blog": []}
    ...
    return context_str, raw_data
```

**두 가지를 동시에 반환하는 이유**

| 반환값 | 타입 | 용도 |
|--------|------|------|
| `context_str` | `str` | LLM 프롬프트에 삽입할 읽기용 텍스트 |
| `raw_data` | `dict` | UI에서 식당 이름/주소/링크를 직접 다루기 위한 원본 데이터 |

LLM에 넘길 텍스트와 UI에서 조작할 구조화 데이터를 분리해 각각의 목적에 맞게 사용합니다.

**HTML 태그 제거**

```python
name = item.get("title", "").replace("<b>", "").replace("</b>", "")
```

네이버 API는 검색어 강조를 위해 `<b>맛집</b>` 형태로 응답을 돌려줍니다.
이를 제거하지 않으면 LLM이 `<b>`, `</b>` 문자열을 텍스트로 읽어 답변에 포함시킵니다.

**주소 Fallback**

```python
address = item.get("roadAddress") or item.get("address", "")
```

도로명 주소(`roadAddress`)를 먼저 시도하고, 없으면 지번 주소(`address`)로 자동 대체합니다.
`or` 연산자를 활용해 빈 문자열(`""`)도 falsy로 처리되도록 합니다.

**에러 핸들링 전략**

```python
try:
    local_items = self.naver.search_local(...)
except Exception as e:
    context_parts.append(f"[지역 검색 오류: {e}]")
```

Local 검색과 Blog 검색을 각각 별도의 `try-except`로 감쌉니다.
하나의 API가 실패해도 나머지 결과로 답변을 이어갈 수 있고, 오류 메시지를 컨텍스트에 포함시켜 LLM이 "현재 정보를 가져올 수 없다"고 안내하게 합니다.

**Hallucination 방지**

```python
context_str = "\n".join(context_parts) if context_parts else "검색 결과가 없습니다."
```

검색 결과가 하나도 없을 때 빈 컨텍스트 대신 명시적으로 `"검색 결과가 없습니다."`를 전달합니다.
LLM이 빈 컨텍스트를 받으면 학습 데이터에서 임의로 식당 정보를 생성할 수 있기 때문입니다.

---

### `get_chain()` — RAG 체인 조립

#### 프롬프트 구성

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", """...\n{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])
```

세 개의 레이어로 구성됩니다.

| 레이어 | 역할 |
|--------|------|
| `system` | 페르소나 부여 + 답변 형식 강제 + 네이버 검색 컨텍스트 삽입 |
| `MessagesPlaceholder` | 이전 대화 기록을 프롬프트 중간에 동적으로 삽입 |
| `human` | 현재 사용자 질문 |

`MessagesPlaceholder`가 없으면 LLM은 매 답변마다 이전 대화를 기억하지 못합니다.
이 한 줄이 멀티턴 대화를 가능하게 합니다.

#### 체인 연결

```python
chain = prompt | llm | StrOutputParser()
```

LangChain의 LCEL(LangChain Expression Language) 파이프라인입니다.

```
prompt          # 변수들을 채워 완성된 프롬프트 생성
    |
    ▼
llm             # ChatOpenAI 모델 호출, AIMessage 반환
    |
    ▼
StrOutputParser # AIMessage에서 텍스트만 추출해 str로 변환
```

#### 내부 실행 함수 `rag_chain_with_history()`

```python
def rag_chain_with_history(question: str, chat_history: list = []) -> dict:
    search_query = self._history_search_query(question, chat_history)  # 1
    context, raw_data = self._fetch_context(search_query)              # 2
    answer = chain.invoke({                                            # 3
        "context": context,
        "chat_history": chat_history,
        "question": question,
    })
    restaurants = [...]                                                # 4
    return {"answer": answer, "restaurants": restaurants, "debug": ...}
```

매 호출마다 실행되는 4단계 파이프라인입니다.

1. **검색어 재작성**: 히스토리를 보고 실제로 검색할 키워드 결정
2. **데이터 수집**: 네이버 API로 컨텍스트 생성
3. **LLM 호출**: 컨텍스트 + 히스토리 + 질문을 조합해 답변 생성
4. **데이터 정제**: `raw_data["local"]`에서 식당 이름과 주소만 추출해 UI용 리스트 구성

**반환 구조**

```python
return {
    "answer": answer,           # 사용자에게 보여줄 최종 답변 텍스트
    "restaurants": restaurants, # UI 렌더링용 [{name, address}, ...] 리스트
    "debug": {
        "search_query": ...,    # 실제 사용된 검색어 (디버깅용)
        "context": ...,         # LLM에 전달된 컨텍스트 전문 (디버깅용)
    }
}
```

`debug` 키를 별도로 포함시켜, "왜 이런 답변이 나왔지?"를 추적할 때 실제 검색어와 컨텍스트를 바로 확인할 수 있습니다.
