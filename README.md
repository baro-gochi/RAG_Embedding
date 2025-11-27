# RAG Embedding

PDF 문서를 분석하여 임베딩하고, RAG(Retrieval-Augmented Generation) 기반 질의응답을 수행하는 시스템입니다.

## 프로젝트 구조

```
Rag_embedding/
├── docs_data/
│   └── KT 공식 문서/
│
├── src/
│   ├── PDF_embedder.py      # PDF 임베딩 메인 모듈
│   └── test_performance.py  # RAG 성능 테스트 모듈
├── .env                     # 환경 변수 설정 파일
├── .gitignore
└── README.md
```

## 주요 기능

### 1. PDF 임베딩 (`src/PDF_embedder.py`)

PDF 문서를 분석하고 ChromaDB에 임베딩하는 핵심 모듈입니다.

#### 주요 기능:
- **PDF 텍스트 추출**: `unstructured` 라이브러리를 사용하여 PDF에서 텍스트 추출
- **하이브리드 청킹**: 문서 구조를 자동 감지하여 최적의 청킹 방식 적용
  - 조/장 기반 청킹 (법률 문서)
  - 제목(Title) 요소 기반 청킹
  - Q&A 패턴 기반 청킹
  - 번호 기반 청킹
  - 시맨틱 청킹 (폴백)
- **키워드 추출**: GPT를 활용한 핵심 키워드 자동 추출
- **콘텐츠 분류**: GPT를 활용한 문서 자동 분류
- **토큰 관리**: 토큰 제한 초과 시 자동 분할 처리
- **ChromaDB 저장**: OpenAI 임베딩을 사용하여 벡터 DB에 저장

#### 설정 값:
- 청킹: MAX_CHUNK_SIZE=1500, MIN_CHUNK_SIZE=30, CHUNK_OVERLAP=100
- 임베딩 모델: `text-embedding-3-large`
- 최대 임베딩 토큰: 8000

### 2. RAG 테스트 (`src/test_performance.py`)

임베딩된 문서를 기반으로 질의응답을 수행하는 테스트 모듈입니다.

#### 주요 기능:
- **문서 검색**: ChromaDB에서 쿼리와 관련된 문서 검색
- **답변 생성**: GPT-4o-mini를 사용하여 검색된 문서 기반 답변 생성
- **대화형 인터페이스**: 터미널에서 질의응답 테스트 가능

## 환경 설정

`.env` 파일에 다음 환경 변수를 설정해야 합니다:

```env
DIR=<PDF 파일 디렉터리 경로>
API_KEY=<OpenAI API 키>
CATEGORIES=<분류 카테고리 목록>
CHROMA_DB_PATH=<ChromaDB 저장 경로>
COLLECTION_NAME=<컬렉션 이름>
```

## 사용 방법

### PDF 임베딩 실행
```bash
python src/PDF_embedder.py
```

### RAG 테스트 실행
```bash
python src/test_performance.py
```

## 의존성

- `unstructured`: PDF 파싱
- `openai`: GPT API 및 임베딩
- `chromadb`: 벡터 데이터베이스
- `tiktoken`: 토큰 계산
- `python-dotenv`: 환경 변수 관리
