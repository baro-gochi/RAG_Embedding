"""
ChromaDB 임베딩 테스트 스크립트
- JSON_embedder.py로 임베딩된 데이터 검증
- 질문 입력 시 RAG 기반 답변 생성
"""

import os
import json
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

# 환경설정
load_dotenv()
OPENAI_API_KEY = os.getenv("API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH_SIMPLE")
COLLECTION_NAME = os.getenv("aicc_documents")
EMBEDDING_MODEL = "text-embedding-3-small"

# OpenAI 클라이언트
client = OpenAI(api_key=OPENAI_API_KEY)


def get_collection():
    """ChromaDB 컬렉션 가져오기"""
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL
    )

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    try:
        collection = chroma_client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=openai_ef
        )
        return collection
    except Exception as e:
        print(f"[ERROR] Collection not found: {e}")
        return None


def search_relevant_docs(query: str, n_results: int = 5):
    """질문과 관련된 문서 검색"""
    collection = get_collection()
    if not collection:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    docs = []
    if results['ids'][0]:
        for doc, meta, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            similarity = 1 - distance
            docs.append({
                "content": doc,
                "title": meta.get('title', ''),
                "category": meta.get('category', ''),
                "classification": meta.get('classification', ''),
                "similarity": similarity
            })

    return docs


def generate_answer(query: str, context_docs: list):
    """검색된 문서를 기반으로 답변 생성"""

    # 컨텍스트 구성
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        context_parts.append(f"""
[참고문서 {i}] (유사도: {doc['similarity']:.2f})
제목: {doc['title']}
분류: {doc['classification']}
내용:
{doc['content'][:1500]}
""")

    context = "\n".join(context_parts)

    # GPT로 답변 생성
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """당신은 KT 고객센터 상담원입니다.
제공된 참고문서를 기반으로 고객의 질문에 정확하고 친절하게 답변하세요.

규칙:
1. 참고문서에 있는 정보만 사용하여 답변하세요.
2. 참고문서에 없는 내용은 "해당 정보는 확인되지 않습니다"라고 답변하세요.
3. 요금, 할인율 등 숫자 정보는 정확하게 전달하세요.
4. 답변은 간결하고 명확하게 작성하세요.
5. 필요시 관련 추가 정보도 안내해주세요."""
            },
            {
                "role": "user",
                "content": f"""[참고문서]
{context}

[고객 질문]
{query}

위 참고문서를 바탕으로 고객 질문에 답변해주세요."""
            }
        ],
        temperature=0.3,
        max_tokens=1000
    )

    return response.choices[0].message.content


def show_collection_info():
    """컬렉션 정보 표시"""
    collection = get_collection()
    if not collection:
        print("[ERROR] Collection not found")
        return

    count = collection.count()
    print(f"\n[Collection Info]")
    print(f"  - Name: {COLLECTION_NAME}")
    print(f"  - Documents: {count}")
    print(f"  - DB Path: {CHROMA_DB_PATH}")


def chat_mode():
    """대화형 Q&A 모드"""
    print("\n" + "="*60)
    print("  KT 고객센터 RAG 테스트")
    print("="*60)

    show_collection_info()

    print("\n[사용법]")
    print("  - 질문을 입력하면 RAG 기반으로 답변합니다.")
    print("  - 'quit' 또는 'exit' 입력 시 종료")
    print("  - 'info' 입력 시 컬렉션 정보 표시")
    print("  - 'docs' 입력 시 마지막 검색 문서 상세 표시")
    print("="*60)

    last_docs = []

    while True:
        print()
        query = input("[질문] > ").strip()

        if not query:
            continue

        if query.lower() in ['quit', 'exit', 'q']:
            print("\n종료합니다.")
            break

        if query.lower() == 'info':
            show_collection_info()
            continue

        if query.lower() == 'docs':
            if last_docs:
                print("\n[마지막 검색 문서]")
                for i, doc in enumerate(last_docs, 1):
                    print(f"\n--- 문서 {i} (유사도: {doc['similarity']:.4f}) ---")
                    print(f"제목: {doc['title']}")
                    print(f"카테고리: {doc['category']}")
                    print(f"분류: {doc['classification']}")
                    print(f"내용:\n{doc['content'][:500]}...")
            else:
                print("[INFO] 검색된 문서가 없습니다. 먼저 질문을 입력하세요.")
            continue

        # 관련 문서 검색
        print("\n[검색 중...]")
        docs = search_relevant_docs(query, n_results=5)

        if not docs:
            print("[ERROR] 관련 문서를 찾을 수 없습니다.")
            continue

        last_docs = docs

        # 검색 결과 요약
        print(f"[검색 완료] {len(docs)}개 관련 문서 발견")
        for i, doc in enumerate(docs[:3], 1):
            print(f"  {i}. [{doc['similarity']:.2f}] {doc['title'][:40]}...")

        # 답변 생성
        print("\n[답변 생성 중...]")
        try:
            answer = generate_answer(query, docs)
            print("\n" + "-"*60)
            print("[답변]")
            print("-"*60)
            print(answer)
            print("-"*60)
        except Exception as e:
            print(f"[ERROR] 답변 생성 실패: {e}")


def single_query(query: str):
    """단일 질문 처리"""
    print(f"\n[질문] {query}")
    print("\n[검색 중...]")

    docs = search_relevant_docs(query, n_results=5)

    if not docs:
        print("[ERROR] 관련 문서를 찾을 수 없습니다.")
        return

    print(f"[검색 완료] {len(docs)}개 관련 문서 발견")
    for i, doc in enumerate(docs[:3], 1):
        print(f"  {i}. [{doc['similarity']:.2f}] {doc['title'][:40]}...")

    print("\n[답변 생성 중...]")
    try:
        answer = generate_answer(query, docs)
        print("\n" + "-"*60)
        print("[답변]")
        print("-"*60)
        print(answer)
        print("-"*60)
    except Exception as e:
        print(f"[ERROR] 답변 생성 실패: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # 커맨드라인에서 질문 입력
        query = " ".join(sys.argv[1:])
        single_query(query)
    else:
        # 대화형 모드
        chat_mode()
