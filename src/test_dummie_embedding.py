"""
KT 상담사 지원 AI 어시스턴트
- 상담사가 고객 상담 시 필요한 정보를 실시간으로 제공
- RAG 기반으로 관련 상품/서비스 정보 검색
- 상담 키워드, 가격 정보, 할인 조건 등을 상담사에게 제공
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
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH_DUMMIE")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
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


def search_relevant_docs(
    query: str,
    n_results: int = 5,
    filter_intent: str = None,
    filter_document_type: str = None
):
    """질문과 관련된 문서 검색 (상담 메타데이터 필터링 지원)"""
    collection = get_collection()
    if not collection:
        return []

    # 필터 조건 구성
    where_filter = None
    if filter_intent or filter_document_type:
        conditions = []
        if filter_intent:
            conditions.append({"customer_intents": {"$contains": filter_intent}})
        if filter_document_type:
            conditions.append({"document_type": filter_document_type})

        if len(conditions) == 1:
            where_filter = conditions[0]
        else:
            where_filter = {"$and": conditions}

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter,
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

            # 상담 메타데이터 파싱
            price_info = {}
            discount_info = {}
            hypothetical_queries = []

            try:
                if meta.get('price_info'):
                    price_info = json.loads(meta.get('price_info', '{}'))
            except:
                pass

            try:
                if meta.get('discount_info'):
                    discount_info = json.loads(meta.get('discount_info', '{}'))
            except:
                pass

            try:
                if meta.get('hypothetical_queries'):
                    hypothetical_queries = json.loads(meta.get('hypothetical_queries', '[]'))
            except:
                pass

            docs.append({
                "content": doc,
                "title": meta.get('title', ''),
                "category": meta.get('category', ''),
                "classification": meta.get('classification', ''),
                "similarity": similarity,
                # 상담 특화 메타데이터
                "document_type": meta.get('document_type', ''),
                "customer_intents": meta.get('customer_intents', ''),
                "target_customers": meta.get('target_customers', ''),
                "consultation_keywords": meta.get('consultation_keywords', ''),
                "price_info": price_info,
                "discount_info": discount_info,
                "conditions": meta.get('conditions', ''),
                "required_documents": meta.get('required_documents', ''),
                "hypothetical_queries": hypothetical_queries,
                "has_contextual": meta.get('has_contextual', False),
                "has_hyde": meta.get('has_hyde', False)
            })

    return docs


def generate_answer(query: str, context_docs: list):
    """검색된 문서를 기반으로 상담사 지원 정보 생성"""

    # 컨텍스트 구성 (상담 메타데이터 포함)
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        # 메타데이터 정보 추가
        meta_info = []
        if doc.get('document_type'):
            meta_info.append(f"문서유형: {doc['document_type']}")
        if doc.get('customer_intents'):
            meta_info.append(f"관련의도: {doc['customer_intents']}")
        if doc.get('price_info'):
            prices = []
            for name, info in list(doc['price_info'].items())[:2]:
                if isinstance(info, dict) and 'monthly_price' in info:
                    prices.append(f"{name}: {info['monthly_price']:,}원")
            if prices:
                meta_info.append(f"가격정보: {', '.join(prices)}")

        meta_str = f" ({', '.join(meta_info)})" if meta_info else ""

        context_parts.append(f"""
[참고문서 {i}] (유사도: {doc['similarity']:.2f}){meta_str}
제목: {doc['title']}
분류: {doc['classification']}
내용:
{doc['content'][:1500]}
""")

    context = "\n".join(context_parts)

    # GPT로 상담사 지원 정보 생성
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """당신은 KT 고객센터 상담사를 지원하는 AI 어시스턴트입니다.
상담사가 고객과 상담할 때 필요한 정보를 정리하여 제공하세요.

역할:
- 고객에게 직접 답변하는 것이 아니라, 상담사가 고객에게 안내할 때 참고할 정보를 제공합니다.
- 상담사가 빠르게 핵심 정보를 파악할 수 있도록 구조화된 형태로 제공하세요.
- 고객의 문의 유형을 파악하고, 상담을 어떻게 진행해야 하는지 가이드를 제공하세요.

제공 형식:
1. [문의 유형 분석] - 고객 문의의 핵심 의도 파악 (예: 요금제 변경, 해지 방어, 할인 문의 등)
2. [상담 진행 가이드] - 이 상담을 어떻게 진행해야 하는지 단계별 안내
   - 먼저 확인할 사항 (고객 정보, 현재 사용 중인 요금제/서비스 등)
   - 추천 상담 흐름 (어떤 순서로 안내할지)
   - 예상되는 추가 질문과 대응 방안
3. [핵심 정보] - 고객 문의에 대한 핵심 답변 포인트 (2-3줄)
4. [상세 내용] - 상담사가 참고할 상세 정보
5. [요금/할인 정보] - 관련 요금제, 할인 조건 등 (해당 시)
6. [주의사항] - 상담 시 주의할 점이나 추가 확인 필요사항
7. [추천 멘트] - 고객에게 안내할 때 사용할 수 있는 예시 멘트 (상황별 2-3개)

규칙:
- 참고문서에 있는 정보만 사용하세요.
- 숫자 정보(요금, 할인율 등)는 정확하게 제공하세요.
- 정보가 없는 항목은 "확인 필요"로 표시하세요.
- 상담사가 빠르게 읽을 수 있도록 간결하게 작성하세요.
- 고객 유형(신규/기존, 약정 상태 등)에 따라 다른 접근법을 제안하세요."""
            },
            {
                "role": "user",
                "content": f"""[참고문서]
{context}

[고객 문의 내용]
{query}

위 참고문서를 바탕으로 상담사가 고객 응대에 활용할 정보를 정리해주세요."""
            }
        ],
        temperature=0.3,
        max_tokens=1000
    )

    return response.choices[0].message.content


def show_collection_info():
    """컬렉션 정보 표시 (상담 메타데이터 통계 포함)"""
    collection = get_collection()
    if not collection:
        print("[ERROR] Collection not found")
        return

    count = collection.count()
    print(f"\n[Collection Info]")
    print(f"  - Name: {COLLECTION_NAME}")
    print(f"  - Documents: {count}")
    print(f"  - DB Path: {CHROMA_DB_PATH}")

    # 샘플 문서로 메타데이터 확인
    if count > 0:
        sample = collection.peek(5)
        if sample and sample.get('metadatas'):
            print(f"\n[Metadata Sample]")
            meta = sample['metadatas'][0]

            # 상담 특화 메타데이터 표시
            print(f"  - Document Type: {meta.get('document_type', 'N/A')}")
            print(f"  - Customer Intents: {meta.get('customer_intents', 'N/A')[:50]}...")
            print(f"  - Has Contextual: {meta.get('has_contextual', False)}")
            print(f"  - Has HyDE: {meta.get('has_hyde', False)}")


def show_document_types():
    """문서 유형별 통계 표시"""
    collection = get_collection()
    if not collection:
        return

    # 전체 문서 조회 (최대 1000개)
    count = min(collection.count(), 1000)
    if count == 0:
        print("[INFO] No documents in collection")
        return

    results = collection.get(limit=count, include=["metadatas"])

    doc_types = {}
    intents = {}

    for meta in results['metadatas']:
        # 문서 유형 집계
        doc_type = meta.get('document_type', 'unknown')
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

        # 고객 의도 집계
        intent_str = meta.get('customer_intents', '')
        for intent in intent_str.split(', '):
            if intent:
                intents[intent] = intents.get(intent, 0) + 1

    print("\n[Document Type Statistics]")
    for dtype, cnt in sorted(doc_types.items(), key=lambda x: -x[1]):
        print(f"  - {dtype}: {cnt}")

    print("\n[Customer Intent Statistics]")
    for intent, cnt in sorted(intents.items(), key=lambda x: -x[1])[:10]:
        print(f"  - {intent}: {cnt}")


def chat_mode():
    """상담사 지원 모드"""
    print("\n" + "="*60)
    print("  KT 상담사 지원 AI 어시스턴트")
    print("  - 고객 문의 내용을 입력하면 상담에 필요한 정보를 제공합니다")
    print("="*60)

    show_collection_info()

    print("\n[사용법]")
    print("  - 고객의 문의 내용을 입력하세요.")
    print("  - AI가 상담에 필요한 정보를 정리하여 제공합니다.")
    print("  - 'quit' 또는 'exit' 입력 시 종료")
    print("  - 'info' 입력 시 DB 정보 표시")
    print("  - 'types' 입력 시 문서 유형별 통계 표시")
    print("  - 'docs' 입력 시 마지막 검색 문서 상세 표시")
    print("  - 'meta' 입력 시 마지막 검색 문서 메타데이터 표시")
    print("  - 'filter:의도명' 입력 시 해당 의도로 필터링 검색")
    print("    예: filter:요금제문의 5G 요금제 알려주세요")
    print("="*60)

    last_docs = []
    current_filter_intent = None
    current_filter_doctype = None

    while True:
        print()
        query = input("[고객 문의] > ").strip()

        if not query:
            continue

        if query.lower() in ['quit', 'exit', 'q']:
            print("\n종료합니다.")
            break

        if query.lower() == 'info':
            show_collection_info()
            continue

        if query.lower() == 'types':
            show_document_types()
            continue

        if query.lower() == 'docs':
            if last_docs:
                print("\n[마지막 검색 문서]")
                for i, doc in enumerate(last_docs, 1):
                    print(f"\n--- 문서 {i} (유사도: {doc['similarity']:.4f}) ---")
                    print(f"제목: {doc['title']}")
                    print(f"카테고리: {doc['category']}")
                    print(f"분류: {doc['classification']}")
                    print(f"문서유형: {doc['document_type']}")
                    print(f"내용:\n{doc['content'][:500]}...")
            else:
                print("[INFO] 검색된 문서가 없습니다. 먼저 질문을 입력하세요.")
            continue

        if query.lower() == 'meta':
            if last_docs:
                print("\n[마지막 검색 문서 메타데이터]")
                for i, doc in enumerate(last_docs, 1):
                    print(f"\n--- 문서 {i} ---")
                    print(f"  제목: {doc['title'][:50]}...")
                    print(f"  문서유형: {doc['document_type']}")
                    print(f"  고객의도: {doc['customer_intents']}")
                    print(f"  대상고객: {doc['target_customers']}")
                    print(f"  상담키워드: {doc['consultation_keywords']}")
                    print(f"  조건: {doc['conditions']}")
                    print(f"  필요서류: {doc['required_documents']}")
                    if doc['price_info']:
                        print(f"  가격정보: {json.dumps(doc['price_info'], ensure_ascii=False)[:200]}")
                    if doc['discount_info']:
                        print(f"  할인정보: {json.dumps(doc['discount_info'], ensure_ascii=False)[:200]}")
                    if doc['hypothetical_queries']:
                        print(f"  예상질문: {doc['hypothetical_queries'][:3]}")
                    print(f"  Contextual: {doc['has_contextual']}, HyDE: {doc['has_hyde']}")
            else:
                print("[INFO] 검색된 문서가 없습니다. 먼저 질문을 입력하세요.")
            continue

        # 필터 파싱
        filter_intent = None
        filter_doctype = None
        actual_query = query

        if query.startswith('filter:'):
            parts = query.split(' ', 1)
            if len(parts) > 1:
                filter_part = parts[0].replace('filter:', '')
                actual_query = parts[1]

                # 의도 또는 문서유형 필터
                if filter_part in ['요금제문의', '요금제변경', '비용절감', '결합문의', '결합변경',
                                   '부가서비스추가', '부가서비스해지', '명의변경', '약정문의', '일반문의',
                                   '로밍', '혜택문의', '사용량확인', '요금문의']:
                    filter_intent = filter_part
                elif filter_part in ['mobile_plan', 'bundle', 'roaming', 'billing', 'tv_service',
                                     'name_change', 'consultation_guide', 'membership', 'internet', 'generic']:
                    filter_doctype = filter_part
                else:
                    filter_intent = filter_part

                print(f"[필터] 의도: {filter_intent}, 문서유형: {filter_doctype}")

        # 관련 문서 검색
        print("\n[검색 중...]")
        docs = search_relevant_docs(
            actual_query,
            n_results=5,
            filter_intent=filter_intent,
            filter_document_type=filter_doctype
        )

        if not docs:
            print("[ERROR] 관련 문서를 찾을 수 없습니다.")
            continue

        last_docs = docs

        # 검색 결과 요약
        print(f"[검색 완료] {len(docs)}개 관련 문서 발견")
        for i, doc in enumerate(docs[:3], 1):
            meta_hint = f" [{doc['document_type']}]" if doc['document_type'] else ""
            print(f"  {i}. [{doc['similarity']:.2f}]{meta_hint} {doc['title'][:40]}...")

        # 상담 지원 정보 생성
        print("\n[상담 지원 정보 생성 중...]")
        try:
            answer = generate_answer(actual_query, docs)
            print("\n" + "-"*60)
            print("[상담사 참고 정보]")
            print("-"*60)
            print(answer)
            print("-"*60)
        except Exception as e:
            print(f"[ERROR] 정보 생성 실패: {e}")


def single_query(query: str, filter_intent: str = None, filter_doctype: str = None):
    """단일 고객 문의 처리"""
    print(f"\n[고객 문의] {query}")
    if filter_intent:
        print(f"[필터 - 의도] {filter_intent}")
    if filter_doctype:
        print(f"[필터 - 문서유형] {filter_doctype}")

    print("\n[검색 중...]")

    docs = search_relevant_docs(
        query,
        n_results=5,
        filter_intent=filter_intent,
        filter_document_type=filter_doctype
    )

    if not docs:
        print("[ERROR] 관련 문서를 찾을 수 없습니다.")
        return

    print(f"[검색 완료] {len(docs)}개 관련 문서 발견")
    for i, doc in enumerate(docs[:3], 1):
        meta_hint = f" [{doc['document_type']}]" if doc['document_type'] else ""
        intent_hint = f" ({doc['customer_intents'][:30]})" if doc['customer_intents'] else ""
        print(f"  {i}. [{doc['similarity']:.2f}]{meta_hint}{intent_hint} {doc['title'][:40]}...")

    print("\n[상담 지원 정보 생성 중...]")
    try:
        answer = generate_answer(query, docs)
        print("\n" + "-"*60)
        print("[상담사 참고 정보]")
        print("-"*60)
        print(answer)
        print("-"*60)
    except Exception as e:
        print(f"[ERROR] 정보 생성 실패: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # 커맨드라인에서 질문 입력
        # 사용법: python test_dummie_embedding.py "질문"
        # 또는: python test_dummie_embedding.py --intent 요금제문의 "질문"
        # 또는: python test_dummie_embedding.py --doctype mobile_plan "질문"

        args = sys.argv[1:]
        filter_intent = None
        filter_doctype = None
        query_parts = []

        i = 0
        while i < len(args):
            if args[i] == '--intent' and i + 1 < len(args):
                filter_intent = args[i + 1]
                i += 2
            elif args[i] == '--doctype' and i + 1 < len(args):
                filter_doctype = args[i + 1]
                i += 2
            else:
                query_parts.append(args[i])
                i += 1

        query = " ".join(query_parts)
        if query:
            single_query(query, filter_intent, filter_doctype)
        else:
            print("사용법: python test_dummie_embedding.py [--intent 의도] [--doctype 문서유형] \"고객 문의 내용\"")
    else:
        # 대화형 모드
        chat_mode()
