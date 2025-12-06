import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv
import os

# 1. 설정
load_dotenv()
dir =  os.getenv("DIR")
api_key = os.getenv("API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# OpenAI 클라이언트
client = OpenAI(api_key=api_key)

# ChromaDB 임베딩 함수
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api_key,
    model_name="text-embedding-3-large"
)

# ChromaDB 클라이언트 & 컬렉션 연결
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=openai_ef
)


def search_documents(query, n_results=3):
    """ChromaDB에서 관련 문서 검색"""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results


def generate_answer(query, retrieved_docs):
    """검색된 문서를 기반으로 GPT-4o-mini가 답변 생성"""
    
    context = "\n\n".join(retrieved_docs)
    
    prompt = f"""다음 문서를 참고하여 질문에 답변해주세요.
문서에 없는 내용은 "문서에서 해당 정보를 찾을 수 없습니다"라고 답변해주세요.

[참고 문서]
{context}

[질문]
{query}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 문서를 기반으로 정확하게 답변하는 도우미입니다. 한국어로 답변합니다."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.3
    )
    
    return response.choices[0].message.content


def rag_query(query, n_results=3):
    """RAG 전체 파이프라인: 검색 → 답변 생성"""
    
    print(f"\n{'='*50}")
    print(f"질문: {query}")
    print('='*50)
    
    # 1. 문서 검색
    results = search_documents(query, n_results)
    retrieved_docs = results['documents'][0]
    
    # 2. 검색된 문서 출력
    print(f"\n📄 검색된 문서 ({len(retrieved_docs)}개):")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"\n[{i}] {doc}")
    # 3. 답변 생성
    answer = generate_answer(query, retrieved_docs)
    
    print(f"\n💬답변:")
    print(answer)
    print('='*50)
    
    return answer


if __name__ == "__main__":
    print("🔍 RAG 테스트 시스템")
    print("종료하려면 'q' 또는 'quit' 입력\n")
    
    while True:
        query = input("\n질문을 입력하세요: ").strip()
        
        if query.lower() in ['q', 'quit', '종료']:
            print("테스트를 종료합니다.")
            break
        
        if not query:
            print("질문을 입력해주세요.")
            continue
        
        rag_query(query)