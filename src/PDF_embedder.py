from unstructured.partition.pdf import partition_pdf
import re
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import os
from glob import glob
from dotenv import load_dotenv


# 환경설정
load_dotenv()
dir =  os.getenv("DIR")
api_key = os.getenv("API_KEY")


def get_pdf_files(directory):
    """디렉터리 내 모든 PDF 파일 경로 반환"""
    pdf_pattern = os.path.join(directory, "**", "*.pdf")
    pdf_files = glob(pdf_pattern, recursive=True)  # 하위 폴더 포함
    
    print(f"📁 디렉터리: {directory}")
    print(f"📄 발견된 PDF 파일: {len(pdf_files)}개")
    for f in pdf_files:
        print(f"   - {f}")
    
    return pdf_files

# pdf 구역별 텍스트 추출
def extract_elements(file_path):
    """PDF에서 elements 추출"""
    print(f"\n🔄 처리 중: {file_path}")
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True,
        languages=["kor"]
    )
    return elements

# PDF에서 추출된 elements 청킹
def chunk_by_article(elements, file_name, file_path):
    """조(Article) 단위로 청킹"""
    chunks = []
    current_chunk = {
        "article_title": None,
        "chapter": None,
        "content": [],
        "metadata": {}
    }
    
    chapter_pattern = re.compile(r"제\s*\d+\s*장")
    article_pattern = re.compile(r"제\s*\d+\s*조")
    
    current_chapter = None
    
    for el in elements:
        text = el.text.strip()
        
        if chapter_pattern.search(text):
            current_chapter = text
            continue
        
        if article_pattern.search(text):
            if current_chunk["article_title"]:
                current_chunk["content"] = "\n".join(current_chunk["content"])
                chunks.append(current_chunk)
            
            current_chunk = {
                "article_title": text,
                "chapter": current_chapter,
                "content": [],
                "metadata": {
                    "category": file_name,
                    "source": file_path,
                    "page_number": el.metadata.page_number if hasattr(el.metadata, 'page_number') else None
                }
            }
        else:
            if text:
                current_chunk["content"].append(text)
    
    if current_chunk["article_title"]:
        current_chunk["content"] = "\n".join(current_chunk["content"])
        chunks.append(current_chunk)
    
    return chunks

# ChromaDB 설정
def setup_chromadb(api_key):
    """ChromaDB 컬렉션 설정"""
    
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small"
    )
    
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    collection = chroma_client.get_or_create_collection(
        name="aicc_documents",
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"}
    )
    
    return collection

# 임베딩 및 저장
def insert_chunks_to_chroma(chunks, collection):
    """청크를 ChromaDB에 저장"""
    if not chunks:
        print("⚠️ 저장할 청크가 없습니다.")
        return
    
    documents = []
    metadatas = []
    ids = []
    
    for i, chunk in enumerate(chunks):
        doc_text = f"{chunk['article_title']}\n{chunk['content']}"
        documents.append(doc_text)
        
        metadatas.append({
            "category": chunk["metadata"]["category"],
            "chapter": chunk["chapter"] or "",
            "article_title": chunk["article_title"],
            "source": chunk["metadata"]["source"],
            "page_number": chunk["metadata"]["page_number"] or 0
        })
        
        ids.append(f"{chunk['metadata']['category']}_{i}")
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"   ✅ {len(documents)}개 청크 저장 완료")

# 메인 실행
def process_all_pdfs(directory, api_key):
    """디렉터리 내 모든 PDF 처리"""
    
    # PDF 파일 목록 가져오기
    pdf_files = get_pdf_files(directory)
    
    if not pdf_files:
        print("❌ PDF 파일이 없습니다.")
        return
    
    # ChromaDB 설정
    collection = setup_chromadb(api_key)
    
    # 각 PDF 파일 처리
    total_chunks = 0
    for file_path in pdf_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # 추출
        elements = extract_elements(file_path)
        
        # 청킹
        chunks = chunk_by_article(elements, file_name, file_path)
        print(f"   📝 {len(chunks)}개 청크 생성")
        
        # 저장
        insert_chunks_to_chroma(chunks, collection)
        total_chunks += len(chunks)
    
    print(f"\n{'='*50}")
    print(f"🎉 전체 처리 완료!")
    print(f"📄 처리된 PDF: {len(pdf_files)}개")
    print(f"📊 총 청크 수: {total_chunks}개")
    print(f"💾 저장된 문서 수: {collection.count()}개")


if __name__ == "__main__":
    process_all_pdfs(dir, api_key)
