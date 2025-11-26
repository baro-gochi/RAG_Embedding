# 임베딩 단계별 테스트 파일

# 1단계 :PDF 파일 텍스트 추출 방법
# 2단계: 텍스트 데이터 청킹
# 3단계: ChromaDB 임베딩 저장

from unstructured.partition.pdf import partition_pdf
import re

testfile = "src/(공유)KT+디즈니++이용약관.pdf"

# pdf 구역별 텍스트 추출
elements = partition_pdf(
    filename=testfile,
    strategy="hi_res",
    infer_table_structure=True,
    languages=["kor"]
)

# PDF에서 추출된 elements 청킹
def chunk_by_article(elements, file_name):
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
        
        # 장(Chapter) 감지
        if chapter_pattern.search(text):
            current_chapter = text
            continue
        
        # 조(Article) 감지 → 새 청크 시작
        if article_pattern.search(text):
            # 이전 청크 저장
            if current_chunk["article_title"]:
                current_chunk["content"] = "\n".join(current_chunk["content"])
                chunks.append(current_chunk)
            
            # 새 청크 시작
            current_chunk = {
                "article_title": text,
                "chapter": current_chapter,
                "content": [],
                "metadata": {
                    "category": file_name,
                    "page_number": el.metadata.page_number if hasattr(el.metadata, 'page_number') else None
                }
            }
        else:
            # 현재 청크에 내용 추가
            if text:
                current_chunk["content"].append(text)
    
    # 마지막 청크 저장
    if current_chunk["article_title"]:
        current_chunk["content"] = "\n".join(current_chunk["content"])
        chunks.append(current_chunk)
    
    return chunks

chunks = chunk_by_article(elements, testfile)

# 요소별로 Title, NarrativeText, Table 등으로 분류됨
with open("my_file.txt", "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        f.write(f"=== Chunk {i+1} ===\n")
        f.write(f"파일명/페이지: {chunk['metadata']['category']} {chunk['metadata']['page_number']}\n")
        f.write(f"장: {chunk['chapter']}\n")
        f.write(f"조: {chunk['article_title']}\n")
        f.write(f"내용: {chunk['content']}\n\n")