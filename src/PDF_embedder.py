from unstructured.partition.pdf import partition_pdf
import re
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import os
from glob import glob
from dotenv import load_dotenv
import json
import tiktoken

# 환경설정
load_dotenv()
PDF_DIRECTORY =  os.getenv("DIR")
OPENAI_API_KEY = os.getenv("API_KEY")
CLASSIFICATION_CATEGORIES = os.getenv("CATEGORIES")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# 청킹 설정
MAX_CHUNK_SIZE = 1500
MIN_CHUNK_SIZE = 30
CHUNK_OVERLAP = 100

# 임베딩 설정
EMBEDDING_MODEL = "text-embedding-3-large"
MAX_EMBEDDING_TOKENS = 8000

# ========== 토큰 관련 유틸리티 ==========
def count_tokens(text, model="text-embedding-3-large"):
    """텍스트의 토큰 수 계산"""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # 대략적인 계산 (한글 기준 약 1.5자당 1토큰)
        return len(text) // 2


def truncate_text(text, max_tokens=8000):
    """토큰 제한에 맞게 텍스트 자르기"""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        if len(tokens) > max_tokens:
            truncated_tokens = tokens[:max_tokens]
            return encoding.decode(truncated_tokens)
        return text
    except Exception:
        # 대략적인 자르기
        max_chars = max_tokens * 2
        return text[:max_chars]


def split_text_by_tokens(text, max_tokens=8000, overlap_tokens=200):
    """
    토큰 제한에 맞게 텍스트를 여러 청크로 분할
    - 오버랩을 적용하여 문맥 유지
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        
        # 토큰 수가 제한 이하면 그대로 반환
        if len(tokens) <= max_tokens:
            return [text]
        
        # 토큰 단위로 분할
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # 다음 시작점 (오버랩 적용)
            start = end - overlap_tokens if end < len(tokens) else end
        
        return chunks
        
    except Exception as e:
        print(f"   ⚠️ 토큰 분할 실패: {e}")
        # 대략적인 문자 기반 분할
        max_chars = max_tokens * 2
        overlap_chars = overlap_tokens * 2
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunks.append(text[start:end])
            start = end - overlap_chars if end < len(text) else end
        
        return chunks


def clean_text(text):
    """텍스트 정제 (빈 문자열, 특수문자 처리)"""
    if not text:
        return ""
    # None 체크
    if text is None:
        return ""
    # 공백만 있는 경우
    text = str(text).strip()
    if not text:
        return ""
    # 제어 문자 제거
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    return text


client = OpenAI(api_key=OPENAI_API_KEY)

# ========== 1. PDF 파일 목록 가져오기 ==========
def get_pdf_files(directory):
    """디렉터리 내 모든 PDF 파일 경로 반환"""
    pdf_pattern = os.path.join(directory, "**", "*.pdf")
    pdf_files = glob(pdf_pattern, recursive=True)
    
    print(f"📁 디렉터리: {directory}")
    print(f"📄 발견된 PDF 파일: {len(pdf_files)}개")
    for f in pdf_files:
        print(f"   - {f}")
    
    return pdf_files


# ========== 2. PDF 추출 ==========
def extract_elements(file_path):
    """PDF에서 elements 추출"""
    print(f"\n🔄 처리 중: {file_path}")
    try:
        elements = partition_pdf(
            filename=file_path,
            strategy="hi_res",
            infer_table_structure=True,
            languages=["kor"]
        )
        return elements
    except Exception as e:
        print(f"   ⚠️ PDF 추출 실패: {e}")
        return []


# ========== 3. 문서 구조 감지 ==========
def detect_document_structure(elements):
    """문서 구조를 분석하여 적합한 청킹 방식 결정"""
    full_text = " ".join([el.text for el in elements if hasattr(el, 'text')])
    
    # 패턴 정의
    patterns = {
        "article": re.compile(r"제\s*\d+\s*조"),           # 제1조, 제 2 조
        "chapter": re.compile(r"제\s*\d+\s*장"),           # 제1장, 제 2 장
        "number_dot": re.compile(r"^\d+\.\s", re.MULTILINE),  # 1. 2. 3.
        "korean_number": re.compile(r"^[가-힣]\.\s", re.MULTILINE),  # 가. 나. 다.
        "qa_pattern": re.compile(r"(Q\d*[\.:]\s*|A\d*[\.:]\s*|질문\s*:|\답변\s*:)", re.MULTILINE),  # Q: A: 질문: 답변:
        "bracket_number": re.compile(r"[\[【]\d+[\]】]"),   # [1] 【2】
    }
    
    # 각 패턴 매칭 횟수 계산
    matches = {}
    for name, pattern in patterns.items():
        matches[name] = len(pattern.findall(full_text))
    
    # Title 요소 개수 확인
    title_count = sum(1 for el in elements if hasattr(el, 'category') and el.category == "Title")
    matches["title_elements"] = title_count
    
    print(f"   📊 구조 분석: {matches}")
    
    # 청킹 방식 결정
    if matches["article"] >= 3:
        return "article"  # 조/장 기반
    elif matches["qa_pattern"] >= 3:
        return "qa"  # Q&A 기반
    elif title_count >= 3:
        return "title"  # 제목 요소 기반
    elif matches["number_dot"] >= 5 or matches["korean_number"] >= 5:
        return "numbered"  # 번호 기반
    elif matches["bracket_number"] >= 3:
        return "bracket"  # 괄호 번호 기반
    else:
        return "semantic"  # 시맨틱 (폴백)


# ========== 4. 조/장 기반 청킹 ==========
def chunk_by_article(elements, file_name, file_path):
    """조(Article) 단위로 청킹"""
    chunks = []
    current_chunk = {
        "title": None,
        "chapter": None,
        "content": [],
        "metadata": {}
    }
    
    chapter_pattern = re.compile(r"제\s*\d+\s*장")
    article_pattern = re.compile(r"제\s*\d+\s*조")
    
    current_chapter = None
    
    for el in elements:
        if not hasattr(el, 'text'):
            continue
        text = el.text.strip()
        if not text:
            continue
        
        if chapter_pattern.search(text):
            current_chapter = text
            continue
        
        if article_pattern.search(text):
            if current_chunk["title"] and current_chunk["content"]:
                current_chunk["content"] = "\n".join(current_chunk["content"])
                chunks.append(current_chunk)
            
            current_chunk = {
                "title": text,
                "chapter": current_chapter,
                "content": [],
                "metadata": {
                    "category": file_name,
                    "source": file_path,
                    "page_number": el.metadata.page_number if hasattr(el.metadata, 'page_number') else None,
                    "chunk_type": "article"
                }
            }
        else:
            current_chunk["content"].append(text)
    
    if current_chunk["title"] and current_chunk["content"]:
        current_chunk["content"] = "\n".join(current_chunk["content"])
        chunks.append(current_chunk)
    
    return chunks


# ========== 5. 제목(Title) 요소 기반 청킹 ==========
def chunk_by_title(elements, file_name, file_path):
    """Unstructured의 Title 요소 기준으로 청킹"""
    chunks = []
    current_chunk = {
        "title": None,
        "chapter": None,
        "content": [],
        "metadata": {}
    }
    
    for el in elements:
        if not hasattr(el, 'text'):
            continue
        text = el.text.strip()
        if not text:
            continue
        
        # Title 요소를 새 청크의 시작점으로
        if hasattr(el, 'category') and el.category == "Title":
            if current_chunk["content"]:
                current_chunk["content"] = "\n".join(current_chunk["content"])
                if not current_chunk["title"]:
                    current_chunk["title"] = current_chunk["content"][:50] + "..."
                chunks.append(current_chunk)
            
            current_chunk = {
                "title": text,
                "chapter": None,
                "content": [],
                "metadata": {
                    "category": file_name,
                    "source": file_path,
                    "page_number": el.metadata.page_number if hasattr(el.metadata, 'page_number') else None,
                    "chunk_type": "title"
                }
            }
        else:
            current_chunk["content"].append(text)
    
    if current_chunk["content"]:
        current_chunk["content"] = "\n".join(current_chunk["content"])
        if not current_chunk["title"]:
            current_chunk["title"] = current_chunk["content"][:50] + "..."
        chunks.append(current_chunk)
    
    return chunks


# ========== 6. Q&A 기반 청킹 ==========
def chunk_by_qa(elements, file_name, file_path):
    """Q&A 패턴 기준으로 청킹"""
    chunks = []
    full_text = "\n".join([el.text for el in elements if hasattr(el, 'text') and el.text])
    
    # Q&A 패턴으로 분할
    qa_pattern = re.compile(r'(Q\d*[\.:]\s*|질문\s*[\d]*[\.:]*\s*)', re.MULTILINE | re.IGNORECASE)
    parts = qa_pattern.split(full_text)
    
    current_q = None
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        
        if qa_pattern.match(part + " "):
            continue
        
        # Q로 시작하는 부분 찾기
        if i > 0 and qa_pattern.match(parts[i-1] if i-1 < len(parts) else ""):
            current_q = part
        elif current_q:
            # Q와 A를 합쳐서 하나의 청크로
            chunk = {
                "title": f"Q: {current_q[:50]}..." if len(current_q) > 50 else f"Q: {current_q}",
                "chapter": None,
                "content": f"질문: {current_q}\n답변: {part}",
                "metadata": {
                    "category": file_name,
                    "source": file_path,
                    "page_number": None,
                    "chunk_type": "qa"
                }
            }
            chunks.append(chunk)
            current_q = None
    
    # Q&A 패턴이 제대로 작동하지 않으면 시맨틱 청킹으로 폴백
    if len(chunks) < 2:
        return chunk_semantic(elements, file_name, file_path)
    
    return chunks


# ========== 7. 번호 기반 청킹 ==========
def chunk_by_number(elements, file_name, file_path):
    """번호 패턴 (1. 2. 가. 나.) 기준으로 청킹"""
    chunks = []
    full_text = "\n".join([el.text for el in elements if hasattr(el, 'text') and el.text])
    
    # 번호 패턴으로 분할
    number_pattern = re.compile(r'\n(?=\d+\.\s|[가-힣]\.\s)')
    parts = number_pattern.split(full_text)
    
    for i, part in enumerate(parts):
        part = part.strip()
        if len(part) < MIN_CHUNK_SIZE:
            continue
        
        # 첫 줄을 제목으로 사용
        lines = part.split('\n')
        title = lines[0][:50] + "..." if len(lines[0]) > 50 else lines[0]
        
        chunk = {
            "title": title,
            "chapter": None,
            "content": part,
            "metadata": {
                "category": file_name,
                "source": file_path,
                "page_number": None,
                "chunk_type": "numbered"
            }
        }
        chunks.append(chunk)
    
    if len(chunks) < 2:
        return chunk_semantic(elements, file_name, file_path)
    
    return chunks


# ========== 8. 시맨틱 청킹 (폴백) ==========
def chunk_semantic(elements, file_name, file_path):
    """고정 크기 + 문단 기반 시맨틱 청킹"""
    chunks = []
    full_text = "\n".join([el.text for el in elements if hasattr(el, 'text') and el.text])
    
    if not full_text.strip():
        return chunks
    
    # 문단 단위로 분할
    paragraphs = re.split(r'\n\s*\n', full_text)
    
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        para_length = len(para)
        
        # 현재 청크 + 새 문단이 최대 크기를 초과하면
        if current_length + para_length > MAX_CHUNK_SIZE and current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            title = chunk_text[:50] + "..." if len(chunk_text) > 50 else chunk_text
            
            chunk = {
                "title": title,
                "chapter": None,
                "content": chunk_text,
                "metadata": {
                    "category": file_name,
                    "source": file_path,
                    "page_number": None,
                    "chunk_type": "semantic"
                }
            }
            chunks.append(chunk)
            
            # 오버랩 적용
            current_chunk = [para]
            current_length = para_length
        else:
            current_chunk.append(para)
            current_length += para_length
    
    # 마지막 청크 처리
    if current_chunk:
        chunk_text = "\n\n".join(current_chunk)
        if len(chunk_text) >= MIN_CHUNK_SIZE:
            title = chunk_text[:50] + "..." if len(chunk_text) > 50 else chunk_text
            
            chunk = {
                "title": title,
                "chapter": None,
                "content": chunk_text,
                "metadata": {
                    "category": file_name,
                    "source": file_path,
                    "page_number": None,
                    "chunk_type": "semantic"
                }
            }
            chunks.append(chunk)
    
    return chunks


# ========== 9. 하이브리드 청킹 (메인) ==========
def chunk_hybrid(elements, file_name, file_path):
    """문서 구조에 따라 적합한 청킹 방식 자동 선택"""
    
    # 구조 감지
    structure_type = detect_document_structure(elements)
    print(f"   🔍 감지된 구조: {structure_type}")
    
    # 구조에 맞는 청킹 방식 적용
    if structure_type == "article":
        chunks = chunk_by_article(elements, file_name, file_path)
    elif structure_type == "title":
        chunks = chunk_by_title(elements, file_name, file_path)
    elif structure_type == "qa":
        chunks = chunk_by_qa(elements, file_name, file_path)
    elif structure_type == "numbered" or structure_type == "bracket":
        chunks = chunk_by_number(elements, file_name, file_path)
    else:
        chunks = chunk_semantic(elements, file_name, file_path)
    
    # 청크가 없으면 시맨틱으로 폴백
    if not chunks:
        print(f"   ⚠️ {structure_type} 청킹 실패, 시맨틱 청킹으로 전환")
        chunks = chunk_semantic(elements, file_name, file_path)
    
    return chunks


# ========== 10. 키워드 추출 (GPT 활용) ==========
def extract_keywords(text, max_keywords=5):
    """GPT를 활용하여 핵심 키워드 추출"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """당신은 고객센터 상담 문서에서 핵심 키워드를 추출하는 전문가입니다.
주어진 텍스트에서 고객이 검색할 때 사용할 만한 핵심 키워드를 추출하세요.
- 명사 위주로 추출
- 동의어, 유사어도 포함 (예: 해지 → 취소, 끊기)
- 구어체 표현도 포함 (예: 환불 → 돈 돌려받기)

JSON 형식으로만 응답하세요: {"keywords": ["키워드1", "키워드2", ...]}"""
                },
                {
                    "role": "user",
                    "content": f"다음 텍스트에서 핵심 키워드를 {max_keywords}개 추출하세요:\n\n{text[:1000]}"
                }
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get("keywords", [])
    
    except Exception as e:
        print(f"   ⚠️ 키워드 추출 실패: {e}")
        return []


# ========== 11. 분류 자동 생성 (GPT 활용) ==========
def classify_content(text, categories=CLASSIFICATION_CATEGORIES):
    """GPT를 활용하여 콘텐츠 분류"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"""당신은 고객센터 문서를 분류하는 전문가입니다.
주어진 텍스트를 다음 카테고리 중 가장 적합한 것으로 분류하세요.

카테고리: {categories}

JSON 형식으로만 응답하세요: {{"classification": "카테고리명", "confidence": 0.0~1.0}}"""
                },
                {
                    "role": "user",
                    "content": f"다음 텍스트를 분류하세요:\n\n{text[:1000]}"
                }
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get("classification", "기타"), result.get("confidence", 0.0)
    
    except Exception as e:
        print(f"   ⚠️ 분류 실패: {e}")
        return "기타", 0.0


# ========== 12. 키워드 & 분류 추가 ==========
def enrich_chunks(chunks, extract_keywords_flag=True, classify_flag=True):
    """청크에 키워드와 분류 추가"""
    for i, chunk in enumerate(chunks):
        full_text = f"{chunk['title']}\n{chunk['content']}"
        
        # 키워드 추출
        if extract_keywords_flag:
            keywords = extract_keywords(full_text)
            chunk["keywords"] = keywords
            print(f"      청크 {i+1} 키워드: {keywords}")
        else:
            chunk["keywords"] = []
        
        # 분류
        if classify_flag:
            classification, confidence = classify_content(full_text)
            chunk["classification"] = classification
            chunk["classification_confidence"] = confidence
            print(f"      청크 {i+1} 분류: {classification} (신뢰도: {confidence:.2f})")
        else:
            chunk["classification"] = "기타"
            chunk["classification_confidence"] = 0.0
    
    return chunks


# ========== 13. ChromaDB 설정 ==========
def setup_chromadb(api_key):
    """ChromaDB 컬렉션 설정"""
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBEDDING_MODEL  # 설정에서 모델명 가져오기
    )
    
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"}
    )
    
    return collection


# ========== 14. 임베딩 및 저장 ==========
def insert_chunks_to_chroma(chunks, collection):
    """청크를 ChromaDB에 저장 (토큰 초과 시 분할 저장)"""
    if not chunks:
        print("   ⚠️ 저장할 청크가 없습니다.")
        return
    
    documents = []
    metadatas = []
    ids = []
    skipped = 0
    split_count = 0  # 분할된 청크 수
    
    for i, chunk in enumerate(chunks):
        keywords_str = ", ".join(chunk.get("keywords", []))
        classification = chunk.get("classification", "기타")
        
        # 텍스트 정제
        title = clean_text(chunk.get('title', ''))
        content = clean_text(chunk.get('content', ''))
        
        # 빈 콘텐츠 스킵
        if not content:
            print(f"   ⚠️ 청크 {i+1} 스킵: 빈 콘텐츠")
            skipped += 1
            continue
        
        # 임베딩 텍스트 구성
        doc_text = f"""[분류: {classification}]
[키워드: {keywords_str}]
{title}
{content}"""
        
        # 토큰 수 체크
        token_count = count_tokens(doc_text)
        
        if token_count > MAX_EMBEDDING_TOKENS:
            # 토큰 초과 시 분할
            print(f"   📎 청크 {i+1} 토큰 초과 ({token_count} > {MAX_EMBEDDING_TOKENS}), 분할 진행")
            
            # 헤더 (분류, 키워드, 제목) 토큰 계산
            header = f"""[분류: {classification}]
[키워드: {keywords_str}]
{title}
"""
            header_tokens = count_tokens(header)
            
            # 콘텐츠만 분할 (헤더 토큰 제외한 크기로)
            content_max_tokens = MAX_EMBEDDING_TOKENS - header_tokens - 100  # 안전 마진
            content_chunks = split_text_by_tokens(content, max_tokens=content_max_tokens, overlap_tokens=200)
            
            print(f"      → {len(content_chunks)}개로 분할됨")
            split_count += len(content_chunks) - 1  # 원래 1개에서 추가된 수
            
            # 각 분할 청크 저장
            for j, content_part in enumerate(content_chunks):
                part_doc_text = f"""[분류: {classification}]
[키워드: {keywords_str}]
{title} (Part {j+1}/{len(content_chunks)})
{content_part}"""
                
                # 최종 빈 체크
                if not part_doc_text.strip():
                    continue
                
                documents.append(part_doc_text)
                
                metadatas.append({
                    "category": chunk["metadata"]["category"],
                    "chapter": chunk.get("chapter") or "",
                    "title": f"{title[:180]} (Part {j+1}/{len(content_chunks)})" if title else f"Part {j+1}/{len(content_chunks)}",
                    "source": chunk["metadata"]["source"],
                    "page_number": chunk["metadata"].get("page_number") or 0,
                    "chunk_type": chunk["metadata"].get("chunk_type", "unknown"),
                    "keywords": keywords_str[:500] if keywords_str else "",
                    "classification": classification,
                    "classification_confidence": chunk.get("classification_confidence", 0.0),
                    "is_split": True,
                    "split_part": j + 1,
                    "split_total": len(content_chunks)
                })
                
                ids.append(f"{chunk['metadata']['category']}_{i}_part{j}")
        
        else:
            # 토큰 제한 이내 - 그대로 저장
            # 최종 빈 체크
            if not doc_text.strip():
                print(f"   ⚠️ 청크 {i+1} 스킵: 정제 후 빈 텍스트")
                skipped += 1
                continue
            
            documents.append(doc_text)
            
            metadatas.append({
                "category": chunk["metadata"]["category"],
                "chapter": chunk.get("chapter") or "",
                "title": title[:200] if title else "",
                "source": chunk["metadata"]["source"],
                "page_number": chunk["metadata"].get("page_number") or 0,
                "chunk_type": chunk["metadata"].get("chunk_type", "unknown"),
                "keywords": keywords_str[:500] if keywords_str else "",
                "classification": classification,
                "classification_confidence": chunk.get("classification_confidence", 0.0),
                "is_split": False,
                "split_part": 0,
                "split_total": 1
            })
            
            ids.append(f"{chunk['metadata']['category']}_{i}")
    
    if not documents:
        print("   ⚠️ 유효한 문서가 없습니다.")
        return
    
    # 배치 처리 (한 번에 너무 많이 보내지 않기)
    batch_size = 100
    for start in range(0, len(documents), batch_size):
        end = min(start + batch_size, len(documents))
        try:
            collection.add(
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end]
            )
        except Exception as e:
            print(f"   ❌ 배치 {start}-{end} 저장 실패: {e}")
            # 개별 저장 시도
            for j in range(start, end):
                try:
                    collection.add(
                        documents=[documents[j]],
                        metadatas=[metadatas[j]],
                        ids=[ids[j]]
                    )
                except Exception as e2:
                    print(f"      ❌ 개별 청크 {j} 저장 실패: {e2}")
    
    print(f"   ✅ {len(documents)}개 청크 임베딩 및 저장 완료")
    print(f"      (원본: {len(chunks)}개, 분할 추가: {split_count}개, 스킵: {skipped}개)")


# ========== 15. 메인 실행 ==========
def process_all_pdfs(directory, api_key, extract_keywords_flag=True, classify_flag=True):
    """디렉터리 내 모든 PDF 처리"""
    
    pdf_files = get_pdf_files(directory)
    
    if not pdf_files:
        print("❌ PDF 파일이 없습니다.")
        return
    
    collection = setup_chromadb(api_key)
    
    total_chunks = 0
    success_files = 0
    failed_files = []
    
    for file_path in pdf_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        try:
            # 추출
            elements = extract_elements(file_path)
            
            if not elements:
                print(f"   ⚠️ 추출된 요소 없음")
                failed_files.append(file_path)
                continue
            
            # 하이브리드 청킹
            chunks = chunk_hybrid(elements, file_name, file_path)
            print(f"   📝 {len(chunks)}개 청크 생성")
            
            if not chunks:
                failed_files.append(file_path)
                continue
            
            # 키워드 & 분류 추가
            chunks = enrich_chunks(chunks, extract_keywords_flag, classify_flag)
            
            # 저장
            insert_chunks_to_chroma(chunks, collection)
            total_chunks += len(chunks)
            success_files += 1
            
        except Exception as e:
            print(f"   ❌ 처리 실패: {e}")
            failed_files.append(file_path)
    
    print(f"\n{'='*50}")
    print(f"🎉 전체 처리 완료!")
    print(f"📄 성공한 PDF: {success_files}개")
    print(f"❌ 실패한 PDF: {len(failed_files)}개")
    if failed_files:
        for f in failed_files:
            print(f"   - {f}")
    print(f"📊 총 청크 수: {total_chunks}개")
    print(f"💾 저장된 문서 수: {collection.count()}개")


# ========== 실행 ==========
if __name__ == "__main__":
    process_all_pdfs(
        directory=PDF_DIRECTORY,
        api_key=OPENAI_API_KEY,
        extract_keywords_flag=True,
        classify_flag=True
    )