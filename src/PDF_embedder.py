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
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple
import time

# 환경설정
load_dotenv()
PDF_DIRECTORY = os.getenv("DIR")
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
EMBEDDING_DIMENSIONS = 3072  # text-embedding-3-large 차원

# 고급 임베딩 설정
CONTEXTUAL_MODEL = "gpt-4o-mini"  # 컨텍스트 생성용 모델
HYDE_MODEL = "gpt-4o-mini"  # 가상 질문 생성용 모델
MAX_CONCURRENT_REQUESTS = 10  # 동시 API 요청 수

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)

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


# ========== 마크다운 변환 유틸리티 ==========
def element_to_markdown(element):
    """Unstructured element를 마크다운 형식으로 변환"""
    if not hasattr(element, 'text') or not element.text:
        return ""

    text = element.text.strip()
    if not text:
        return ""

    category = getattr(element, 'category', 'NarrativeText')

    # 카테고리별 마크다운 변환
    if category == "Title":
        # 제목 레벨 추정 (텍스트 길이, 폰트 크기 등 고려)
        return f"## {text}\n\n"

    elif category == "Header":
        return f"### {text}\n\n"

    elif category == "Table":
        # 테이블은 별도 처리 (table_to_markdown 사용)
        return text + "\n\n"

    elif category == "ListItem":
        # 번호 또는 불릿 리스트 처리
        if re.match(r'^\d+[\.\)]\s*', text):
            return f"{text}\n"
        elif re.match(r'^[가-힣][\.\)]\s*', text):
            return f"{text}\n"
        else:
            return f"- {text}\n"

    elif category == "FigureCaption":
        return f"*{text}*\n\n"

    elif category == "Footer" or category == "PageNumber":
        # 페이지 번호, 푸터는 스킵
        return ""

    else:
        # NarrativeText 등 일반 텍스트
        return f"{text}\n\n"


def table_to_markdown_with_gpt(table_html_or_text, context=""):
    """GPT를 활용하여 테이블을 마크다운 형식으로 변환하고 설명 추가"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """당신은 표(테이블) 데이터를 분석하고 마크다운으로 변환하는 전문가입니다.

주어진 테이블 데이터를 다음 형식으로 변환하세요:

1. **마크다운 테이블**: 깔끔한 마크다운 테이블 형식으로 변환
2. **테이블 요약**: 테이블의 핵심 내용을 2-3문장으로 요약
3. **주요 정보**: 테이블에서 추출할 수 있는 핵심 정보 (가격, 조건, 혜택 등)

JSON 형식으로만 응답:
{
    "markdown_table": "| 헤더1 | 헤더2 |\\n|---|---|\\n| 값1 | 값2 |",
    "summary": "이 테이블은 ... 를 보여줍니다.",
    "key_info": ["정보1", "정보2", "정보3"]
}"""
                },
                {
                    "role": "user",
                    "content": f"""다음 테이블을 마크다운으로 변환하고 분석해주세요.

문서 컨텍스트: {context[:500] if context else '없음'}

테이블 데이터:
{table_html_or_text}"""
                }
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        # 마크다운 형식으로 조합
        markdown_output = ""

        if result.get("markdown_table"):
            markdown_output += result["markdown_table"] + "\n\n"

        if result.get("summary"):
            markdown_output += f"**테이블 요약**: {result['summary']}\n\n"

        if result.get("key_info"):
            markdown_output += "**주요 정보**:\n"
            for info in result["key_info"]:
                markdown_output += f"- {info}\n"
            markdown_output += "\n"

        return markdown_output

    except Exception as e:
        print(f"   ⚠️ 테이블 GPT 변환 실패: {e}")
        # 실패 시 원본 텍스트 반환
        return f"```\n{table_html_or_text}\n```\n\n"


def convert_elements_to_markdown(elements, file_name=""):
    """모든 elements를 마크다운 문서로 변환"""
    markdown_parts = []
    current_context = ""  # 테이블 주변 문맥 저장

    # 파일명을 최상위 제목으로
    if file_name:
        markdown_parts.append(f"# {file_name}\n\n")

    for i, element in enumerate(elements):
        if not hasattr(element, 'text') or not element.text:
            continue

        category = getattr(element, 'category', 'NarrativeText')
        text = element.text.strip()

        if not text:
            continue

        # 테이블 처리
        if category == "Table":
            # HTML 테이블이 있으면 사용
            table_content = text
            if hasattr(element, 'metadata') and hasattr(element.metadata, 'text_as_html'):
                table_content = element.metadata.text_as_html or text

            # GPT로 테이블 변환
            table_markdown = table_to_markdown_with_gpt(table_content, current_context)
            markdown_parts.append(table_markdown)
        else:
            # 일반 요소 마크다운 변환
            md = element_to_markdown(element)
            if md:
                markdown_parts.append(md)
                # 컨텍스트 업데이트 (최근 500자)
                current_context = (current_context + " " + text)[-500:]

    return "".join(markdown_parts)


def filter_appendix_sections(markdown_text: str) -> str:
    """
    마크다운 문서에서 '부칙' 섹션을 필터링하여 마지막 부칙만 남김

    부칙이 여러 개 있는 경우:
    - 마지막 부칙과 그 내용만 유지
    - 이전 부칙들은 모두 제거

    Args:
        markdown_text: 마크다운 변환된 전체 문서

    Returns:
        필터링된 마크다운 텍스트
    """
    if not markdown_text:
        return markdown_text

    # 부칙 패턴: "부칙", "부 칙", "附則", "## 부칙", "### 부칙" 등
    # 날짜나 번호가 붙는 경우도 포함: "부칙 (2024.01.01)", "부칙 <제1호>"
    appendix_pattern = re.compile(
        r'^(#{1,6}\s*)?(부\s*칙|附\s*則)(\s*[\(<\[【].*?[\)>\]】])?(\s*$|\s+)',
        re.MULTILINE | re.IGNORECASE
    )

    # 모든 부칙 위치 찾기
    matches = list(appendix_pattern.finditer(markdown_text))

    if len(matches) <= 1:
        # 부칙이 없거나 1개만 있으면 그대로 반환
        return markdown_text

    print(f"   📋 부칙 {len(matches)}개 발견 → 마지막 부칙만 유지")

    # 마지막 부칙의 시작 위치
    last_appendix_start = matches[-1].start()

    # 첫 번째 부칙부터 마지막 부칙 직전까지의 내용을 제거
    first_appendix_start = matches[0].start()

    # 부칙 이전 내용 + 마지막 부칙 내용
    content_before_appendix = markdown_text[:first_appendix_start]
    last_appendix_content = markdown_text[last_appendix_start:]

    filtered_text = content_before_appendix + last_appendix_content

    # 제거된 부칙 수 출력
    removed_count = len(matches) - 1
    print(f"      ✂️ {removed_count}개 부칙 제거됨")

    return filtered_text


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


# ========== 8-1. 마크다운 문법 기반 청킹 (개선) ==========

class MarkdownBlock:
    """마크다운 블록을 표현하는 클래스"""
    def __init__(self, block_type, content, level=0):
        self.block_type = block_type  # heading, paragraph, table, list, code, blockquote
        self.content = content
        self.level = level  # 헤딩 레벨 (1-6) 또는 리스트 깊이

    def __len__(self):
        return len(self.content)

    def is_atomic(self):
        """분할하면 안 되는 블록인지 확인"""
        return self.block_type in ('table', 'code', 'blockquote')


def parse_markdown_blocks(markdown_text):
    """마크다운 텍스트를 블록 단위로 파싱"""
    blocks = []
    lines = markdown_text.split('\n')

    i = 0
    while i < len(lines):
        line = lines[i]

        # 빈 줄 스킵
        if not line.strip():
            i += 1
            continue

        # 1. 코드 블록 (```)
        if line.strip().startswith('```'):
            code_lines = [line]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                code_lines.append(lines[i])  # 닫는 ```
                i += 1
            blocks.append(MarkdownBlock('code', '\n'.join(code_lines)))
            continue

        # 2. 테이블 (| 로 시작)
        if line.strip().startswith('|') or (i + 1 < len(lines) and '|---' in lines[i + 1]):
            table_lines = []
            while i < len(lines) and (lines[i].strip().startswith('|') or '|---' in lines[i] or lines[i].strip().endswith('|')):
                table_lines.append(lines[i])
                i += 1
            if table_lines:
                blocks.append(MarkdownBlock('table', '\n'.join(table_lines)))
            continue

        # 3. 헤딩 (# ~ ######)
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if heading_match:
            level = len(heading_match.group(1))
            blocks.append(MarkdownBlock('heading', line, level=level))
            i += 1
            continue

        # 4. 인용 블록 (>)
        if line.strip().startswith('>'):
            quote_lines = []
            while i < len(lines) and (lines[i].strip().startswith('>') or (lines[i].strip() and quote_lines)):
                if lines[i].strip().startswith('>'):
                    quote_lines.append(lines[i])
                    i += 1
                elif lines[i].strip():  # 연속된 인용 내용
                    quote_lines.append(lines[i])
                    i += 1
                else:
                    break
            blocks.append(MarkdownBlock('blockquote', '\n'.join(quote_lines)))
            continue

        # 5. 리스트 (-, *, 숫자.)
        list_match = re.match(r'^(\s*)([-*]|\d+\.)\s+', line)
        if list_match:
            list_lines = []
            base_indent = len(list_match.group(1))
            while i < len(lines):
                current_line = lines[i]
                # 리스트 항목이거나 들여쓰기된 연속 내용
                is_list_item = re.match(r'^(\s*)([-*]|\d+\.)\s+', current_line)
                is_continuation = current_line.startswith(' ' * (base_indent + 2)) and current_line.strip()

                if is_list_item or is_continuation:
                    list_lines.append(current_line)
                    i += 1
                elif not current_line.strip():  # 빈 줄은 리스트 끝일 수 있음
                    # 다음 줄이 리스트면 계속
                    if i + 1 < len(lines) and re.match(r'^(\s*)([-*]|\d+\.)\s+', lines[i + 1]):
                        list_lines.append(current_line)
                        i += 1
                    else:
                        break
                else:
                    break

            if list_lines:
                blocks.append(MarkdownBlock('list', '\n'.join(list_lines)))
            continue

        # 6. 일반 문단
        para_lines = [line]
        i += 1
        while i < len(lines):
            next_line = lines[i]
            # 빈 줄이면 문단 끝
            if not next_line.strip():
                i += 1
                break
            # 다른 블록 시작이면 문단 끝
            if (next_line.strip().startswith('#') or
                next_line.strip().startswith('|') or
                next_line.strip().startswith('>') or
                next_line.strip().startswith('```') or
                re.match(r'^(\s*)([-*]|\d+\.)\s+', next_line)):
                break
            para_lines.append(next_line)
            i += 1

        blocks.append(MarkdownBlock('paragraph', '\n'.join(para_lines)))

    return blocks


def get_heading_context(blocks, current_index):
    """현재 위치의 상위 헤딩 컨텍스트 반환"""
    context = []
    current_level = 7  # 최대 레벨보다 높게 시작

    for i in range(current_index - 1, -1, -1):
        block = blocks[i]
        if block.block_type == 'heading' and block.level < current_level:
            context.insert(0, block.content)
            current_level = block.level
            if current_level == 1:
                break

    return context


def chunk_markdown_semantic(markdown_text, file_name, file_path):
    """마크다운 문법을 보존하면서 의미 단위로 청킹"""
    chunks = []

    if not markdown_text.strip():
        return chunks

    # 마크다운을 블록 단위로 파싱
    blocks = parse_markdown_blocks(markdown_text)

    if not blocks:
        return chunks

    print(f"   📑 파싱된 블록: {len(blocks)}개")
    block_types = {}
    for b in blocks:
        block_types[b.block_type] = block_types.get(b.block_type, 0) + 1
    print(f"      블록 타입: {block_types}")

    # 블록들을 청크로 그룹화
    current_chunk_blocks = []
    current_length = 0
    current_heading = file_name
    heading_context = []  # 상위 헤딩 컨텍스트

    for i, block in enumerate(blocks):
        block_length = len(block)

        # 헤딩 블록 처리
        if block.block_type == 'heading':
            # 이전 청크 저장 (내용이 있으면)
            if current_chunk_blocks and any(b.block_type != 'heading' for b in current_chunk_blocks):
                chunk_content = build_chunk_content(current_chunk_blocks, heading_context)
                if len(chunk_content) >= MIN_CHUNK_SIZE:
                    chunks.append(create_chunk(current_heading, chunk_content, file_name, file_path, "markdown_heading"))

            # 새 청크 시작
            current_chunk_blocks = [block]
            current_length = block_length
            current_heading = re.sub(r'^#+\s*', '', block.content).strip()
            heading_context = get_heading_context(blocks, i)
            continue

        # 원자적 블록(테이블, 코드블록)이 너무 크면 단독 청크로
        if block.is_atomic() and block_length > MAX_CHUNK_SIZE:
            # 이전 청크 저장
            if current_chunk_blocks:
                chunk_content = build_chunk_content(current_chunk_blocks, heading_context)
                if len(chunk_content) >= MIN_CHUNK_SIZE:
                    chunks.append(create_chunk(current_heading, chunk_content, file_name, file_path, "markdown_heading"))
                current_chunk_blocks = []
                current_length = 0

            # 큰 원자적 블록은 분할하지 않고 그대로 저장
            chunk_content = block.content
            if heading_context:
                chunk_content = '\n'.join(heading_context) + '\n\n' + chunk_content
            chunks.append(create_chunk(current_heading, chunk_content, file_name, file_path, f"markdown_{block.block_type}"))
            continue

        # 현재 청크에 블록 추가 시 크기 초과 체크
        if current_length + block_length > MAX_CHUNK_SIZE and current_chunk_blocks:
            # 원자적 블록은 분리해서 다음 청크로
            if block.is_atomic():
                # 현재 청크 저장
                chunk_content = build_chunk_content(current_chunk_blocks, heading_context)
                if len(chunk_content) >= MIN_CHUNK_SIZE:
                    chunks.append(create_chunk(current_heading, chunk_content, file_name, file_path, "markdown_semantic"))

                # 새 청크 시작 (원자적 블록으로)
                current_chunk_blocks = [block]
                current_length = block_length
            else:
                # 일반 블록도 현재 청크 저장 후 새 청크 시작
                chunk_content = build_chunk_content(current_chunk_blocks, heading_context)
                if len(chunk_content) >= MIN_CHUNK_SIZE:
                    chunks.append(create_chunk(current_heading, chunk_content, file_name, file_path, "markdown_semantic"))

                current_chunk_blocks = [block]
                current_length = block_length
        else:
            # 현재 청크에 추가
            current_chunk_blocks.append(block)
            current_length += block_length

    # 마지막 청크 처리
    if current_chunk_blocks:
        chunk_content = build_chunk_content(current_chunk_blocks, heading_context)
        if len(chunk_content) >= MIN_CHUNK_SIZE:
            chunks.append(create_chunk(current_heading, chunk_content, file_name, file_path, "markdown_semantic"))

    return chunks


def build_chunk_content(blocks, heading_context=None):
    """블록들을 하나의 청크 콘텐츠로 조합"""
    parts = []

    # 상위 헤딩 컨텍스트 추가 (옵션)
    if heading_context:
        for ctx in heading_context:
            parts.append(ctx)
        parts.append('')  # 빈 줄 구분

    for block in blocks:
        parts.append(block.content)

    return '\n\n'.join(parts)


def create_chunk(title, content, file_name, file_path, chunk_type):
    """청크 딕셔너리 생성"""
    # 제목에서 마크다운 기호 제거
    clean_title = re.sub(r'^#+\s*', '', title).strip()
    if len(clean_title) > 50:
        clean_title = clean_title[:50] + "..."

    return {
        "title": clean_title,
        "chapter": None,
        "content": content,
        "metadata": {
            "category": file_name,
            "source": file_path,
            "page_number": None,
            "chunk_type": chunk_type
        }
    }


def split_large_section(section, file_name, file_path):
    """큰 섹션을 마크다운 블록 단위로 분할"""
    chunks = []
    content = section["content"]
    title = section["title"]

    # 블록 단위로 파싱
    blocks = parse_markdown_blocks(content)

    current_chunk_blocks = []
    current_length = 0
    part_num = 1

    for block in blocks:
        block_length = len(block)

        # 원자적 블록이 너무 크면 단독 청크로
        if block.is_atomic() and block_length > MAX_CHUNK_SIZE:
            # 이전 청크 저장
            if current_chunk_blocks:
                chunk_content = build_chunk_content(current_chunk_blocks)
                chunks.append({
                    "title": f"{title} (Part {part_num})",
                    "chapter": None,
                    "content": chunk_content,
                    "metadata": {
                        "category": file_name,
                        "source": file_path,
                        "page_number": None,
                        "chunk_type": "markdown_semantic"
                    }
                })
                part_num += 1
                current_chunk_blocks = []
                current_length = 0

            # 큰 블록 단독 저장
            chunks.append({
                "title": f"{title} (Part {part_num})",
                "chapter": None,
                "content": block.content,
                "metadata": {
                    "category": file_name,
                    "source": file_path,
                    "page_number": None,
                    "chunk_type": f"markdown_{block.block_type}"
                }
            })
            part_num += 1
            continue

        if current_length + block_length > MAX_CHUNK_SIZE and current_chunk_blocks:
            chunk_content = build_chunk_content(current_chunk_blocks)
            chunks.append({
                "title": f"{title} (Part {part_num})",
                "chapter": None,
                "content": chunk_content,
                "metadata": {
                    "category": file_name,
                    "source": file_path,
                    "page_number": None,
                    "chunk_type": "markdown_semantic"
                }
            })
            part_num += 1
            current_chunk_blocks = [block]
            current_length = block_length
        else:
            current_chunk_blocks.append(block)
            current_length += block_length

    # 마지막 청크
    if current_chunk_blocks:
        chunk_content = build_chunk_content(current_chunk_blocks)
        if len(chunk_content) >= MIN_CHUNK_SIZE:
            chunks.append({
                "title": f"{title} (Part {part_num})" if part_num > 1 else title,
                "chapter": None,
                "content": chunk_content,
                "metadata": {
                    "category": file_name,
                    "source": file_path,
                    "page_number": None,
                    "chunk_type": "markdown_semantic"
                }
            })

    return chunks


def chunk_markdown_by_paragraph(markdown_text, file_name, file_path):
    """헤딩이 없는 경우 블록 기반 청킹"""
    # 블록 단위로 파싱하여 처리
    blocks = parse_markdown_blocks(markdown_text)

    if not blocks:
        return []

    chunks = []
    current_chunk_blocks = []
    current_length = 0

    for block in blocks:
        block_length = len(block)

        # 원자적 블록이 너무 크면 단독 청크로
        if block.is_atomic() and block_length > MAX_CHUNK_SIZE:
            if current_chunk_blocks:
                chunk_content = build_chunk_content(current_chunk_blocks)
                title = get_title_from_content(chunk_content)
                chunks.append(create_chunk(title, chunk_content, file_name, file_path, "markdown_paragraph"))
                current_chunk_blocks = []
                current_length = 0

            title = f"{block.block_type.capitalize()} block"
            chunks.append(create_chunk(title, block.content, file_name, file_path, f"markdown_{block.block_type}"))
            continue

        if current_length + block_length > MAX_CHUNK_SIZE and current_chunk_blocks:
            chunk_content = build_chunk_content(current_chunk_blocks)
            title = get_title_from_content(chunk_content)
            chunks.append(create_chunk(title, chunk_content, file_name, file_path, "markdown_paragraph"))

            current_chunk_blocks = [block]
            current_length = block_length
        else:
            current_chunk_blocks.append(block)
            current_length += block_length

    # 마지막 청크
    if current_chunk_blocks:
        chunk_content = build_chunk_content(current_chunk_blocks)
        if len(chunk_content) >= MIN_CHUNK_SIZE:
            title = get_title_from_content(chunk_content)
            chunks.append(create_chunk(title, chunk_content, file_name, file_path, "markdown_paragraph"))

    return chunks


def get_title_from_content(content):
    """콘텐츠에서 제목 추출"""
    # 첫 줄에서 마크다운 기호 제거
    first_line = content.split('\n')[0].strip()
    title = re.sub(r'^#+\s*', '', first_line)
    title = re.sub(r'\*+', '', title)
    title = re.sub(r'^\|.*\|$', 'Table', title)  # 테이블이면 Table로

    if len(title) > 50:
        title = title[:50] + "..."

    return title if title else "Untitled"


# ========== 9. 하이브리드 청킹 (메인) ==========
def chunk_hybrid(elements, file_name, file_path, use_markdown=True):
    """문서 구조에 따라 적합한 청킹 방식 자동 선택

    Args:
        elements: Unstructured에서 추출한 elements
        file_name: 파일명
        file_path: 파일 경로
        use_markdown: 마크다운 변환 사용 여부 (기본값: True)
    """

    if use_markdown:
        # 마크다운 기반 청킹 (권장)
        print(f"   📝 마크다운 변환 모드 사용")

        # elements를 마크다운으로 변환
        markdown_text = convert_elements_to_markdown(elements, file_name)

        if not markdown_text.strip():
            print(f"   ⚠️ 마크다운 변환 결과가 비어있음, 기존 방식으로 전환")
            return chunk_hybrid_legacy(elements, file_name, file_path)

        # 부칙 필터링: 여러 부칙 중 마지막 것만 유지
        markdown_text = filter_appendix_sections(markdown_text)

        # 마크다운 기반 시맨틱 청킹
        chunks = chunk_markdown_semantic(markdown_text, file_name, file_path)

        if not chunks:
            print(f"   ⚠️ 마크다운 청킹 실패, 기존 방식으로 전환")
            return chunk_hybrid_legacy(elements, file_name, file_path)

        print(f"   ✅ 마크다운 기반 청킹 완료: {len(chunks)}개 청크")
        return chunks

    else:
        # 기존 방식
        return chunk_hybrid_legacy(elements, file_name, file_path)


def chunk_hybrid_legacy(elements, file_name, file_path):
    """기존 하이브리드 청킹 방식 (폴백용)"""

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


# ========== 12-1. Contextual Embedding (문서 컨텍스트 추가) ==========
def generate_contextual_description(chunk_content: str, document_context: str) -> str:
    """
    청크에 대한 문서 컨텍스트 설명을 생성
    - 전체 문서 맥락에서 이 청크가 어떤 내용인지 설명
    """
    try:
        response = client.chat.completions.create(
            model=CONTEXTUAL_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """당신은 고객센터 문서 분석 전문가입니다.
주어진 청크(chunk)가 전체 문서에서 어떤 맥락과 의미를 가지는지 간결하게 설명하세요.

출력 형식:
- 2-3문장으로 간결하게 작성
- 이 청크가 다루는 핵심 주제와 문서 내 위치/역할 설명
- 고객이 이 정보를 찾을 때 사용할 만한 상황 포함"""
                },
                {
                    "role": "user",
                    "content": f"""## 문서 개요
{document_context[:1500]}

## 분석할 청크
{chunk_content[:2000]}

이 청크의 문서 내 맥락을 설명해주세요."""
                }
            ],
            temperature=0,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"   ⚠️ 컨텍스트 생성 실패: {e}")
        return ""


def generate_hypothetical_queries(chunk_content: str, num_queries: int = 3) -> List[str]:
    """
    HyDE (Hypothetical Document Embeddings) 방식
    - 이 청크를 찾을 때 사용할 만한 가상의 질문들을 생성
    """
    try:
        response = client.chat.completions.create(
            model=HYDE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"""당신은 고객센터 상담사입니다.
주어진 문서 내용을 보고, 고객이 이 정보를 찾기 위해 물어볼 수 있는 질문 {num_queries}개를 생성하세요.

규칙:
1. 실제 고객이 사용할 법한 자연스러운 질문
2. 구어체와 문어체 혼합
3. 다양한 표현 방식 사용 (직접 질문, 상황 설명, 요청 등)
4. 동의어/유사어 활용

JSON 형식으로만 응답: {{"queries": ["질문1", "질문2", "질문3"]}}"""
                },
                {
                    "role": "user",
                    "content": f"다음 내용에 대한 고객 질문을 생성하세요:\n\n{chunk_content[:1500]}"
                }
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return result.get("queries", [])
    except Exception as e:
        print(f"   ⚠️ 가상 질문 생성 실패: {e}")
        return []


def create_enhanced_embedding_text(
    chunk: Dict,
    document_context: str,
    use_contextual: bool = True,
    use_hyde: bool = True
) -> Tuple[str, Dict]:
    """
    향상된 임베딩 텍스트 생성 (Contextual + HyDE 결합)

    Returns:
        Tuple[str, Dict]: (임베딩할 텍스트, 추가 메타데이터)
    """
    title = chunk.get('title', '')
    content = chunk.get('content', '')
    keywords = chunk.get('keywords', [])
    classification = chunk.get('classification', '기타')

    # 기본 텍스트 구성
    base_text = f"""[분류: {classification}]
[키워드: {', '.join(keywords) if keywords else ''}]
{title}
{content}"""

    enhanced_parts = [base_text]
    extra_metadata = {}

    # 1. Contextual Embedding
    if use_contextual and document_context:
        contextual_desc = generate_contextual_description(content, document_context)
        if contextual_desc:
            enhanced_parts.append(f"\n[문서 컨텍스트]\n{contextual_desc}")
            extra_metadata['contextual_description'] = contextual_desc

    # 2. HyDE - 가상 질문 추가
    if use_hyde:
        hypothetical_queries = generate_hypothetical_queries(content)
        if hypothetical_queries:
            queries_text = "\n".join([f"- {q}" for q in hypothetical_queries])
            enhanced_parts.append(f"\n[예상 질문]\n{queries_text}")
            extra_metadata['hypothetical_queries'] = hypothetical_queries

    enhanced_text = "\n".join(enhanced_parts)

    return enhanced_text, extra_metadata


async def generate_contextual_async(chunk_content: str, document_context: str, semaphore: asyncio.Semaphore) -> str:
    """비동기 컨텍스트 생성"""
    async with semaphore:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            generate_contextual_description,
            chunk_content,
            document_context
        )


async def generate_hyde_async(chunk_content: str, semaphore: asyncio.Semaphore) -> List[str]:
    """비동기 HyDE 질문 생성"""
    async with semaphore:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            generate_hypothetical_queries,
            chunk_content
        )


async def process_chunks_async(
    chunks: List[Dict],
    document_context: str,
    document_keywords: List[str] = None,
    use_contextual: bool = True,
    use_hyde: bool = True
) -> List[Tuple[str, Dict]]:
    """
    청크들을 비동기로 병렬 처리하여 향상된 임베딩 텍스트 생성

    Args:
        chunks: 청크 리스트
        document_context: 문서 요약/컨텍스트
        document_keywords: 문서 전체 핵심 키워드 리스트
        use_contextual: Contextual Embedding 사용 여부
        use_hyde: HyDE 사용 여부
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    results = []

    print(f"   🚀 {len(chunks)}개 청크 병렬 처리 시작...")
    start_time = time.time()

    # 문서 키워드 문자열 준비
    doc_keywords_str = ', '.join(document_keywords) if document_keywords else ''

    async def process_single_chunk(chunk: Dict, idx: int) -> Tuple[str, Dict]:
        title = chunk.get('title', '')
        content = chunk.get('content', '')
        chunk_keywords = chunk.get('keywords', [])
        classification = chunk.get('classification', '기타')

        # 청크 키워드 + 문서 키워드 결합 (중복 제거)
        all_keywords = list(chunk_keywords) if chunk_keywords else []
        if document_keywords:
            for kw in document_keywords:
                if kw not in all_keywords:
                    all_keywords.append(kw)

        combined_keywords_str = ', '.join(all_keywords) if all_keywords else ''

        # 기본 텍스트 (문서 키워드 포함)
        base_text = f"""[분류: {classification}]
[문서 키워드: {doc_keywords_str}]
[청크 키워드: {', '.join(chunk_keywords) if chunk_keywords else ''}]
{title}
{content}"""

        enhanced_parts = [base_text]
        extra_metadata = {
            'document_keywords': document_keywords or [],
            'combined_keywords': all_keywords
        }

        # 병렬로 Contextual과 HyDE 처리
        tasks = []
        if use_contextual and document_context:
            tasks.append(('contextual', generate_contextual_async(content, document_context, semaphore)))
        if use_hyde:
            tasks.append(('hyde', generate_hyde_async(content, semaphore)))

        if tasks:
            task_results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)

            for i, (task_type, _) in enumerate(tasks):
                result = task_results[i]
                if isinstance(result, Exception):
                    print(f"      ⚠️ 청크 {idx+1} {task_type} 실패: {result}")
                    continue

                if task_type == 'contextual' and result:
                    enhanced_parts.append(f"\n[문서 컨텍스트]\n{result}")
                    extra_metadata['contextual_description'] = result
                elif task_type == 'hyde' and result:
                    queries_text = "\n".join([f"- {q}" for q in result])
                    enhanced_parts.append(f"\n[예상 질문]\n{queries_text}")
                    extra_metadata['hypothetical_queries'] = result

        return "\n".join(enhanced_parts), extra_metadata

    # 모든 청크 병렬 처리
    tasks = [process_single_chunk(chunk, i) for i, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start_time
    print(f"   ✅ 병렬 처리 완료: {elapsed:.2f}초 ({len(chunks)/elapsed:.1f} 청크/초)")

    return results


def enrich_chunks_with_embeddings(
    chunks: List[Dict],
    document_context: str = "",
    document_keywords: List[str] = None,
    use_contextual: bool = True,
    use_hyde: bool = True
) -> List[Dict]:
    """
    청크들에 향상된 임베딩 정보 추가 (동기 래퍼)

    Args:
        chunks: 청크 리스트
        document_context: 문서 요약/컨텍스트
        document_keywords: 문서 전체 핵심 키워드 리스트
        use_contextual: Contextual Embedding 사용 여부
        use_hyde: HyDE 사용 여부
    """
    try:
        # 이벤트 루프 실행
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        results = loop.run_until_complete(
            process_chunks_async(
                chunks,
                document_context,
                document_keywords=document_keywords,
                use_contextual=use_contextual,
                use_hyde=use_hyde
            )
        )

        # 결과를 청크에 반영
        for i, (enhanced_text, extra_metadata) in enumerate(results):
            chunks[i]['enhanced_text'] = enhanced_text
            chunks[i]['extra_metadata'] = extra_metadata

        return chunks

    except Exception as e:
        print(f"   ❌ 향상된 임베딩 처리 실패: {e}")
        # 실패 시 기본 텍스트 사용 (문서 키워드 포함)
        doc_keywords_str = ', '.join(document_keywords) if document_keywords else ''

        for chunk in chunks:
            title = chunk.get('title', '')
            content = chunk.get('content', '')
            keywords = chunk.get('keywords', [])
            classification = chunk.get('classification', '기타')

            chunk['enhanced_text'] = f"""[분류: {classification}]
[문서 키워드: {doc_keywords_str}]
[청크 키워드: {', '.join(keywords) if keywords else ''}]
{title}
{content}"""
            chunk['extra_metadata'] = {'document_keywords': document_keywords or []}

        return chunks
    finally:
        loop.close()


def get_document_summary(elements, file_name: str) -> str:
    """문서 전체 요약 생성 (Contextual Embedding용)"""
    try:
        # 전체 텍스트 수집 (앞부분 위주)
        full_text = "\n".join([el.text for el in elements[:30] if hasattr(el, 'text') and el.text])

        response = client.chat.completions.create(
            model=CONTEXTUAL_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """문서의 핵심 내용을 요약하세요.
- 문서의 주제와 목적
- 주요 다루는 내용들
- 대상 독자나 사용 맥락

3-5문장으로 간결하게 작성."""
                },
                {
                    "role": "user",
                    "content": f"문서명: {file_name}\n\n내용:\n{full_text[:3000]}"
                }
            ],
            temperature=0,
            max_tokens=300
        )

        summary = response.choices[0].message.content.strip()
        print(f"   📄 문서 요약 생성 완료")
        return f"[문서: {file_name}]\n{summary}"

    except Exception as e:
        print(f"   ⚠️ 문서 요약 실패: {e}")
        return f"[문서: {file_name}]"


def extract_document_keywords(elements, file_name: str, max_keywords: int = 15) -> List[str]:
    """
    PDF 문서 전체에서 핵심 키워드를 추출
    - 문서의 주요 주제, 용어, 개념을 포괄적으로 추출
    - 각 청크에 문서 키워드를 포함시켜 검색 정확도 향상
    """
    try:
        # 전체 텍스트 수집 (앞, 중간, 끝 부분 샘플링)
        all_elements = [el for el in elements if hasattr(el, 'text') and el.text]

        # 문서 전체를 고르게 샘플링
        sample_texts = []
        if len(all_elements) <= 50:
            sample_texts = [el.text for el in all_elements]
        else:
            # 앞부분 20개, 중간 15개, 끝부분 15개
            front = [el.text for el in all_elements[:20]]
            mid_start = len(all_elements) // 2 - 7
            middle = [el.text for el in all_elements[mid_start:mid_start + 15]]
            back = [el.text for el in all_elements[-15:]]
            sample_texts = front + middle + back

        full_text = "\n".join(sample_texts)

        response = client.chat.completions.create(
            model=CONTEXTUAL_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"""당신은 문서 분석 전문가입니다.
주어진 문서에서 핵심 키워드를 {max_keywords}개 추출하세요.

추출 규칙:
1. **주요 주제**: 문서가 다루는 핵심 주제/분야 (예: 요금제, 인터넷, 모바일)
2. **핵심 용어**: 문서에서 반복되는 중요 용어 (예: 해지, 가입, 변경)
3. **서비스/상품명**: 구체적인 서비스나 상품 이름
4. **고객 관련 키워드**: 고객이 검색할 만한 표현 (예: 환불, 위약금, 혜택)
5. **동의어/유사어**: 같은 의미의 다른 표현도 포함 (예: 해지/취소/끊기)

JSON 형식으로만 응답: {{"document_keywords": ["키워드1", "키워드2", ...]}}"""
                },
                {
                    "role": "user",
                    "content": f"문서명: {file_name}\n\n내용:\n{full_text[:4000]}"
                }
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        keywords = result.get("document_keywords", [])

        print(f"   🔑 문서 키워드 추출 완료: {keywords}")
        return keywords

    except Exception as e:
        print(f"   ⚠️ 문서 키워드 추출 실패: {e}")
        return []


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
def insert_chunks_to_chroma(chunks, collection, use_enhanced: bool = True):
    """청크를 ChromaDB에 저장 (향상된 임베딩 텍스트 사용)

    Args:
        chunks: 청크 리스트
        collection: ChromaDB 컬렉션
        use_enhanced: 향상된 임베딩 텍스트 사용 여부 (Contextual + HyDE)
    """
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

        # 향상된 임베딩 텍스트 사용 (Contextual + HyDE)
        if use_enhanced and 'enhanced_text' in chunk:
            doc_text = clean_text(chunk['enhanced_text'])
        else:
            # 기본 텍스트 구성
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
            content_chunks = split_text_by_tokens(doc_text, max_tokens=content_max_tokens, overlap_tokens=200)

            print(f"      → {len(content_chunks)}개로 분할됨")
            split_count += len(content_chunks) - 1  # 원래 1개에서 추가된 수

            # 각 분할 청크 저장
            for j, content_part in enumerate(content_chunks):
                # 최종 빈 체크
                if not content_part.strip():
                    continue

                documents.append(content_part)

                # 추가 메타데이터 포함
                extra_meta = chunk.get('extra_metadata', {})
                doc_keywords = extra_meta.get('document_keywords', [])
                combined_keywords = extra_meta.get('combined_keywords', [])

                metadatas.append({
                    "category": chunk["metadata"]["category"],
                    "chapter": chunk.get("chapter") or "",
                    "title": f"{title[:180]} (Part {j+1}/{len(content_chunks)})" if title else f"Part {j+1}/{len(content_chunks)}",
                    "source": chunk["metadata"]["source"],
                    "page_number": chunk["metadata"].get("page_number") or 0,
                    "chunk_type": chunk["metadata"].get("chunk_type", "unknown"),
                    "keywords": keywords_str[:500] if keywords_str else "",
                    "document_keywords": ', '.join(doc_keywords)[:500] if doc_keywords else "",
                    "combined_keywords": ', '.join(combined_keywords)[:500] if combined_keywords else "",
                    "classification": classification,
                    "classification_confidence": chunk.get("classification_confidence", 0.0),
                    "is_split": True,
                    "split_part": j + 1,
                    "split_total": len(content_chunks),
                    "has_contextual": bool(extra_meta.get('contextual_description')),
                    "has_hyde": bool(extra_meta.get('hypothetical_queries')),
                    "hypothetical_queries": json.dumps(extra_meta.get('hypothetical_queries', []), ensure_ascii=False)[:500]
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

            # 추가 메타데이터 포함
            extra_meta = chunk.get('extra_metadata', {})
            doc_keywords = extra_meta.get('document_keywords', [])
            combined_keywords = extra_meta.get('combined_keywords', [])

            metadatas.append({
                "category": chunk["metadata"]["category"],
                "chapter": chunk.get("chapter") or "",
                "title": title[:200] if title else "",
                "source": chunk["metadata"]["source"],
                "page_number": chunk["metadata"].get("page_number") or 0,
                "chunk_type": chunk["metadata"].get("chunk_type", "unknown"),
                "keywords": keywords_str[:500] if keywords_str else "",
                "document_keywords": ', '.join(doc_keywords)[:500] if doc_keywords else "",
                "combined_keywords": ', '.join(combined_keywords)[:500] if combined_keywords else "",
                "classification": classification,
                "classification_confidence": chunk.get("classification_confidence", 0.0),
                "is_split": False,
                "split_part": 0,
                "split_total": 1,
                "has_contextual": bool(extra_meta.get('contextual_description')),
                "has_hyde": bool(extra_meta.get('hypothetical_queries')),
                "hypothetical_queries": json.dumps(extra_meta.get('hypothetical_queries', []), ensure_ascii=False)[:500]
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
    if use_enhanced:
        contextual_count = sum(1 for c in chunks if c.get('extra_metadata', {}).get('contextual_description'))
        hyde_count = sum(1 for c in chunks if c.get('extra_metadata', {}).get('hypothetical_queries'))
        print(f"      (Contextual: {contextual_count}개, HyDE: {hyde_count}개 적용)")


# ========== 15. 메인 실행 ==========
def process_all_pdfs(
    directory,
    api_key,
    extract_keywords_flag=True,
    classify_flag=True,
    use_markdown=True,
    use_contextual=True, #문맥 임베딩
    use_hyde=True        #HyDE 
):
    """디렉터리 내 모든 PDF 처리

    Args:
        directory: PDF 파일이 있는 디렉터리 경로
        api_key: OpenAI API 키
        extract_keywords_flag: 키워드 추출 여부 (기본값: True)
        classify_flag: 분류 수행 여부 (기본값: True)
        use_markdown: 마크다운 변환 사용 여부 (기본값: True, 권장)
        use_contextual: Contextual Embedding 사용 여부 (기본값: True)
        use_hyde: HyDE(가상 질문 생성) 사용 여부 (기본값: True)
    """

    pdf_files = get_pdf_files(directory)

    if not pdf_files:
        print("❌ PDF 파일이 없습니다.")
        return

    collection = setup_chromadb(api_key)

    total_chunks = 0
    success_files = 0
    failed_files = []
    table_count = 0  # 처리된 테이블 수

    print(f"\n{'='*50}")
    print(f"📋 처리 설정:")
    print(f"   - 마크다운 변환: {'사용' if use_markdown else '미사용'}")
    print(f"   - 키워드 추출: {'사용' if extract_keywords_flag else '미사용'}")
    print(f"   - 자동 분류: {'사용' if classify_flag else '미사용'}")
    print(f"   - Contextual Embedding: {'사용' if use_contextual else '미사용'}")
    print(f"   - HyDE (가상 질문): {'사용' if use_hyde else '미사용'}")
    print(f"{'='*50}\n")

    for file_path in pdf_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        try:
            # 추출
            elements = extract_elements(file_path)

            if not elements:
                print(f"   ⚠️ 추출된 요소 없음")
                failed_files.append(file_path)
                continue

            # 문서 요약 생성 (Contextual Embedding용)
            document_context = ""
            if use_contextual:
                document_context = get_document_summary(elements, file_name)

            # 문서 전체 핵심 키워드 추출
            document_keywords = []
            if use_contextual or use_hyde:
                document_keywords = extract_document_keywords(elements, file_name)

            # 테이블 수 카운트
            file_table_count = sum(1 for el in elements if getattr(el, 'category', '') == 'Table')
            if file_table_count > 0:
                print(f"   📊 테이블 {file_table_count}개 발견 (GPT 변환 예정)")
            table_count += file_table_count

            # 하이브리드 청킹 (마크다운 옵션 전달)
            chunks = chunk_hybrid(elements, file_name, file_path, use_markdown=use_markdown)
            print(f"   📝 {len(chunks)}개 청크 생성")

            if not chunks:
                failed_files.append(file_path)
                continue

            # 키워드 & 분류 추가
            chunks = enrich_chunks(chunks, extract_keywords_flag, classify_flag)

            # 향상된 임베딩 처리 (Contextual + HyDE + 문서 키워드)
            if use_contextual or use_hyde:
                print(f"   🔄 향상된 임베딩 텍스트 생성 중...")
                chunks = enrich_chunks_with_embeddings(
                    chunks,
                    document_context=document_context,
                    document_keywords=document_keywords,
                    use_contextual=use_contextual,
                    use_hyde=use_hyde
                )

            # 저장
            use_enhanced = use_contextual or use_hyde
            insert_chunks_to_chroma(chunks, collection, use_enhanced=use_enhanced)
            total_chunks += len(chunks)
            success_files += 1

        except Exception as e:
            print(f"   ❌ 처리 실패: {e}")
            import traceback
            traceback.print_exc()
            failed_files.append(file_path)

    print(f"\n{'='*50}")
    print(f"🎉 전체 처리 완료!")
    print(f"📄 성공한 PDF: {success_files}개")
    print(f"❌ 실패한 PDF: {len(failed_files)}개")
    if failed_files:
        for f in failed_files:
            print(f"   - {f}")
    print(f"📊 총 청크 수: {total_chunks}개")
    print(f"📊 처리된 테이블 수: {table_count}개")
    print(f"💾 저장된 문서 수: {collection.count()}개")


# ========== 실행 ==========
if __name__ == "__main__":
    process_all_pdfs(
        directory=PDF_DIRECTORY,
        api_key=OPENAI_API_KEY,
        extract_keywords_flag=True,
        classify_flag=True,
        use_markdown=True,       # 마크다운 변환 사용 (테이블 GPT 변환 포함)
        use_contextual=True,     # Contextual Embedding (문서 맥락 추가)
        use_hyde=True            # HyDE (가상 질문 생성)
    )