"""
JSON 파일 임베딩 파이프라인
- PDF_embedder.py와 동일한 향상된 임베딩 기능 사용
- JSON 구조를 마크다운으로 변환 후 청킹
- Contextual Embedding + HyDE 지원
- 다양한 JSON 구조 지원 (요금제, 멤버십, 인터넷 상품, 결합할인 등)
"""

import re
import os
import json
import asyncio
import time
from glob import glob
from typing import List, Dict, Optional, Tuple, Any
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import tiktoken
from bs4 import BeautifulSoup

# 환경설정
load_dotenv()
JSON_DIRECTORY = os.getenv("DIR_SIMPLE")
OPENAI_API_KEY = os.getenv("API_KEY")
CLASSIFICATION_CATEGORIES = os.getenv("CATEGORIES")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH_SIMPLE")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# 청킹 설정
MAX_CHUNK_SIZE = 1500
MIN_CHUNK_SIZE = 30
CHUNK_OVERLAP = 100

# 임베딩 설정
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_EMBEDDING_TOKENS = 8000
EMBEDDING_DIMENSIONS = 3072

# 고급 임베딩 설정
CONTEXTUAL_MODEL = "gpt-4o-mini"
HYDE_MODEL = "gpt-4o-mini"
MAX_CONCURRENT_REQUESTS = 10

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)


# ========== 토큰 관련 유틸리티 ==========
def count_tokens(text, model=EMBEDDING_MODEL):
    """텍스트의 토큰 수 계산"""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
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
        max_chars = max_tokens * 2
        return text[:max_chars]


def split_text_by_tokens(text, max_tokens=8000, overlap_tokens=200):
    """토큰 제한에 맞게 텍스트를 여러 청크로 분할"""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)

        if len(tokens) <= max_tokens:
            return [text]

        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            start = end - overlap_tokens if end < len(tokens) else end

        return chunks

    except Exception as e:
        print(f"   [WARN] Token split failed: {e}")
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
    """텍스트 정제"""
    if not text:
        return ""
    if text is None:
        return ""
    text = str(text).strip()
    if not text:
        return ""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    return text


# ========== JSON 파일 처리 ==========
def get_json_files(directory: str) -> List[str]:
    """디렉터리 내 모든 JSON 파일 경로 반환"""
    json_pattern = os.path.join(directory, "**", "*.json")
    json_files = glob(json_pattern, recursive=True)

    print(f"[FOLDER] {directory}")
    print(f"[FILES] JSON files found: {len(json_files)}")
    for f in json_files:
        print(f"   - {f}")

    return json_files


def load_json_file(file_path: str) -> Optional[List[Dict]]:
    """JSON 파일 로드"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 리스트가 아니면 리스트로 감싸기
        if isinstance(data, dict):
            data = [data]

        print(f"\n[LOAD] {file_path}")
        print(f"   [ITEMS] {len(data)}")

        return data
    except Exception as e:
        print(f"   [WARN] JSON load failed: {e}")
        return None


# ========== HTML 테이블 파싱 ==========
def parse_html_table(html_content: str) -> str:
    """HTML 테이블을 마크다운 형식으로 변환"""
    if not html_content:
        return ""

    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        table = soup.find('table')

        if not table:
            # 테이블이 없으면 텍스트만 추출
            text = soup.get_text(separator=' ', strip=True)
            return text if text else ""

        rows = table.find_all('tr')
        if not rows:
            return ""

        md_lines = []
        max_cols = 0

        # 최대 열 수 계산 (colspan 고려)
        for row in rows:
            cells = row.find_all(['td', 'th'])
            col_count = sum(int(cell.get('colspan', 1)) for cell in cells)
            max_cols = max(max_cols, col_count)

        for row_idx, row in enumerate(rows):
            cells = row.find_all(['td', 'th'])
            row_values = []

            for cell in cells:
                # 셀 텍스트 추출 (HTML 태그 제거)
                cell_text = cell.get_text(separator=' ', strip=True)
                # 줄바꿈을 공백으로 대체
                cell_text = re.sub(r'\s+', ' ', cell_text)
                colspan = int(cell.get('colspan', 1))
                row_values.extend([cell_text] + [''] * (colspan - 1))

            # 열 수 맞추기
            while len(row_values) < max_cols:
                row_values.append('')

            md_lines.append("| " + " | ".join(row_values[:max_cols]) + " |")

            # 첫 번째 행 다음에 구분선 추가
            if row_idx == 0:
                md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")

        return "\n".join(md_lines)

    except Exception as e:
        print(f"   [WARN] HTML table parsing failed: {e}")
        # 실패 시 텍스트만 추출
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            return soup.get_text(separator=' ', strip=True)
        except:
            return ""


def detect_json_type(item: Dict) -> str:
    """JSON 구조 유형 감지"""
    # 모바일 요금제 (나무위키 크롤링)
    if 'category' in item and 'sub_categories' in item:
        return 'mobile_plans'

    # 멤버십 정보
    if 'section' in item and 'content' in item:
        return 'membership'

    # 인터넷 상품
    if 'provider' in item and 'products' in item:
        return 'internet_product'

    # 결합할인
    if 'bundles' in item or ('provider' in item and 'category' in item and 'bundles' not in item and 'products' not in item):
        return 'bundle_discount'

    # 일반 항목
    return 'generic'


# ========== JSON을 마크다운으로 변환 ==========
def json_to_markdown(item: Dict, item_index: int = 0) -> str:
    """
    JSON 항목을 마크다운 형식으로 변환
    다양한 JSON 구조를 자동 감지하여 처리
    """
    json_type = detect_json_type(item)

    if json_type == 'mobile_plans':
        return mobile_plans_to_markdown(item)
    elif json_type == 'membership':
        return membership_to_markdown(item)
    elif json_type == 'internet_product':
        return internet_product_to_markdown(item)
    elif json_type == 'bundle_discount':
        return bundle_discount_to_markdown(item)
    else:
        return generic_json_to_markdown(item, item_index)


def mobile_plans_to_markdown(item: Dict) -> str:
    """모바일 요금제 JSON을 마크다운으로 변환"""
    markdown_parts = []

    category = item.get('category', '요금제')
    markdown_parts.append(f"# {category}\n")

    sub_categories = item.get('sub_categories', [])
    for sub_cat in sub_categories:
        sub_name = sub_cat.get('name', '')
        if sub_name:
            markdown_parts.append(f"\n## {sub_name}\n")

        # plans 처리
        plans = sub_cat.get('plans', [])
        for plan in plans:
            plan_name = plan.get('plan_name', '')
            if plan_name:
                markdown_parts.append(f"\n### {plan_name}\n")

            # tables 처리
            tables = plan.get('tables', [])
            for table in tables:
                table_name = table.get('table_name', '')
                if table_name:
                    markdown_parts.append(f"\n**{table_name}**\n")

                # HTML 테이블 파싱
                table_html = table.get('table_html', '')
                if table_html:
                    parsed_table = parse_html_table(table_html)
                    if parsed_table:
                        markdown_parts.append(f"\n{parsed_table}\n")

                # description 추가
                description = table.get('description', '')
                if description:
                    markdown_parts.append(f"\n{description}\n")

        # direct_tables 처리
        direct_tables = sub_cat.get('direct_tables', [])
        for table in direct_tables:
            table_name = table.get('table_name', '')
            if table_name:
                markdown_parts.append(f"\n**{table_name}**\n")

            table_html = table.get('table_html', '')
            if table_html:
                parsed_table = parse_html_table(table_html)
                if parsed_table:
                    markdown_parts.append(f"\n{parsed_table}\n")

    return "\n".join(markdown_parts)


def membership_to_markdown(item: Dict) -> str:
    """멤버십 정보 JSON을 마크다운으로 변환"""
    markdown_parts = []

    section = item.get('section', '')
    section_number = item.get('section_number', '')

    # 섹션 제목
    if section:
        markdown_parts.append(f"# {section}\n")
    if section_number:
        markdown_parts.append(f"*섹션 번호: {section_number}*\n")

    # 콘텐츠 처리
    content = item.get('content', {})
    if content:
        # 텍스트
        texts = content.get('text', [])
        for text in texts:
            if text:
                markdown_parts.append(f"\n{text}\n")

        # 리스트 아이템
        list_items = content.get('list_items', [])
        if list_items:
            markdown_parts.append("\n")
            for li in list_items:
                if li:
                    markdown_parts.append(f"- {li}\n")

        # 테이블
        tables = content.get('tables', [])
        for table in tables:
            table_name = table.get('table_name', '')
            if table_name:
                markdown_parts.append(f"\n**{table_name}**\n")

            table_html = table.get('table_html', '')
            if table_html:
                parsed_table = parse_html_table(table_html)
                if parsed_table:
                    markdown_parts.append(f"\n{parsed_table}\n")

    # 하위 섹션 처리
    sub_sections = item.get('sub_sections', [])
    for sub_sec in sub_sections:
        sub_name = sub_sec.get('name', '')
        sub_number = sub_sec.get('section_number', '')

        if sub_name:
            markdown_parts.append(f"\n## {sub_name}\n")
        if sub_number:
            markdown_parts.append(f"*섹션 번호: {sub_number}*\n")

        sub_content = sub_sec.get('content', {})
        if sub_content:
            texts = sub_content.get('text', [])
            for text in texts:
                if text:
                    markdown_parts.append(f"\n{text}\n")

            list_items = sub_content.get('list_items', [])
            if list_items:
                markdown_parts.append("\n")
                for li in list_items:
                    if li:
                        markdown_parts.append(f"- {li}\n")

    return "\n".join(markdown_parts)


def internet_product_to_markdown(item: Dict) -> str:
    """인터넷 상품 JSON을 마크다운으로 변환"""
    markdown_parts = []

    provider = item.get('provider', '')
    category = item.get('category', '')

    markdown_parts.append(f"# {provider} {category}\n")

    products = item.get('products', [])
    for product in products:
        name = product.get('name', '')
        if name:
            markdown_parts.append(f"\n## {name}\n")

        # 속도 정보
        speed = product.get('speed', {})
        if speed:
            download = speed.get('download', '')
            upload = speed.get('upload', '')
            speed_type = speed.get('type', '')
            markdown_parts.append(f"- **다운로드 속도**: {download}\n")
            markdown_parts.append(f"- **업로드 속도**: {upload}\n")
            if speed_type:
                markdown_parts.append(f"- **속도 유형**: {speed_type}\n")

        # 기타 정보
        technology = product.get('technology', '')
        if technology:
            markdown_parts.append(f"- **기술**: {technology}\n")

        availability = product.get('availability', '')
        if availability:
            markdown_parts.append(f"- **가용성**: {availability}\n")

        qos_limit = product.get('qos_daily_limit_gb')
        if qos_limit:
            markdown_parts.append(f"- **일일 QoS 제한**: {qos_limit}GB\n")

        notes = product.get('notes', '')
        if notes:
            markdown_parts.append(f"- **비고**: {notes}\n")

    return "\n".join(markdown_parts)


def bundle_discount_to_markdown(item: Dict) -> str:
    """결합할인 JSON을 마크다운으로 변환"""
    markdown_parts = []

    provider = item.get('provider', '')
    category = item.get('category', '')

    markdown_parts.append(f"# {provider} {category}\n")

    sources = item.get('sources', [])
    if sources:
        markdown_parts.append(f"*출처: {', '.join(sources)}*\n")

    last_updated = item.get('last_updated', '')
    if last_updated:
        markdown_parts.append(f"*최종 업데이트: {last_updated}*\n")

    bundles = item.get('bundles', [])
    for bundle in bundles:
        name = bundle.get('name', '')
        if name:
            markdown_parts.append(f"\n## {name}\n")

        # 조건
        conditions = bundle.get('conditions', {})
        if conditions:
            markdown_parts.append("\n### 가입 조건\n")
            for key, value in conditions.items():
                if value is not None:
                    # key를 한글로 변환
                    key_kr = {
                        'phone_plan_min_price': '최소 휴대폰 요금제',
                        'phone_lines_min': '최소 휴대폰 회선 수',
                        'phone_lines_max': '최대 휴대폰 회선 수',
                        'phone_lines': '휴대폰 회선 수',
                        'internet_required': '인터넷 필수 여부',
                        'internet_speed_min': '최소 인터넷 속도',
                        'discount_basis': '할인 기준',
                        'contract_years': '약정 기간(년)',
                        'phone_plan_note': '요금제 참고'
                    }.get(key, key)
                    markdown_parts.append(f"- **{key_kr}**: {value}\n")

        # 할인 정보
        discounts = bundle.get('discounts', {})
        if discounts:
            markdown_parts.append("\n### 할인 혜택\n")
            for discount_type, discount_info in discounts.items():
                discount_type_kr = {'internet': '인터넷', 'phone': '휴대폰'}.get(discount_type, discount_type)

                if isinstance(discount_info, dict):
                    amount = discount_info.get('amount') or discount_info.get('amount_max')
                    rate = discount_info.get('rate')
                    unit = discount_info.get('unit', '')
                    note = discount_info.get('note', '')

                    if amount:
                        markdown_parts.append(f"- **{discount_type_kr}**: {amount}{unit} 할인\n")
                    elif rate:
                        markdown_parts.append(f"- **{discount_type_kr}**: {rate}{unit} 할인\n")

                    if note:
                        markdown_parts.append(f"  - {note}\n")

        # 추천 대상
        recommended = bundle.get('recommended_for', '')
        if recommended:
            markdown_parts.append(f"\n**추천 대상**: {recommended}\n")

        # 할인 테이블
        discount_table = bundle.get('discount_table', {})
        if discount_table:
            markdown_parts.append("\n### 세부 할인 테이블\n")
            for table_name, table_rows in discount_table.items():
                markdown_parts.append(f"\n**{table_name}**\n")
                if isinstance(table_rows, list) and table_rows:
                    # 테이블 헤더
                    headers = list(table_rows[0].keys())
                    headers_kr = {
                        'phone_total_min': '최소 요금',
                        'phone_total_max': '최대 요금',
                        'total_discount': '총 할인',
                        'internet_discount': '인터넷 할인',
                        'mobile_discount': '모바일 할인'
                    }
                    headers_display = [headers_kr.get(h, h) for h in headers]
                    markdown_parts.append("| " + " | ".join(headers_display) + " |\n")
                    markdown_parts.append("| " + " | ".join(["---"] * len(headers)) + " |\n")

                    for row in table_rows[:10]:  # 최대 10개 행만 표시
                        values = [str(row.get(h, '')) for h in headers]
                        markdown_parts.append("| " + " | ".join(values) + " |\n")

    return "\n".join(markdown_parts)


def generic_json_to_markdown(item: Dict, item_index: int = 0) -> str:
    """
    일반 JSON 항목을 마크다운 형식으로 변환

    지원하는 JSON 구조:
    1. KT 요금제 형식: name, sections, tables 등
    2. Q&A 형식: question, answer
    3. 일반 key-value 형식
    """
    markdown_parts = []

    # 1. 제목 (name 또는 title 필드)
    title = item.get('name') or item.get('title') or item.get('제목') or f"항목 {item_index + 1}"
    markdown_parts.append(f"# {title}\n")

    # 2. 기본 메타 정보
    meta_fields = ['product_type', 'filter_name', 'url', 'category', '분류', '카테고리']
    meta_info = []
    for field in meta_fields:
        if field in item and item[field]:
            meta_info.append(f"- **{field}**: {item[field]}")

    if meta_info:
        markdown_parts.append("\n".join(meta_info) + "\n")

    # 3. sections 처리 (KT 요금제 형식)
    if 'sections' in item and isinstance(item['sections'], dict):
        for section_name, section_content in item['sections'].items():
            markdown_parts.append(f"\n## {section_name}\n")

            if isinstance(section_content, dict):
                # tables 처리
                if 'tables' in section_content:
                    for table in section_content['tables']:
                        markdown_parts.append(table_to_markdown(table))

                # texts 처리
                if 'texts' in section_content:
                    for text in section_content['texts']:
                        if isinstance(text, str):
                            markdown_parts.append(f"{text}\n")
                        elif isinstance(text, dict):
                            for k, v in text.items():
                                markdown_parts.append(f"**{k}**: {v}\n")

                # 기타 필드
                for key, value in section_content.items():
                    if key not in ['tables', 'texts']:
                        markdown_parts.append(f"**{key}**: {value}\n")

            elif isinstance(section_content, list):
                for item_content in section_content:
                    if isinstance(item_content, str):
                        markdown_parts.append(f"- {item_content}\n")
                    elif isinstance(item_content, dict):
                        markdown_parts.append(dict_to_markdown(item_content))

            elif isinstance(section_content, str):
                markdown_parts.append(f"{section_content}\n")

    # 4. Q&A 형식 처리
    if 'question' in item or 'answer' in item or '질문' in item or '답변' in item:
        question = item.get('question') or item.get('질문') or ''
        answer = item.get('answer') or item.get('답변') or ''

        if question:
            markdown_parts.append(f"\n## 질문\n{question}\n")
        if answer:
            markdown_parts.append(f"\n## 답변\n{answer}\n")

    # 5. 일반 key-value 처리 (아직 처리되지 않은 필드들)
    processed_keys = {'name', 'title', '제목', 'sections', 'question', 'answer',
                      '질문', '답변', 'product_type', 'filter_name', 'url',
                      'category', '분류', '카테고리', 'filter_code', 'html_type'}

    remaining_fields = {k: v for k, v in item.items() if k not in processed_keys}
    if remaining_fields:
        markdown_parts.append("\n## 추가 정보\n")
        markdown_parts.append(dict_to_markdown(remaining_fields))

    return "\n".join(markdown_parts)


def table_to_markdown(table: List[Dict]) -> str:
    """테이블 데이터를 마크다운 테이블로 변환"""
    if not table:
        return ""

    # 헤더 추출
    if isinstance(table[0], dict):
        headers = list(table[0].keys())
    else:
        return str(table)

    # 마크다운 테이블 생성
    md_lines = []
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in table:
        if isinstance(row, dict):
            values = [str(row.get(h, '')) for h in headers]
            md_lines.append("| " + " | ".join(values) + " |")

    return "\n".join(md_lines) + "\n"


def dict_to_markdown(d: Dict, indent: int = 0) -> str:
    """딕셔너리를 마크다운으로 변환"""
    md_parts = []
    prefix = "  " * indent

    for key, value in d.items():
        if isinstance(value, dict):
            md_parts.append(f"{prefix}**{key}**:")
            md_parts.append(dict_to_markdown(value, indent + 1))
        elif isinstance(value, list):
            md_parts.append(f"{prefix}**{key}**:")
            for item in value:
                if isinstance(item, dict):
                    md_parts.append(dict_to_markdown(item, indent + 1))
                else:
                    md_parts.append(f"{prefix}  - {item}")
        else:
            md_parts.append(f"{prefix}- **{key}**: {value}")

    return "\n".join(md_parts) + "\n"


# ========== 마크다운 청킹 (PDF_embedder에서 재사용) ==========
class MarkdownBlock:
    """마크다운 블록을 표현하는 클래스"""
    def __init__(self, block_type, content, level=0):
        self.block_type = block_type
        self.content = content
        self.level = level

    def __len__(self):
        return len(self.content)

    def is_atomic(self):
        return self.block_type in ('table', 'code', 'blockquote')


def parse_markdown_blocks(markdown_text: str) -> List[MarkdownBlock]:
    """마크다운 텍스트를 블록 단위로 파싱"""
    blocks = []
    lines = markdown_text.split('\n')

    i = 0
    while i < len(lines):
        line = lines[i]

        if not line.strip():
            i += 1
            continue

        # 코드 블록
        if line.strip().startswith('```'):
            code_lines = [line]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                code_lines.append(lines[i])
                i += 1
            blocks.append(MarkdownBlock('code', '\n'.join(code_lines)))
            continue

        # 테이블
        if line.strip().startswith('|') or (i + 1 < len(lines) and '|---' in lines[i + 1]):
            table_lines = []
            while i < len(lines) and (lines[i].strip().startswith('|') or '|---' in lines[i] or lines[i].strip().endswith('|')):
                table_lines.append(lines[i])
                i += 1
            if table_lines:
                blocks.append(MarkdownBlock('table', '\n'.join(table_lines)))
            continue

        # 헤딩
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if heading_match:
            level = len(heading_match.group(1))
            blocks.append(MarkdownBlock('heading', line, level=level))
            i += 1
            continue

        # 인용 블록
        if line.strip().startswith('>'):
            quote_lines = []
            while i < len(lines) and (lines[i].strip().startswith('>') or (lines[i].strip() and quote_lines)):
                if lines[i].strip().startswith('>'):
                    quote_lines.append(lines[i])
                    i += 1
                elif lines[i].strip():
                    quote_lines.append(lines[i])
                    i += 1
                else:
                    break
            blocks.append(MarkdownBlock('blockquote', '\n'.join(quote_lines)))
            continue

        # 리스트
        list_match = re.match(r'^(\s*)([-*]|\d+\.)\s+', line)
        if list_match:
            list_lines = []
            base_indent = len(list_match.group(1))
            while i < len(lines):
                current_line = lines[i]
                is_list_item = re.match(r'^(\s*)([-*]|\d+\.)\s+', current_line)
                is_continuation = current_line.startswith(' ' * (base_indent + 2)) and current_line.strip()

                if is_list_item or is_continuation:
                    list_lines.append(current_line)
                    i += 1
                elif not current_line.strip():
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

        # 일반 문단
        para_lines = [line]
        i += 1
        while i < len(lines):
            next_line = lines[i]
            if not next_line.strip():
                i += 1
                break
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


def chunk_markdown_semantic(markdown_text: str, file_name: str, file_path: str, item_name: str = "") -> List[Dict]:
    """마크다운 문법을 보존하면서 의미 단위로 청킹"""
    chunks = []

    if not markdown_text.strip():
        return chunks

    blocks = parse_markdown_blocks(markdown_text)

    if not blocks:
        return chunks

    current_chunk_blocks = []
    current_length = 0
    current_heading = item_name or file_name
    heading_context = []

    for i, block in enumerate(blocks):
        block_length = len(block)

        # 헤딩 블록 처리
        if block.block_type == 'heading':
            if current_chunk_blocks and any(b.block_type != 'heading' for b in current_chunk_blocks):
                chunk_content = build_chunk_content(current_chunk_blocks, heading_context)
                if len(chunk_content) >= MIN_CHUNK_SIZE:
                    chunks.append(create_chunk(current_heading, chunk_content, file_name, file_path, "json_markdown"))

            current_chunk_blocks = [block]
            current_length = block_length
            current_heading = re.sub(r'^#+\s*', '', block.content).strip()
            heading_context = get_heading_context(blocks, i)
            continue

        # 원자적 블록이 너무 크면 단독 청크로
        if block.is_atomic() and block_length > MAX_CHUNK_SIZE:
            if current_chunk_blocks:
                chunk_content = build_chunk_content(current_chunk_blocks, heading_context)
                if len(chunk_content) >= MIN_CHUNK_SIZE:
                    chunks.append(create_chunk(current_heading, chunk_content, file_name, file_path, "json_markdown"))
                current_chunk_blocks = []
                current_length = 0

            chunk_content = block.content
            if heading_context:
                chunk_content = '\n'.join(heading_context) + '\n\n' + chunk_content
            chunks.append(create_chunk(current_heading, chunk_content, file_name, file_path, f"json_{block.block_type}"))
            continue

        # 크기 초과 체크
        if current_length + block_length > MAX_CHUNK_SIZE and current_chunk_blocks:
            chunk_content = build_chunk_content(current_chunk_blocks, heading_context)
            if len(chunk_content) >= MIN_CHUNK_SIZE:
                chunks.append(create_chunk(current_heading, chunk_content, file_name, file_path, "json_markdown"))

            current_chunk_blocks = [block]
            current_length = block_length
        else:
            current_chunk_blocks.append(block)
            current_length += block_length

    # 마지막 청크 처리
    if current_chunk_blocks:
        chunk_content = build_chunk_content(current_chunk_blocks, heading_context)
        if len(chunk_content) >= MIN_CHUNK_SIZE:
            chunks.append(create_chunk(current_heading, chunk_content, file_name, file_path, "json_markdown"))

    return chunks


def get_heading_context(blocks: List[MarkdownBlock], current_index: int) -> List[str]:
    """현재 위치의 상위 헤딩 컨텍스트 반환"""
    context = []
    current_level = 7

    for i in range(current_index - 1, -1, -1):
        block = blocks[i]
        if block.block_type == 'heading' and block.level < current_level:
            context.insert(0, block.content)
            current_level = block.level
            if current_level == 1:
                break

    return context


def build_chunk_content(blocks: List[MarkdownBlock], heading_context: List[str] = None) -> str:
    """블록들을 하나의 청크 콘텐츠로 조합"""
    parts = []

    if heading_context:
        for ctx in heading_context:
            parts.append(ctx)
        parts.append('')

    for block in blocks:
        parts.append(block.content)

    return '\n\n'.join(parts)


def create_chunk(title: str, content: str, file_name: str, file_path: str, chunk_type: str) -> Dict:
    """청크 딕셔너리 생성"""
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


# ========== 키워드 추출 및 분류 ==========
def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
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
- 동의어, 유사어도 포함
- 구어체 표현도 포함

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
        print(f"   [WARN] Keyword extraction failed: {e}")
        return []


def classify_content(text: str, categories: str = CLASSIFICATION_CATEGORIES) -> Tuple[str, float]:
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
        print(f"   [WARN] Classification failed: {e}")
        return "기타", 0.0


def extract_document_keywords(json_data: List[Dict], file_name: str, max_keywords: int = 15) -> List[str]:
    """JSON 문서 전체에서 핵심 키워드를 추출"""
    try:
        # 전체 텍스트 수집
        sample_texts = []
        for item in json_data[:10]:  # 처음 10개 항목 샘플링
            sample_texts.append(json.dumps(item, ensure_ascii=False)[:500])

        full_text = "\n".join(sample_texts)

        response = client.chat.completions.create(
            model=CONTEXTUAL_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"""당신은 문서 분석 전문가입니다.
주어진 문서에서 핵심 키워드를 {max_keywords}개 추출하세요.

추출 규칙:
1. 주요 주제: 문서가 다루는 핵심 주제/분야
2. 핵심 용어: 문서에서 반복되는 중요 용어
3. 서비스/상품명: 구체적인 서비스나 상품 이름
4. 고객 관련 키워드: 고객이 검색할 만한 표현
5. 동의어/유사어: 같은 의미의 다른 표현

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

        print(f"   [KEYWORDS] Document keywords: {keywords}")
        return keywords

    except Exception as e:
        print(f"   [WARN] Document keyword extraction failed: {e}")
        return []


def enrich_chunks(chunks: List[Dict], extract_keywords_flag: bool = True, classify_flag: bool = True) -> List[Dict]:
    """청크에 키워드와 분류 추가"""
    for i, chunk in enumerate(chunks):
        full_text = f"{chunk['title']}\n{chunk['content']}"

        if extract_keywords_flag:
            keywords = extract_keywords(full_text)
            chunk["keywords"] = keywords
            print(f"      청크 {i+1} 키워드: {keywords}")
        else:
            chunk["keywords"] = []

        if classify_flag:
            classification, confidence = classify_content(full_text)
            chunk["classification"] = classification
            chunk["classification_confidence"] = confidence
            print(f"      청크 {i+1} 분류: {classification} (신뢰도: {confidence:.2f})")
        else:
            chunk["classification"] = "기타"
            chunk["classification_confidence"] = 0.0

    return chunks


# ========== 향상된 임베딩 (Contextual + HyDE) ==========
def generate_contextual_description(chunk_content: str, document_context: str) -> str:
    """청크에 대한 문서 컨텍스트 설명을 생성"""
    try:
        response = client.chat.completions.create(
            model=CONTEXTUAL_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """당신은 고객센터 문서 분석 전문가입니다.
주어진 청크가 전체 문서에서 어떤 맥락과 의미를 가지는지 간결하게 설명하세요.

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
        print(f"   [WARN] Context generation failed: {e}")
        return ""


def generate_hypothetical_queries(chunk_content: str, num_queries: int = 3) -> List[str]:
    """HyDE 방식 - 가상 질문 생성"""
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
3. 다양한 표현 방식 사용
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
        print(f"   [WARN] Hypothetical query generation failed: {e}")
        return []


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
    """청크들을 비동기로 병렬 처리"""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    results = []

    print(f"   [PARALLEL] Processing {len(chunks)} chunks...")
    start_time = time.time()

    doc_keywords_str = ', '.join(document_keywords) if document_keywords else ''

    async def process_single_chunk(chunk: Dict, idx: int) -> Tuple[str, Dict]:
        title = chunk.get('title', '')
        content = chunk.get('content', '')
        chunk_keywords = chunk.get('keywords', [])
        classification = chunk.get('classification', '기타')

        all_keywords = list(chunk_keywords) if chunk_keywords else []
        if document_keywords:
            for kw in document_keywords:
                if kw not in all_keywords:
                    all_keywords.append(kw)

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
                    print(f"      [WARN] Chunk {idx+1} {task_type} failed: {result}")
                    continue

                if task_type == 'contextual' and result:
                    enhanced_parts.append(f"\n[문서 컨텍스트]\n{result}")
                    extra_metadata['contextual_description'] = result
                elif task_type == 'hyde' and result:
                    queries_text = "\n".join([f"- {q}" for q in result])
                    enhanced_parts.append(f"\n[예상 질문]\n{queries_text}")
                    extra_metadata['hypothetical_queries'] = result

        return "\n".join(enhanced_parts), extra_metadata

    tasks = [process_single_chunk(chunk, i) for i, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start_time
    print(f"   [DONE] Parallel processing: {elapsed:.2f}s ({len(chunks)/elapsed:.1f} chunks/s)")

    return results


def enrich_chunks_with_embeddings(
    chunks: List[Dict],
    document_context: str = "",
    document_keywords: List[str] = None,
    use_contextual: bool = True,
    use_hyde: bool = True
) -> List[Dict]:
    """청크들에 향상된 임베딩 정보 추가"""
    try:
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

        for i, (enhanced_text, extra_metadata) in enumerate(results):
            chunks[i]['enhanced_text'] = enhanced_text
            chunks[i]['extra_metadata'] = extra_metadata

        return chunks

    except Exception as e:
        print(f"   [ERROR] Enhanced embedding failed: {e}")
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


def get_document_summary(json_data: List[Dict], file_name: str) -> str:
    """문서 전체 요약 생성"""
    try:
        sample_texts = []
        for item in json_data[:5]:
            sample_texts.append(json.dumps(item, ensure_ascii=False)[:600])

        full_text = "\n".join(sample_texts)

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
        print(f"   [SUMMARY] Document summary generated")
        return f"[문서: {file_name}]\n{summary}"

    except Exception as e:
        print(f"   [WARN] Document summary failed: {e}")
        return f"[문서: {file_name}]"


# ========== ChromaDB 설정 및 저장 ==========
def setup_chromadb(api_key: str, collection_name: str = COLLECTION_NAME):
    """ChromaDB 컬렉션 설정"""
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBEDDING_MODEL
    )

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"}
    )

    return collection


def insert_chunks_to_chroma(chunks: List[Dict], collection, use_enhanced: bool = True):
    """청크를 ChromaDB에 저장"""
    if not chunks:
        print("   [WARN] No chunks to save.")
        return

    documents = []
    metadatas = []
    ids = []
    skipped = 0
    split_count = 0

    for i, chunk in enumerate(chunks):
        keywords_str = ", ".join(chunk.get("keywords", []))
        classification = chunk.get("classification", "기타")

        title = clean_text(chunk.get('title', ''))
        content = clean_text(chunk.get('content', ''))

        if not content:
            print(f"   [WARN] Chunk {i+1} skipped: empty content")
            skipped += 1
            continue

        if use_enhanced and 'enhanced_text' in chunk:
            doc_text = clean_text(chunk['enhanced_text'])
        else:
            doc_text = f"""[분류: {classification}]
[키워드: {keywords_str}]
{title}
{content}"""

        token_count = count_tokens(doc_text)

        if token_count > MAX_EMBEDDING_TOKENS:
            print(f"   [SPLIT] Chunk {i+1} exceeds token limit ({token_count} > {MAX_EMBEDDING_TOKENS}), splitting...")

            content_max_tokens = MAX_EMBEDDING_TOKENS - 500
            content_chunks = split_text_by_tokens(doc_text, max_tokens=content_max_tokens, overlap_tokens=200)

            print(f"      -> Split into {len(content_chunks)} parts")
            split_count += len(content_chunks) - 1

            for j, content_part in enumerate(content_chunks):
                if not content_part.strip():
                    continue

                documents.append(content_part)

                extra_meta = chunk.get('extra_metadata', {})
                doc_keywords = extra_meta.get('document_keywords', [])
                combined_keywords = extra_meta.get('combined_keywords', [])

                metadatas.append({
                    "category": chunk["metadata"]["category"],
                    "title": f"{title[:180]} (Part {j+1}/{len(content_chunks)})" if title else f"Part {j+1}/{len(content_chunks)}",
                    "source": chunk["metadata"]["source"],
                    "chunk_type": chunk["metadata"].get("chunk_type", "json"),
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

                ids.append(f"json_{chunk['metadata']['category']}_{i}_part{j}")

        else:
            if not doc_text.strip():
                print(f"   [WARN] Chunk {i+1} skipped: empty after cleaning")
                skipped += 1
                continue

            documents.append(doc_text)

            extra_meta = chunk.get('extra_metadata', {})
            doc_keywords = extra_meta.get('document_keywords', [])
            combined_keywords = extra_meta.get('combined_keywords', [])

            metadatas.append({
                "category": chunk["metadata"]["category"],
                "title": title[:200] if title else "",
                "source": chunk["metadata"]["source"],
                "chunk_type": chunk["metadata"].get("chunk_type", "json"),
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

            ids.append(f"json_{chunk['metadata']['category']}_{i}")

    if not documents:
        print("   [WARN] No valid documents.")
        return

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
            print(f"   [ERROR] Batch {start}-{end} save failed: {e}")
            for j in range(start, end):
                try:
                    collection.add(
                        documents=[documents[j]],
                        metadatas=[metadatas[j]],
                        ids=[ids[j]]
                    )
                except Exception as e2:
                    print(f"      [ERROR] Individual chunk {j} save failed: {e2}")

    print(f"   [OK] {len(documents)} chunks embedded and saved")
    print(f"      (Original: {len(chunks)}, Split: {split_count}, Skipped: {skipped})")
    if use_enhanced:
        contextual_count = sum(1 for c in chunks if c.get('extra_metadata', {}).get('contextual_description'))
        hyde_count = sum(1 for c in chunks if c.get('extra_metadata', {}).get('hypothetical_queries'))
        print(f"      (Contextual: {contextual_count}, HyDE: {hyde_count})")


# ========== 메인 실행 ==========
def process_all_jsons(
    directory: str,
    api_key: str,
    extract_keywords_flag: bool = True,
    classify_flag: bool = True,
    use_contextual: bool = True,
    use_hyde: bool = True,
    collection_name: str = COLLECTION_NAME
):
    """디렉터리 내 모든 JSON 파일 처리"""

    json_files = get_json_files(directory)

    if not json_files:
        print("[ERROR] No JSON files found.")
        return

    collection = setup_chromadb(api_key, collection_name)

    total_chunks = 0
    total_items = 0
    success_files = 0
    failed_files = []

    print(f"\n{'='*50}")
    print(f"[CONFIG] Settings:")
    print(f"   - Keyword extraction: {'ON' if extract_keywords_flag else 'OFF'}")
    print(f"   - Auto classification: {'ON' if classify_flag else 'OFF'}")
    print(f"   - Contextual Embedding: {'ON' if use_contextual else 'OFF'}")
    print(f"   - HyDE (Hypothetical queries): {'ON' if use_hyde else 'OFF'}")
    print(f"   - Collection name: {collection_name}")
    print(f"{'='*50}\n")

    for file_path in json_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        try:
            # JSON 로드
            json_data = load_json_file(file_path)

            if not json_data:
                print(f"   [WARN] No JSON data")
                failed_files.append(file_path)
                continue

            total_items += len(json_data)

            # 문서 요약 생성
            document_context = ""
            if use_contextual:
                document_context = get_document_summary(json_data, file_name)

            # 문서 키워드 추출
            document_keywords = []
            if use_contextual or use_hyde:
                document_keywords = extract_document_keywords(json_data, file_name)

            # 각 JSON 항목을 마크다운으로 변환 후 청킹
            all_chunks = []
            for idx, item in enumerate(json_data):
                item_name = item.get('name') or item.get('title') or f"항목_{idx+1}"
                markdown_text = json_to_markdown(item, idx)
                chunks = chunk_markdown_semantic(markdown_text, file_name, file_path, item_name)
                all_chunks.extend(chunks)

            print(f"   [CHUNKS] {len(all_chunks)} chunks created (from {len(json_data)} items)")

            if not all_chunks:
                failed_files.append(file_path)
                continue

            # 키워드 & 분류 추가
            all_chunks = enrich_chunks(all_chunks, extract_keywords_flag, classify_flag)

            # 향상된 임베딩 처리
            if use_contextual or use_hyde:
                print(f"   [ENHANCE] Generating enhanced embedding text...")
                all_chunks = enrich_chunks_with_embeddings(
                    all_chunks,
                    document_context=document_context,
                    document_keywords=document_keywords,
                    use_contextual=use_contextual,
                    use_hyde=use_hyde
                )

            # 저장
            use_enhanced = use_contextual or use_hyde
            insert_chunks_to_chroma(all_chunks, collection, use_enhanced=use_enhanced)
            total_chunks += len(all_chunks)
            success_files += 1

        except Exception as e:
            print(f"   [ERROR] Processing failed: {e}")
            import traceback
            traceback.print_exc()
            failed_files.append(file_path)

    print(f"\n{'='*50}")
    print(f"[COMPLETE] All processing done!")
    print(f"[SUCCESS] JSON files: {success_files}")
    print(f"[FAILED] JSON files: {len(failed_files)}")
    if failed_files:
        for f in failed_files:
            print(f"   - {f}")
    print(f"[TOTAL] Items: {total_items}")
    print(f"[TOTAL] Chunks: {total_chunks}")
    print(f"[SAVED] Documents in DB: {collection.count()}")


# ========== 실행 ==========
if __name__ == "__main__":
    # JSON 파일이 있는 디렉터리 지정
    # 기본값: docs_data/testdir
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_json_dir = os.path.join(project_root, "docs_data", "testdir")

    json_directory = JSON_DIRECTORY or default_json_dir

    print(f"\n{'='*60}")
    print(f"[START] JSON Embedding Pipeline")
    print(f"[DIR] {json_directory}")
    print(f"{'='*60}\n")

    process_all_jsons(
        directory=json_directory,
        api_key=OPENAI_API_KEY,
        extract_keywords_flag=True,
        classify_flag=True,
        use_contextual=True,
        use_hyde=True,
        collection_name=COLLECTION_NAME
    )
