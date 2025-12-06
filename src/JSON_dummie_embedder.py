"""
KT 고객상담용 JSON 임베딩 파이프라인
====================================
- 상담에 필요한 메타데이터 자동 추출 및 임베딩
- 고객 의도(Intent) 기반 가상 질문 생성
- 상담 시나리오 컨텍스트 포함
- 요금/할인 계산 정보 구조화
- Contextual Embedding + HyDE + 상담 특화 기능

주요 기능:
1. 상담 메타데이터 자동 추출 (가격, 할인율, 대상 고객, 조건 등)
2. 고객 의도(Intent) 분류 및 태깅
3. 상담 시나리오별 가상 질문 생성 (HyDE 확장)
4. 비교/추천에 필요한 구조화된 정보 임베딩
5. FAQ 스타일 질문-답변 쌍 생성
"""

import re
import os
import json
import asyncio
import time
from glob import glob
from typing import List, Dict, Optional, Tuple, Any, Set
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import tiktoken
from bs4 import BeautifulSoup
from dataclasses import dataclass, asdict
from enum import Enum

# 환경설정
load_dotenv()
JSON_DIRECTORY = os.getenv("DIR_DUMMIE")
OPENAI_API_KEY = os.getenv("API_KEY")
CLASSIFICATION_CATEGORIES = os.getenv("CATEGORIES")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH_DUMMIE")
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


# ========== 상담 특화 Enum 및 데이터 클래스 ==========
class CustomerIntent(Enum):
    """고객 의도 분류"""
    COST_REDUCTION = "비용절감"          # 요금 낮추고 싶어요
    PLAN_CHANGE = "요금제변경"           # 요금제 바꾸고 싶어요
    PLAN_INQUIRY = "요금제문의"          # 요금제 알아보고 싶어요
    BUNDLE_INQUIRY = "결합문의"          # 결합할인 문의
    BUNDLE_CHANGE = "결합변경"           # 결합 구성 변경
    BENEFIT_INQUIRY = "혜택문의"         # 혜택 알아보고 싶어요
    USAGE_CHECK = "사용량확인"           # 사용량 확인
    BILLING_INQUIRY = "요금문의"         # 청구서/요금 문의
    SERVICE_ADD = "부가서비스추가"       # 부가서비스 추가
    SERVICE_CANCEL = "부가서비스해지"    # 부가서비스 해지
    NAME_CHANGE = "명의변경"             # 명의변경
    CONTRACT_INQUIRY = "약정문의"        # 약정/위약금 문의
    ROAMING = "로밍"                    # 해외 로밍
    GENERAL_INQUIRY = "일반문의"         # 기타 문의


@dataclass
class ConsultationMetadata:
    """상담용 메타데이터 구조"""
    # 기본 정보
    category: str                        # 문서 카테고리
    sub_category: str = ""               # 세부 카테고리
    document_type: str = ""              # 문서 유형 (요금제, 결합, 로밍 등)
    
    # 상담 관련
    customer_intents: List[str] = None   # 관련 고객 의도
    target_customers: List[str] = None   # 대상 고객
    consultation_keywords: List[str] = None  # 상담 키워드
    
    # 가격/할인 정보
    price_info: Dict = None              # 가격 정보
    discount_info: Dict = None           # 할인 정보
    
    # 조건/제한
    conditions: List[str] = None         # 적용 조건
    restrictions: List[str] = None       # 제한사항
    
    # 비교 정보
    comparison_points: List[str] = None  # 비교 포인트
    alternatives: List[str] = None       # 대안 상품
    
    # 프로세스
    required_documents: List[str] = None # 필요 서류
    process_steps: List[str] = None      # 처리 절차
    
    def __post_init__(self):
        # None인 필드를 빈 리스트/딕셔너리로 초기화
        if self.customer_intents is None:
            self.customer_intents = []
        if self.target_customers is None:
            self.target_customers = []
        if self.consultation_keywords is None:
            self.consultation_keywords = []
        if self.price_info is None:
            self.price_info = {}
        if self.discount_info is None:
            self.discount_info = {}
        if self.conditions is None:
            self.conditions = []
        if self.restrictions is None:
            self.restrictions = []
        if self.comparison_points is None:
            self.comparison_points = []
        if self.alternatives is None:
            self.alternatives = []
        if self.required_documents is None:
            self.required_documents = []
        if self.process_steps is None:
            self.process_steps = []


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


# ========== 상담 메타데이터 추출 ==========
def extract_consultation_metadata(item: Dict, file_name: str) -> ConsultationMetadata:
    """JSON 항목에서 상담용 메타데이터 자동 추출"""
    
    metadata = ConsultationMetadata(
        category=item.get('category', file_name),
        document_type=detect_document_type(item, file_name)
    )
    
    # 문서 유형별 메타데이터 추출
    doc_type = metadata.document_type
    
    if doc_type == 'mobile_plan':
        metadata = extract_plan_metadata(item, metadata)
    elif doc_type == 'bundle':
        metadata = extract_bundle_metadata(item, metadata)
    elif doc_type == 'roaming':
        metadata = extract_roaming_metadata(item, metadata)
    elif doc_type == 'billing':
        metadata = extract_billing_metadata(item, metadata)
    elif doc_type == 'tv_service':
        metadata = extract_tv_metadata(item, metadata)
    elif doc_type == 'name_change':
        metadata = extract_name_change_metadata(item, metadata)
    elif doc_type == 'consultation_guide':
        metadata = extract_guide_metadata(item, metadata)
    else:
        metadata = extract_generic_metadata(item, metadata)
    
    return metadata


def detect_document_type(item: Dict, file_name: str) -> str:
    """문서 유형 감지"""
    file_lower = file_name.lower()
    
    # 파일명 기반 판단
    if 'plan' in file_lower or '요금제' in file_lower:
        return 'mobile_plan'
    if 'bundle' in file_lower or '결합' in file_lower:
        return 'bundle'
    if 'roaming' in file_lower or '로밍' in file_lower:
        return 'roaming'
    if 'billing' in file_lower or 'micropayment' in file_lower or '소액결제' in file_lower:
        return 'billing'
    if 'tv' in file_lower:
        return 'tv_service'
    if 'name_change' in file_lower or '명의' in file_lower:
        return 'name_change'
    if 'consultation' in file_lower or 'guide' in file_lower or '상담' in file_lower:
        return 'consultation_guide'
    if 'membership' in file_lower or '멤버십' in file_lower:
        return 'membership'
    if 'internet' in file_lower or '인터넷' in file_lower:
        return 'internet'
    
    # 내용 기반 판단
    item_str = json.dumps(item, ensure_ascii=False).lower()
    if 'monthly_price' in item_str or '월정액' in item_str:
        return 'mobile_plan'
    if 'bundle' in item_str or '결합' in item_str:
        return 'bundle'
    if 'roaming' in item_str or '로밍' in item_str:
        return 'roaming'
    
    return 'generic'


def extract_plan_metadata(item: Dict, metadata: ConsultationMetadata) -> ConsultationMetadata:
    """요금제 관련 메타데이터 추출"""
    
    # 고객 의도 설정
    metadata.customer_intents = [
        CustomerIntent.PLAN_INQUIRY.value,
        CustomerIntent.PLAN_CHANGE.value,
        CustomerIntent.COST_REDUCTION.value
    ]
    
    # plans 배열에서 정보 추출
    plans = item.get('plans', [])
    for plan in plans:
        plan_name = plan.get('name', '')
        price = plan.get('monthly_price')
        
        if plan_name:
            metadata.consultation_keywords.append(plan_name)
        
        if price:
            if not metadata.price_info:
                metadata.price_info = {}
            metadata.price_info[plan_name] = {
                'monthly_price': price,
                'data': plan.get('data', {}),
                'membership': plan.get('membership', '')
            }
        
        # 대상 고객
        targets = plan.get('target_users', [])
        metadata.target_customers.extend(targets)
        
        # OTT 혜택
        ott = plan.get('ott_benefits', {})
        if ott:
            metadata.comparison_points.append(f"{plan_name}: OTT 혜택 포함")
    
    # 할인 정보
    discounts = item.get('promotional_discounts', [])
    for discount in discounts:
        name = discount.get('name', '')
        if name:
            metadata.discount_info[name] = {
                'amount': discount.get('discount_amount'),
                'rate': discount.get('discount_rate'),
                'period': discount.get('commitment_period_months')
            }
    
    # QoS 정보
    qos = item.get('qos_speed_guide', {})
    if qos:
        metadata.consultation_keywords.extend(['QoS', '속도제한', '데이터 소진'])
    
    return metadata


def extract_bundle_metadata(item: Dict, metadata: ConsultationMetadata) -> ConsultationMetadata:
    """결합할인 관련 메타데이터 추출"""
    
    metadata.customer_intents = [
        CustomerIntent.BUNDLE_INQUIRY.value,
        CustomerIntent.BUNDLE_CHANGE.value,
        CustomerIntent.COST_REDUCTION.value
    ]
    
    # 결합 상품 정보
    bundles = item.get('bundle_products', item.get('bundles', []))
    for bundle in bundles:
        name = bundle.get('name', '')
        if name:
            metadata.consultation_keywords.append(name)
        
        # 할인 구조
        discount_struct = bundle.get('discount_structure', {})
        if discount_struct:
            by_lines = discount_struct.get('by_mobile_lines', [])
            for line_info in by_lines:
                lines = line_info.get('lines')
                rate = line_info.get('discount_rate')
                if lines and rate:
                    if not metadata.discount_info:
                        metadata.discount_info = {}
                    metadata.discount_info[f'{name}_{lines}회선'] = {
                        'lines': lines,
                        'rate': rate
                    }
        
        # 조건
        components = bundle.get('components', {})
        if components:
            for comp, req in components.items():
                metadata.conditions.append(f"{comp}: {req}")
    
    # 최적화 전략
    strategies = item.get('bundle_optimization_strategies', [])
    for strategy in strategies:
        scenario = strategy.get('scenario', '')
        if scenario:
            metadata.comparison_points.append(scenario)
    
    # 명의변경 관련
    membership_rules = item.get('bundle_membership_rules', {})
    if membership_rules:
        metadata.consultation_keywords.extend(['명의변경', '결합유지', '회선분리'])
    
    return metadata


def extract_roaming_metadata(item: Dict, metadata: ConsultationMetadata) -> ConsultationMetadata:
    """로밍 관련 메타데이터 추출"""
    
    metadata.customer_intents = [
        CustomerIntent.ROAMING.value,
        CustomerIntent.BENEFIT_INQUIRY.value
    ]
    
    services = item.get('roaming_services', [])
    for service in services:
        name = service.get('name', '')
        if name:
            metadata.consultation_keywords.append(name)
        
        price = service.get('daily_price') or service.get('price')
        if price:
            metadata.price_info[name] = {'price': price}
    
    # VIP 할인
    vip_discounts = item.get('vip_discounts', {})
    if vip_discounts:
        metadata.discount_info['VIP 로밍할인'] = vip_discounts
        metadata.consultation_keywords.append('VIP 로밍할인')
    
    return metadata


def extract_billing_metadata(item: Dict, metadata: ConsultationMetadata) -> ConsultationMetadata:
    """요금/소액결제 관련 메타데이터 추출"""
    
    metadata.customer_intents = [
        CustomerIntent.BILLING_INQUIRY.value,
        CustomerIntent.GENERAL_INQUIRY.value
    ]
    
    # 소액결제 카테고리
    categories = item.get('payment_categories', [])
    for cat in categories:
        name = cat.get('name', '')
        if name:
            metadata.consultation_keywords.append(name)
    
    # 차단/관리
    management = item.get('management_options', {})
    if management:
        metadata.consultation_keywords.extend(['소액결제 차단', '한도설정', '결제관리'])
        metadata.process_steps = management.get('how_to_block', [])
    
    return metadata


def extract_tv_metadata(item: Dict, metadata: ConsultationMetadata) -> ConsultationMetadata:
    """TV 서비스 관련 메타데이터 추출"""
    
    metadata.customer_intents = [
        CustomerIntent.SERVICE_ADD.value,
        CustomerIntent.SERVICE_CANCEL.value,
        CustomerIntent.BENEFIT_INQUIRY.value
    ]
    
    services = item.get('tv_services', item.get('packages', []))
    for service in services:
        name = service.get('name', '')
        if name:
            metadata.consultation_keywords.append(name)
        
        price = service.get('monthly_price')
        if price:
            metadata.price_info[name] = {'monthly_price': price}
    
    return metadata


def extract_name_change_metadata(item: Dict, metadata: ConsultationMetadata) -> ConsultationMetadata:
    """명의변경 관련 메타데이터 추출"""
    
    metadata.customer_intents = [
        CustomerIntent.NAME_CHANGE.value,
        CustomerIntent.BUNDLE_CHANGE.value
    ]
    
    # 필요 서류
    types = item.get('change_types', [])
    for change_type in types:
        docs = change_type.get('required_documents', [])
        metadata.required_documents.extend(docs)
    
    # 처리 절차
    process = item.get('recommended_process', {})
    steps = process.get('steps', [])
    for step in steps:
        desc = step.get('description', '')
        if desc:
            metadata.process_steps.append(desc)
    
    metadata.consultation_keywords.extend(['명의변경', '가족관계증명서', '주민등록등본'])
    
    return metadata


def extract_guide_metadata(item: Dict, metadata: ConsultationMetadata) -> ConsultationMetadata:
    """상담 가이드 관련 메타데이터 추출"""
    
    metadata.customer_intents = [CustomerIntent.GENERAL_INQUIRY.value]
    
    # 추천 프레임워크
    framework = item.get('plan_recommendation_framework', {})
    if framework:
        steps = framework.get('steps', [])
        for step in steps:
            action = step.get('action', '')
            if action:
                metadata.process_steps.append(action)
    
    # 자주 묻는 질문
    common_questions = item.get('common_questions_and_responses', {})
    for q_type, q_info in common_questions.items():
        question = q_info.get('question', '')
        if question:
            metadata.consultation_keywords.append(question[:50])
    
    return metadata


def extract_generic_metadata(item: Dict, metadata: ConsultationMetadata) -> ConsultationMetadata:
    """일반 메타데이터 추출"""
    
    metadata.customer_intents = [CustomerIntent.GENERAL_INQUIRY.value]
    
    # 키워드 추출 (최상위 키 기반)
    for key in item.keys():
        if key not in ['provider', 'category', 'sources', 'last_updated']:
            metadata.consultation_keywords.append(key)
    
    return metadata


# ========== 상담 특화 가상 질문 생성 ==========
def generate_consultation_queries(
    chunk_content: str, 
    metadata: ConsultationMetadata,
    num_queries: int = 5
) -> Dict[str, List[str]]:
    """상담 시나리오 기반 가상 질문 생성 (HyDE 확장)"""
    try:
        intents_str = ', '.join(metadata.customer_intents) if metadata.customer_intents else '일반문의'
        keywords_str = ', '.join(metadata.consultation_keywords[:10]) if metadata.consultation_keywords else ''
        
        response = client.chat.completions.create(
            model=HYDE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"""당신은 KT 고객센터 상담사입니다.
주어진 문서 내용을 보고, 실제 고객이 전화로 물어볼 수 있는 질문들을 생성하세요.

관련 고객 의도: {intents_str}
관련 키워드: {keywords_str}

질문 생성 규칙:
1. 실제 고객이 사용할 법한 자연스러운 구어체
2. 다양한 표현 방식 (직접적/간접적, 긍정/부정)
3. 구체적인 상황이 포함된 질문
4. 비교/추천 요청 형태의 질문
5. 불만/우려 표현이 포함된 질문

JSON 형식으로만 응답:
{{
    "direct_questions": ["요금제 어떻게 바꿔요?", ...],
    "situation_questions": ["저 지금 월 10만원 내는데 너무 비싸서요...", ...],
    "comparison_questions": ["5G 스탠다드랑 초이스 중에 뭐가 나아요?", ...],
    "concern_questions": ["약정하면 위약금 많이 나와요?", ...]
}}"""
                },
                {
                    "role": "user",
                    "content": f"다음 내용에 대한 고객 질문을 생성하세요:\n\n{chunk_content[:2000]}"
                }
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"   [WARN] Consultation query generation failed: {e}")
        return {}


def generate_faq_pairs(chunk_content: str, metadata: ConsultationMetadata) -> List[Dict]:
    """FAQ 형식의 질문-답변 쌍 생성"""
    try:
        response = client.chat.completions.create(
            model=HYDE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """당신은 KT 고객센터 FAQ 작성자입니다.
주어진 내용을 바탕으로 자주 묻는 질문과 간결한 답변 쌍을 생성하세요.

규칙:
1. 질문은 고객 관점에서 자연스럽게
2. 답변은 핵심만 간결하게 (2-3문장)
3. 가격, 조건, 절차 등 구체적 정보 포함

JSON 형식으로만 응답:
{
    "faq_pairs": [
        {"question": "...", "answer": "..."},
        ...
    ]
}"""
                },
                {
                    "role": "user",
                    "content": f"다음 내용으로 FAQ를 생성하세요:\n\n{chunk_content[:1500]}"
                }
            ],
            temperature=0.5,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return result.get('faq_pairs', [])
    except Exception as e:
        print(f"   [WARN] FAQ generation failed: {e}")
        return []


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
            text = soup.get_text(separator=' ', strip=True)
            return text if text else ""

        rows = table.find_all('tr')
        if not rows:
            return ""

        md_lines = []
        max_cols = 0

        for row in rows:
            cells = row.find_all(['td', 'th'])
            col_count = sum(int(cell.get('colspan', 1)) for cell in cells)
            max_cols = max(max_cols, col_count)

        for row_idx, row in enumerate(rows):
            cells = row.find_all(['td', 'th'])
            row_values = []

            for cell in cells:
                cell_text = cell.get_text(separator=' ', strip=True)
                cell_text = re.sub(r'\s+', ' ', cell_text)
                colspan = int(cell.get('colspan', 1))
                row_values.extend([cell_text] + [''] * (colspan - 1))

            while len(row_values) < max_cols:
                row_values.append('')

            md_lines.append("| " + " | ".join(row_values[:max_cols]) + " |")

            if row_idx == 0:
                md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")

        return "\n".join(md_lines)

    except Exception as e:
        print(f"   [WARN] HTML table parsing failed: {e}")
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            return soup.get_text(separator=' ', strip=True)
        except:
            return ""


# ========== JSON을 마크다운으로 변환 (상담 최적화) ==========
def json_to_consultation_markdown(item: Dict, item_index: int = 0, metadata: ConsultationMetadata = None) -> str:
    """JSON 항목을 상담용 마크다운으로 변환 (메타데이터 포함)"""
    markdown_parts = []
    
    # 문서 유형별 변환
    doc_type = metadata.document_type if metadata else 'generic'
    
    if doc_type == 'mobile_plan':
        markdown_parts.append(mobile_plan_to_consultation_md(item, metadata))
    elif doc_type == 'bundle':
        markdown_parts.append(bundle_to_consultation_md(item, metadata))
    else:
        markdown_parts.append(generic_to_consultation_md(item, item_index, metadata))
    
    # 상담 메타데이터 섹션 추가
    if metadata:
        meta_section = generate_metadata_section(metadata)
        if meta_section:
            markdown_parts.append("\n" + meta_section)
    
    return "\n".join(markdown_parts)


def mobile_plan_to_consultation_md(item: Dict, metadata: ConsultationMetadata) -> str:
    """요금제 JSON을 상담용 마크다운으로 변환"""
    md_parts = []
    
    category = item.get('category', '요금제 정보')
    md_parts.append(f"# {category}\n")
    
    plans = item.get('plans', [])
    for plan in plans:
        name = plan.get('name', '')
        price = plan.get('monthly_price', '')
        
        md_parts.append(f"\n## {name}\n")
        
        # 핵심 정보 요약 (상담사가 빠르게 참고할 수 있도록)
        md_parts.append("### 핵심 정보\n")
        md_parts.append(f"- **월정액**: {price:,}원\n" if isinstance(price, int) else f"- **월정액**: {price}\n")
        
        data = plan.get('data', {})
        if data:
            md_parts.append(f"- **기본 데이터**: {data.get('base_amount', '')}\n")
            md_parts.append(f"- **소진 후 속도**: {data.get('after_exhaustion_speed', '')}\n")
        
        membership = plan.get('membership', '')
        if membership:
            md_parts.append(f"- **멤버십 등급**: {membership}\n")
        
        # OTT 혜택 (중요!)
        ott = plan.get('ott_benefits', {})
        if ott:
            md_parts.append("\n### OTT 혜택\n")
            md_parts.append(f"- {ott.get('description', '')}\n")
            options = ott.get('options', [])
            for opt in options:
                opt_name = opt.get('name', '')
                opt_price = opt.get('normal_price', 0)
                md_parts.append(f"  - {opt_name} (정상가 {opt_price:,}원)\n")
        
        # 대상 고객
        targets = plan.get('target_users', [])
        if targets:
            md_parts.append("\n### 추천 대상\n")
            for t in targets:
                md_parts.append(f"- {t}\n")
        
        # 비용 절감 예시
        saving = plan.get('cost_saving_example', {})
        if saving:
            md_parts.append("\n### 비용 절감 예시\n")
            md_parts.append(f"- 시나리오: {saving.get('scenario', '')}\n")
            md_parts.append(f"- 현재 총액: {saving.get('current_total', 0):,}원\n")
            md_parts.append(f"- 변경 후: {saving.get('after_change', 0):,}원\n")
            md_parts.append(f"- **월 절감액: {saving.get('monthly_saving', 0):,}원**\n")
    
    # 선택약정 할인
    discounts = item.get('promotional_discounts', [])
    if discounts:
        md_parts.append("\n## 선택약정 할인\n")
        for disc in discounts:
            name = disc.get('name', '')
            md_parts.append(f"\n### {name}\n")
            
            amount = disc.get('discount_amount')
            rate = disc.get('discount_rate')
            if amount:
                md_parts.append(f"- **할인금액**: {amount:,}원/월\n")
            if rate:
                md_parts.append(f"- **할인율**: {rate}%\n")
            
            period = disc.get('commitment_period_months')
            if period:
                md_parts.append(f"- **약정기간**: {period}개월\n")
            
            fee = disc.get('early_termination_fee', {})
            if fee:
                md_parts.append(f"- **위약금**: {fee.get('calculation', '')}\n")
    
    # QoS 속도 가이드
    qos = item.get('qos_speed_guide', {})
    if qos:
        md_parts.append("\n## 데이터 소진 후 속도 가이드\n")
        speeds = qos.get('speeds', [])
        for speed_info in speeds:
            speed = speed_info.get('speed', '')
            usage = speed_info.get('usage', '')
            md_parts.append(f"- **{speed}**: {usage}\n")
    
    return "\n".join(md_parts)


def bundle_to_consultation_md(item: Dict, metadata: ConsultationMetadata) -> str:
    """결합할인 JSON을 상담용 마크다운으로 변환"""
    md_parts = []
    
    category = item.get('category', '결합할인 정보')
    md_parts.append(f"# {category}\n")
    
    bundles = item.get('bundle_products', item.get('bundles', []))
    for bundle in bundles:
        name = bundle.get('name', '')
        md_parts.append(f"\n## {name}\n")
        
        desc = bundle.get('description', '')
        if desc:
            md_parts.append(f"{desc}\n")
        
        # 할인 구조
        discount_struct = bundle.get('discount_structure', {})
        if discount_struct:
            md_parts.append("\n### 회선별 할인율\n")
            by_lines = discount_struct.get('by_mobile_lines', [])
            
            if by_lines:
                md_parts.append("| 회선 수 | 할인율 |\n")
                md_parts.append("|---------|--------|\n")
                for line_info in by_lines:
                    lines = line_info.get('lines', '')
                    rate = line_info.get('discount_rate', '')
                    md_parts.append(f"| {lines}회선 | {rate}% |\n")
            
            # 추가 할인
            additional = discount_struct.get('additional_mobile_discount', {})
            if additional:
                amount = additional.get('amount_per_line', 0)
                md_parts.append(f"\n**추가 할인**: 회선당 {amount:,}원 추가 할인\n")
    
    # 최적화 전략
    strategies = item.get('bundle_optimization_strategies', [])
    if strategies:
        md_parts.append("\n## 결합 최적화 전략\n")
        for strategy in strategies:
            scenario = strategy.get('scenario', '')
            strat = strategy.get('strategy', '')
            md_parts.append(f"\n### {scenario}\n")
            md_parts.append(f"**전략**: {strat}\n")
            
            example = strategy.get('example', {})
            if example:
                before = example.get('before', {})
                after = example.get('optimized', {})
                if before and after:
                    md_parts.append(f"\n**변경 전**: {before.get('monthly_total', 0):,}원/월\n")
                    md_parts.append(f"**최적화 후**: {after.get('monthly_total', 0):,}원/월\n")
                    savings = after.get('savings_vs_unoptimized', 0)
                    md_parts.append(f"**절감액**: {savings:,}원/월\n")
    
    return "\n".join(md_parts)


def generic_to_consultation_md(item: Dict, item_index: int, metadata: ConsultationMetadata) -> str:
    """일반 JSON을 상담용 마크다운으로 변환"""
    md_parts = []
    
    title = item.get('name') or item.get('title') or item.get('category') or f"항목 {item_index + 1}"
    md_parts.append(f"# {title}\n")
    
    # 재귀적으로 모든 필드 변환
    def process_value(key: str, value: Any, level: int = 2) -> str:
        result = []
        heading = "#" * min(level, 6)
        
        if isinstance(value, dict):
            result.append(f"\n{heading} {key}\n")
            for k, v in value.items():
                result.append(process_value(k, v, level + 1))
        elif isinstance(value, list):
            result.append(f"\n{heading} {key}\n")
            for item in value:
                if isinstance(item, dict):
                    for k, v in item.items():
                        result.append(process_value(k, v, level + 1))
                else:
                    result.append(f"- {item}\n")
        else:
            result.append(f"- **{key}**: {value}\n")
        
        return "".join(result)
    
    skip_keys = {'provider', 'category', 'sources', 'last_updated', 'name', 'title'}
    for key, value in item.items():
        if key not in skip_keys:
            md_parts.append(process_value(key, value))
    
    return "\n".join(md_parts)


def generate_metadata_section(metadata: ConsultationMetadata) -> str:
    """상담 메타데이터 섹션 생성"""
    sections = []
    
    # 고객 의도
    if metadata.customer_intents:
        sections.append(f"[고객 의도: {', '.join(metadata.customer_intents)}]")
    
    # 대상 고객
    if metadata.target_customers:
        sections.append(f"[대상 고객: {', '.join(metadata.target_customers[:5])}]")
    
    # 상담 키워드
    if metadata.consultation_keywords:
        sections.append(f"[상담 키워드: {', '.join(metadata.consultation_keywords[:10])}]")
    
    # 가격 정보 요약
    if metadata.price_info:
        prices = []
        for name, info in list(metadata.price_info.items())[:3]:
            if isinstance(info, dict) and 'monthly_price' in info:
                prices.append(f"{name}: {info['monthly_price']:,}원")
        if prices:
            sections.append(f"[가격 정보: {', '.join(prices)}]")
    
    return "\n".join(sections)


# ========== 마크다운 청킹 ==========
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

        # 메타데이터 라인 (대괄호로 시작)
        if line.strip().startswith('[') and line.strip().endswith(']'):
            blocks.append(MarkdownBlock('metadata', line))
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
                next_line.strip().startswith('[') or
                re.match(r'^(\s*)([-*]|\d+\.)\s+', next_line)):
                break
            para_lines.append(next_line)
            i += 1

        blocks.append(MarkdownBlock('paragraph', '\n'.join(para_lines)))

    return blocks


def chunk_markdown_for_consultation(
    markdown_text: str, 
    file_name: str, 
    file_path: str, 
    item_name: str = "",
    metadata: ConsultationMetadata = None
) -> List[Dict]:
    """상담용 마크다운 청킹 (메타데이터 포함)"""
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
            # 현재까지의 청크 저장
            if current_chunk_blocks and current_length >= MIN_CHUNK_SIZE:
                chunk_content = "\n\n".join([b.content for b in current_chunk_blocks])
                chunks.append(create_consultation_chunk(
                    content=chunk_content,
                    title=current_heading,
                    file_name=file_name,
                    file_path=file_path,
                    heading_context=heading_context.copy(),
                    metadata=metadata
                ))
                current_chunk_blocks = []
                current_length = 0

            # 헤딩 컨텍스트 업데이트
            level = block.level
            heading_context = [h for h in heading_context if h[0] < level]
            heading_text = re.sub(r'^#+\s*', '', block.content).strip()
            heading_context.append((level, heading_text))
            current_heading = " > ".join([h[1] for h in heading_context])

        # 원자적 블록 (테이블, 코드)
        if block.is_atomic():
            if current_chunk_blocks:
                chunk_content = "\n\n".join([b.content for b in current_chunk_blocks])
                chunks.append(create_consultation_chunk(
                    content=chunk_content,
                    title=current_heading,
                    file_name=file_name,
                    file_path=file_path,
                    heading_context=heading_context.copy(),
                    metadata=metadata
                ))
                current_chunk_blocks = []
                current_length = 0

            chunks.append(create_consultation_chunk(
                content=block.content,
                title=current_heading,
                file_name=file_name,
                file_path=file_path,
                heading_context=heading_context.copy(),
                chunk_type=block.block_type,
                metadata=metadata
            ))
            continue

        # 크기 제한 확인
        if current_length + block_length > MAX_CHUNK_SIZE and current_chunk_blocks:
            chunk_content = "\n\n".join([b.content for b in current_chunk_blocks])
            chunks.append(create_consultation_chunk(
                content=chunk_content,
                title=current_heading,
                file_name=file_name,
                file_path=file_path,
                heading_context=heading_context.copy(),
                metadata=metadata
            ))
            current_chunk_blocks = []
            current_length = 0

        current_chunk_blocks.append(block)
        current_length += block_length

    # 마지막 청크
    if current_chunk_blocks:
        chunk_content = "\n\n".join([b.content for b in current_chunk_blocks])
        chunks.append(create_consultation_chunk(
            content=chunk_content,
            title=current_heading,
            file_name=file_name,
            file_path=file_path,
            heading_context=heading_context.copy(),
            metadata=metadata
        ))

    return chunks


def create_consultation_chunk(
    content: str,
    title: str,
    file_name: str,
    file_path: str,
    heading_context: List[Tuple[int, str]],
    chunk_type: str = "text",
    metadata: ConsultationMetadata = None
) -> Dict:
    """상담용 청크 생성"""
    chunk = {
        "title": title,
        "content": content,
        "metadata": {
            "category": file_name,
            "source": file_path,
            "chunk_type": chunk_type,
            "heading_path": " > ".join([h[1] for h in heading_context]) if heading_context else title
        },
        "keywords": [],
        "classification": "기타",
        "classification_confidence": 0.0
    }
    
    # 상담 메타데이터 추가
    if metadata:
        chunk["consultation_metadata"] = {
            "document_type": metadata.document_type,
            "customer_intents": metadata.customer_intents,
            "target_customers": metadata.target_customers[:5] if metadata.target_customers else [],
            "consultation_keywords": metadata.consultation_keywords[:10] if metadata.consultation_keywords else [],
            "price_info": metadata.price_info,
            "discount_info": metadata.discount_info,
            "conditions": metadata.conditions[:5] if metadata.conditions else [],
            "required_documents": metadata.required_documents,
            "process_steps": metadata.process_steps[:5] if metadata.process_steps else []
        }
    
    return chunk


# ========== 키워드 및 분류 ==========
def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """GPT를 이용한 키워드 추출"""
    try:
        response = client.chat.completions.create(
            model=CONTEXTUAL_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"""당신은 KT 고객센터 상담을 위한 키워드 추출 전문가입니다.
주어진 텍스트에서 상담에 유용한 키워드를 {max_keywords}개 추출하세요.

추출 규칙:
1. 요금제명, 서비스명 (예: 5G 초이스, 온가족 결합)
2. 가격/금액 관련 (예: 69,000원, 25% 할인)
3. 고객이 검색할 만한 표현 (예: 요금 줄이기, 인터넷 속도)
4. 조건/자격 관련 (예: VIP, 3회선 이상)
5. 동의어/유사어 포함 (예: 할인/절감/저렴)

JSON 형식으로만 응답: {{"keywords": ["키워드1", "키워드2", ...]}}"""
                },
                {
                    "role": "user",
                    "content": text[:3000]
                }
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        return result.get("keywords", [])[:max_keywords]

    except Exception as e:
        print(f"   [WARN] Keyword extraction failed: {e}")
        return []


def classify_for_consultation(text: str) -> Tuple[str, float]:
    """상담 분류 (고객 의도 기반)"""
    categories = """
- 요금제문의: 요금제 종류, 가격, 혜택 문의
- 요금제변경: 요금제 변경 요청, 절차 문의
- 비용절감: 요금 낮추기, 할인 받기
- 결합문의: 결합상품 문의, 할인율 확인
- 결합변경: 결합 구성 변경, 회선 추가/제거
- 부가서비스: TV, 로밍, 소액결제 등 부가서비스
- 명의변경: 명의변경 절차, 서류
- 약정문의: 약정 조건, 위약금
- 일반문의: 기타 문의
"""
    
    try:
        response = client.chat.completions.create(
            model=CONTEXTUAL_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"""다음 상담 카테고리 중 가장 적합한 것을 선택하세요:
{categories}

JSON 형식으로만 응답: {{"classification": "카테고리명", "confidence": 0.0~1.0}}"""
                },
                {
                    "role": "user",
                    "content": text[:2000]
                }
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        return result.get("classification", "일반문의"), result.get("confidence", 0.0)

    except Exception as e:
        print(f"   [WARN] Classification failed: {e}")
        return "일반문의", 0.0


def enrich_chunks_for_consultation(
    chunks: List[Dict], 
    extract_keywords_flag: bool = True, 
    classify_flag: bool = True
) -> List[Dict]:
    """청크에 상담용 키워드와 분류 추가"""
    for i, chunk in enumerate(chunks):
        full_text = f"{chunk['title']}\n{chunk['content']}"

        if extract_keywords_flag:
            keywords = extract_keywords(full_text)
            chunk["keywords"] = keywords
            print(f"      청크 {i+1} 키워드: {keywords[:5]}...")
        else:
            chunk["keywords"] = []

        if classify_flag:
            classification, confidence = classify_for_consultation(full_text)
            chunk["classification"] = classification
            chunk["classification_confidence"] = confidence
            print(f"      청크 {i+1} 분류: {classification} (신뢰도: {confidence:.2f})")
        else:
            chunk["classification"] = "일반문의"
            chunk["classification_confidence"] = 0.0

    return chunks


# ========== 향상된 임베딩 (Contextual + HyDE + 상담 특화) ==========
def get_document_summary(json_data: List[Dict], file_name: str) -> str:
    """문서 전체 요약 생성"""
    try:
        sample_texts = []
        for item in json_data[:5]:
            sample_texts.append(json.dumps(item, ensure_ascii=False)[:1000])

        full_text = "\n".join(sample_texts)

        response = client.chat.completions.create(
            model=CONTEXTUAL_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """당신은 KT 고객센터 문서 분석가입니다.
주어진 JSON 문서의 전체적인 내용을 요약하세요.

요약 포함 내용:
1. 문서 주제 (어떤 서비스/상품에 대한 정보인지)
2. 포함된 핵심 정보 유형 (가격, 할인, 조건 등)
3. 이 문서가 어떤 고객 문의에 도움이 되는지
4. 주요 키워드 5개"""
                },
                {
                    "role": "user",
                    "content": f"문서명: {file_name}\n\n내용:\n{full_text[:5000]}"
                }
            ],
            temperature=0,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"   [WARN] Document summary failed: {e}")
        return ""


def generate_contextual_description(chunk_content: str, document_context: str) -> str:
    """청크에 대한 문서 컨텍스트 설명을 생성"""
    try:
        response = client.chat.completions.create(
            model=CONTEXTUAL_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """당신은 KT 고객센터 문서 분석 전문가입니다.
주어진 청크가 전체 문서에서 어떤 맥락과 의미를 가지는지 설명하세요.

설명 포함:
1. 이 청크가 다루는 핵심 주제
2. 어떤 고객 질문에 답변할 수 있는지
3. 관련된 다른 정보와의 연결점

2-3문장으로 간결하게 작성하세요."""
                },
                {
                    "role": "user",
                    "content": f"""## 문서 개요
{document_context[:1500]}

## 분석할 청크
{chunk_content[:2000]}"""
                }
            ],
            temperature=0,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"   [WARN] Context generation failed: {e}")
        return ""


def generate_hypothetical_queries(chunk_content: str, metadata: ConsultationMetadata = None, num_queries: int = 5) -> List[str]:
    """상담 시나리오 기반 가상 질문 생성 (HyDE)"""
    try:
        intents_str = ', '.join(metadata.customer_intents) if metadata and metadata.customer_intents else ''
        
        response = client.chat.completions.create(
            model=HYDE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"""당신은 KT 고객센터에 전화하는 고객입니다.
주어진 정보와 관련하여 실제로 전화해서 물어볼 수 있는 질문 {num_queries}개를 생성하세요.

관련 고객 의도: {intents_str}

질문 유형:
1. 직접적 질문: "요금제 어떻게 바꿔요?"
2. 상황 설명형: "저 지금 월 10만원 내는데 너무 비싸서요..."
3. 비교 요청형: "5G 스탠다드랑 초이스 중에 뭐가 나아요?"
4. 우려/불만형: "약정하면 위약금 많이 나와요?"
5. 확인형: "VIP면 로밍 할인 받을 수 있어요?"

JSON 형식으로만 응답: {{"queries": ["질문1", "질문2", ...]}}"""
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


async def generate_hyde_async(chunk_content: str, metadata: ConsultationMetadata, semaphore: asyncio.Semaphore) -> List[str]:
    """비동기 HyDE 질문 생성"""
    async with semaphore:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            generate_hypothetical_queries,
            chunk_content,
            metadata
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
        classification = chunk.get('classification', '일반문의')
        consultation_meta = chunk.get('consultation_metadata', {})

        # 상담 메타데이터에서 추가 정보 추출
        customer_intents = consultation_meta.get('customer_intents', [])
        target_customers = consultation_meta.get('target_customers', [])
        
        all_keywords = list(chunk_keywords) if chunk_keywords else []
        if document_keywords:
            for kw in document_keywords:
                if kw not in all_keywords:
                    all_keywords.append(kw)

        # 상담 최적화된 임베딩 텍스트 구성
        base_text = f"""[상담 분류: {classification}]
[고객 의도: {', '.join(customer_intents) if customer_intents else '일반문의'}]
[대상 고객: {', '.join(target_customers[:3]) if target_customers else '일반'}]
[문서 키워드: {doc_keywords_str}]
[청크 키워드: {', '.join(chunk_keywords) if chunk_keywords else ''}]

{title}

{content}"""

        enhanced_parts = [base_text]
        extra_metadata = {
            'document_keywords': document_keywords or [],
            'combined_keywords': all_keywords,
            'customer_intents': customer_intents,
            'target_customers': target_customers
        }

        # 메타데이터 객체 생성
        meta_obj = ConsultationMetadata(
            category=chunk.get('metadata', {}).get('category', ''),
            customer_intents=customer_intents
        )

        tasks = []
        if use_contextual and document_context:
            tasks.append(('contextual', generate_contextual_async(content, document_context, semaphore)))
        if use_hyde:
            tasks.append(('hyde', generate_hyde_async(content, meta_obj, semaphore)))

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
                    enhanced_parts.append(f"\n[예상 고객 질문]\n{queries_text}")
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
    """청크에 향상된 임베딩 텍스트 추가"""

    results = asyncio.run(process_chunks_async(
        chunks,
        document_context,
        document_keywords,
        use_contextual,
        use_hyde
    ))

    for i, (enhanced_text, extra_metadata) in enumerate(results):
        chunks[i]['embedding_text'] = enhanced_text
        chunks[i]['extra_metadata'] = extra_metadata

    return chunks


# ========== ChromaDB 관리 ==========
def setup_chromadb(api_key: str, collection_name: str = COLLECTION_NAME):
    """ChromaDB 설정 및 컬렉션 생성"""
    
    # 임베딩 함수 설정
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBEDDING_MODEL
    )
    
    # ChromaDB 클라이언트 생성
    client_db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # 컬렉션 생성 또는 가져오기
    try:
        collection = client_db.get_or_create_collection(
            name=collection_name,
            embedding_function=openai_ef,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"[DB] Collection '{collection_name}' ready (documents: {collection.count()})")
    except Exception as e:
        print(f"[ERROR] ChromaDB setup failed: {e}")
        raise

    return collection


def insert_chunks_to_chroma(chunks: List[Dict], collection, use_enhanced: bool = True):
    """청크들을 ChromaDB에 저장 (상담 메타데이터 포함)"""
    
    if not chunks:
        print("   [WARN] No chunks to insert")
        return

    documents = []
    metadatas = []
    ids = []
    
    skipped = 0
    split_count = 0

    for i, chunk in enumerate(chunks):
        # 임베딩 텍스트 선택
        if use_enhanced and 'embedding_text' in chunk:
            doc_text = chunk['embedding_text']
        else:
            doc_text = f"{chunk['title']}\n{chunk['content']}"

        doc_text = clean_text(doc_text)
        title = clean_text(chunk.get('title', ''))
        keywords = chunk.get('keywords', [])
        keywords_str = ', '.join(keywords) if isinstance(keywords, list) else str(keywords)
        classification = chunk.get('classification', '일반문의')
        
        # 상담 메타데이터 추출
        consultation_meta = chunk.get('consultation_metadata', {})
        extra_meta = chunk.get('extra_metadata', {})

        # 토큰 수 확인
        token_count = count_tokens(doc_text)

        if token_count > MAX_EMBEDDING_TOKENS:
            content_chunks = split_text_by_tokens(doc_text, MAX_EMBEDDING_TOKENS - 500, 200)
            print(f"      -> Split into {len(content_chunks)} parts")
            split_count += len(content_chunks) - 1

            for j, content_part in enumerate(content_chunks):
                if not content_part.strip():
                    continue

                documents.append(content_part)

                doc_keywords = extra_meta.get('document_keywords', [])
                combined_keywords = extra_meta.get('combined_keywords', [])
                customer_intents = consultation_meta.get('customer_intents', [])
                hypothetical_queries = extra_meta.get('hypothetical_queries', [])

                metadatas.append({
                    # 기본 메타데이터
                    "category": chunk["metadata"]["category"],
                    "title": f"{title[:180]} (Part {j+1}/{len(content_chunks)})" if title else f"Part {j+1}/{len(content_chunks)}",
                    "source": chunk["metadata"]["source"],
                    "chunk_type": chunk["metadata"].get("chunk_type", "json"),
                    
                    # 키워드 및 분류
                    "keywords": keywords_str[:500] if keywords_str else "",
                    "document_keywords": ', '.join(doc_keywords)[:500] if doc_keywords else "",
                    "combined_keywords": ', '.join(combined_keywords)[:500] if combined_keywords else "",
                    "classification": classification,
                    "classification_confidence": chunk.get("classification_confidence", 0.0),
                    
                    # 상담 특화 메타데이터
                    "document_type": consultation_meta.get("document_type", ""),
                    "customer_intents": ', '.join(customer_intents) if customer_intents else "",
                    "target_customers": ', '.join(consultation_meta.get("target_customers", []))[:300],
                    "consultation_keywords": ', '.join(consultation_meta.get("consultation_keywords", []))[:300],
                    "price_info": json.dumps(consultation_meta.get("price_info", {}), ensure_ascii=False)[:500],
                    "discount_info": json.dumps(consultation_meta.get("discount_info", {}), ensure_ascii=False)[:500],
                    "conditions": ', '.join(consultation_meta.get("conditions", []))[:300],
                    "required_documents": ', '.join(consultation_meta.get("required_documents", []))[:200],
                    
                    # HyDE 및 컨텍스트
                    "has_contextual": bool(extra_meta.get('contextual_description')),
                    "has_hyde": bool(hypothetical_queries),
                    "hypothetical_queries": json.dumps(hypothetical_queries, ensure_ascii=False)[:500],
                    
                    # 분할 정보
                    "is_split": True,
                    "split_part": j + 1,
                    "split_total": len(content_chunks)
                })

                ids.append(f"kt_consultation_{chunk['metadata']['category']}_{i}_part{j}")

        else:
            if not doc_text.strip():
                print(f"   [WARN] Chunk {i+1} skipped: empty after cleaning")
                skipped += 1
                continue

            documents.append(doc_text)

            doc_keywords = extra_meta.get('document_keywords', [])
            combined_keywords = extra_meta.get('combined_keywords', [])
            customer_intents = consultation_meta.get('customer_intents', [])
            hypothetical_queries = extra_meta.get('hypothetical_queries', [])

            metadatas.append({
                # 기본 메타데이터
                "category": chunk["metadata"]["category"],
                "title": title[:200] if title else "",
                "source": chunk["metadata"]["source"],
                "chunk_type": chunk["metadata"].get("chunk_type", "json"),
                
                # 키워드 및 분류
                "keywords": keywords_str[:500] if keywords_str else "",
                "document_keywords": ', '.join(doc_keywords)[:500] if doc_keywords else "",
                "combined_keywords": ', '.join(combined_keywords)[:500] if combined_keywords else "",
                "classification": classification,
                "classification_confidence": chunk.get("classification_confidence", 0.0),
                
                # 상담 특화 메타데이터
                "document_type": consultation_meta.get("document_type", ""),
                "customer_intents": ', '.join(customer_intents) if customer_intents else "",
                "target_customers": ', '.join(consultation_meta.get("target_customers", []))[:300],
                "consultation_keywords": ', '.join(consultation_meta.get("consultation_keywords", []))[:300],
                "price_info": json.dumps(consultation_meta.get("price_info", {}), ensure_ascii=False)[:500],
                "discount_info": json.dumps(consultation_meta.get("discount_info", {}), ensure_ascii=False)[:500],
                "conditions": ', '.join(consultation_meta.get("conditions", []))[:300],
                "required_documents": ', '.join(consultation_meta.get("required_documents", []))[:200],
                
                # HyDE 및 컨텍스트
                "has_contextual": bool(extra_meta.get('contextual_description')),
                "has_hyde": bool(hypothetical_queries),
                "hypothetical_queries": json.dumps(hypothetical_queries, ensure_ascii=False)[:500],
                
                # 분할 정보
                "is_split": False,
                "split_part": 0,
                "split_total": 1
            })

            ids.append(f"kt_consultation_{chunk['metadata']['category']}_{i}")

    if not documents:
        print("   [WARN] No valid documents.")
        return

    # 배치 저장
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


def extract_document_keywords(json_data: List[Dict], file_name: str, max_keywords: int = 15) -> List[str]:
    """JSON 문서 전체에서 핵심 키워드를 추출"""
    try:
        sample_texts = []
        for item in json_data[:10]:
            sample_texts.append(json.dumps(item, ensure_ascii=False)[:500])

        full_text = "\n".join(sample_texts)

        response = client.chat.completions.create(
            model=CONTEXTUAL_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"""당신은 KT 고객센터 문서 분석 전문가입니다.
주어진 문서에서 상담에 유용한 핵심 키워드를 {max_keywords}개 추출하세요.

추출 규칙:
1. 서비스/상품명: 구체적인 서비스나 상품 이름
2. 가격 관련: 금액, 할인율 등
3. 고객 관련 키워드: 고객이 검색할 만한 표현
4. 조건/자격: VIP, 회선 수 등
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
    """디렉터리 내 모든 JSON 파일 처리 (상담 최적화)"""

    json_files = get_json_files(directory)

    if not json_files:
        print("[ERROR] No JSON files found.")
        return

    collection = setup_chromadb(api_key, collection_name)

    total_chunks = 0
    total_items = 0
    success_files = 0
    failed_files = []

    print(f"\n{'='*60}")
    print(f"[CONFIG] KT 고객상담용 임베딩 설정:")
    print(f"   - 키워드 추출: {'ON' if extract_keywords_flag else 'OFF'}")
    print(f"   - 상담 분류: {'ON' if classify_flag else 'OFF'}")
    print(f"   - Contextual Embedding: {'ON' if use_contextual else 'OFF'}")
    print(f"   - HyDE (상담 시나리오 질문): {'ON' if use_hyde else 'OFF'}")
    print(f"   - 상담 메타데이터 추출: ON")
    print(f"   - Collection name: {collection_name}")
    print(f"{'='*60}\n")

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

            # 각 JSON 항목 처리
            all_chunks = []
            for idx, item in enumerate(json_data):
                item_name = item.get('name') or item.get('title') or item.get('category') or f"항목_{idx+1}"
                
                # 상담 메타데이터 추출
                metadata = extract_consultation_metadata(item, file_name)
                print(f"   [META] {item_name}: {metadata.document_type}, 의도: {metadata.customer_intents[:2]}")
                
                # 마크다운 변환
                markdown_text = json_to_consultation_markdown(item, idx, metadata)
                
                # 청킹
                chunks = chunk_markdown_for_consultation(
                    markdown_text, 
                    file_name, 
                    file_path, 
                    item_name,
                    metadata
                )
                all_chunks.extend(chunks)

            print(f"   [CHUNKS] {len(all_chunks)} chunks created (from {len(json_data)} items)")

            if not all_chunks:
                failed_files.append(file_path)
                continue

            # 키워드 & 분류 추가
            all_chunks = enrich_chunks_for_consultation(all_chunks, extract_keywords_flag, classify_flag)

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

    print(f"\n{'='*60}")
    print(f"[COMPLETE] KT 상담용 임베딩 완료!")
    print(f"[SUCCESS] JSON files: {success_files}")
    print(f"[FAILED] JSON files: {len(failed_files)}")
    if failed_files:
        for f in failed_files:
            print(f"   - {f}")
    print(f"[TOTAL] Items: {total_items}")
    print(f"[TOTAL] Chunks: {total_chunks}")
    print(f"[SAVED] Documents in DB: {collection.count()}")
    print(f"{'='*60}")


# ========== 검색 유틸리티 (테스트용) ==========
def search_for_consultation(
    query: str,
    collection,
    n_results: int = 5,
    filter_intent: str = None,
    filter_document_type: str = None
) -> List[Dict]:
    """상담용 검색 (메타데이터 필터링 지원)"""
    
    where_filter = {}
    if filter_intent:
        where_filter["customer_intents"] = {"$contains": filter_intent}
    if filter_document_type:
        where_filter["document_type"] = filter_document_type
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter if where_filter else None,
        include=["documents", "metadatas", "distances"]
    )
    
    formatted_results = []
    for i in range(len(results['documents'][0])):
        formatted_results.append({
            'content': results['documents'][0][i],
            'metadata': results['metadatas'][0][i],
            'distance': results['distances'][0][i]
        })
    
    return formatted_results


# ========== 실행 ==========
if __name__ == "__main__":
    # JSON 파일이 있는 디렉터리 지정
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_json_dir = os.path.join(project_root, "docs_data", "kt_data")

    json_directory = JSON_DIRECTORY or default_json_dir

    print(f"\n{'='*60}")
    print(f"[START] KT 고객상담용 JSON Embedding Pipeline")
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