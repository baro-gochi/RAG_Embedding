"""
KT 요금제 통합 크롤러 v5
========================
다중 URL 지원:

모바일 요금제 (FilterCode):
- 2: 5G 요금제
- 3: LTE 요금제  
- 4: 시니어/청소년
- 5: 기타

인터넷 요금제 (FilterCode):
- 118: 인터넷
- 119: 인터넷+TV

설치:
    pip install requests beautifulsoup4 selenium webdriver-manager
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field

# Selenium 관련
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("⚠️ Selenium이 설치되지 않았습니다.")


@dataclass
class CategoryConfig:
    """카테고리 설정"""
    name: str
    code: str
    filter_codes: List[Dict[str, str]]  # [{'code': '2', 'name': '5G'}, ...]
    
    def get_urls(self) -> List[Dict[str, str]]:
        """모든 URL 반환"""
        base_url = "https://product.kt.com/wDic/index.do"
        urls = []
        for fc in self.filter_codes:
            if fc['code']:
                url = f"{base_url}?CateCode={self.code}&FilterCode={fc['code']}"
            else:
                url = f"{base_url}?CateCode={self.code}"
            urls.append({
                'url': url,
                'filter_name': fc['name'],
                'filter_code': fc['code']
            })
        return urls


# 카테고리 설정
CATEGORIES = {
    'mobile': CategoryConfig(
        name='모바일',
        code='6002',
        filter_codes=[
            {'code': '2', 'name': '5G 요금제'},
            {'code': '3', 'name': 'LTE 요금제'},
            {'code': '4', 'name': '시니어/청소년'},
            {'code': '5', 'name': '기타'},
        ]
    ),
    'internet': CategoryConfig(
        name='인터넷',
        code='6005',
        filter_codes=[
            {'code': '118', 'name': '인터넷'},
            {'code': '119', 'name': '인터넷+TV'},
        ]
    ),
    'tv': CategoryConfig(
        name='TV',
        code='6008',
        filter_codes=[
            {'code': '', 'name': 'TV 요금제'},
        ]
    )
}


class KTCrawler:
    """KT 요금제 통합 크롤러"""
    
    BASE_URL = "https://product.kt.com"
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        'Connection': 'keep-alive',
        'Referer': 'https://product.kt.com/',
    }
    
    # HTML 구조별 셀렉터
    SELECTORS = {
        'type1': {
            'container': '.N-pdt-accordion-section',
            'section': '.N-pdt-accordion-column',
            'title': '.N-pdt-accordion-tit',
            'content': '.N-pdt-accordion-cont',
            'table': 'table.N-pdt-tbl-plan',
            'lists': ['N-pdt-list', 'N-pdt-num-list', 'N-pdt-desh-list', 'N-pdt-noted-list'],
            'desc': '.N-pdt-tbl-desc',
            'sub_title': 'strong.N-pdt-tit',
            'td_list': 'ul.td-list',
        },
        'type2': {
            'container': '#appendPriceDiv.desc, #appendPriceDiv, .desc',
            'table': 'table.pduct-tbl-plan',
            'lists': ['pduct-list', 'pduct-desh-list', 'pduct-num-list', 'pduct-no-list', 'pduct-noted-list'],
            'desc': '.pduct-tbl-top-desc',
            'sub_title': 'strong.pduct-tit',
            'td_list': 'ul.pduct-list, ul.td-list',
        }
    }
    
    def __init__(self, category_key: str, use_selenium: bool = True, delay: float = 1.0):
        """
        Args:
            category_key: 'mobile' 또는 'internet'
            use_selenium: Selenium 사용 여부
            delay: 요청 간 대기 시간 (초)
        """
        if category_key not in CATEGORIES:
            raise ValueError(f"지원하지 않는 카테고리: {category_key}. 'mobile' 또는 'internet' 사용")
        
        self.category = CATEGORIES[category_key]
        self.category_key = category_key
        self.use_selenium = use_selenium and SELENIUM_AVAILABLE
        self.delay = delay
        self.driver = None
        
        # 메모리 저장소
        self.plan_urls: List[Dict] = []
        self.results: List[Dict] = []
        
    # ========== Selenium 관리 ==========
    
    def _init_selenium(self):
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium이 설치되지 않았습니다.")
            
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument(f'user-agent={self.HEADERS["User-Agent"]}')
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.implicitly_wait(10)
        print("✅ Selenium WebDriver 초기화 완료")
        
    def _close_selenium(self):
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    # ========== 페이지 요청 ==========
    
    def _fetch_with_requests(self, url: str) -> Optional[str]:
        try:
            response = requests.get(url, headers=self.HEADERS, timeout=30)
            response.raise_for_status()
            response.encoding = 'utf-8'
            return response.text
        except requests.RequestException as e:
            print(f"  ❌ requests 실패: {e}")
            return None
    
    def _fetch_with_selenium(self, url: str) -> Optional[str]:
        try:
            if not self.driver:
                self._init_selenium()
            
            self.driver.get(url)
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(2)
            
            # 스크롤 다운
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            while True:
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(0.5)
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
            
            return self.driver.page_source
        except Exception as e:
            print(f"  ❌ Selenium 실패: {e}")
            return None
    
    def fetch_page(self, url: str) -> Optional[str]:
        if self.use_selenium:
            html = self._fetch_with_selenium(url)
            if html:
                return html
        return self._fetch_with_requests(url)
    
    # ========== 1단계: 모든 URL에서 상품 목록 수집 ==========
    
    def collect_plan_urls(self) -> List[Dict]:
        print("\n" + "="*60)
        print(f"📋 1단계: {self.category.name} 요금제 목록 수집")
        print("="*60)
        
        list_urls = self.category.get_urls()
        seen_urls = set()
        
        for list_info in list_urls:
            list_url = list_info['url']
            filter_name = list_info['filter_name']
            
            print(f"\n🔍 [{filter_name}] 스캔 중...")
            print(f"   URL: {list_url}")
            
            html = self.fetch_page(list_url)
            if not html:
                continue
            
            soup = BeautifulSoup(html, 'html.parser')
            plan_items = soup.select('[class*="product"], [class*="plan"], [class*="item"], [class*="card"]')
            
            count = 0
            for item in plan_items:
                name_elem = item.select_one('h2, h3, h4, .title, .name, [class*="title"], [class*="name"]')
                if not name_elem:
                    continue
                
                name = name_elem.get_text(strip=True)
                if not name or len(name) < 2:
                    continue
                
                link = item.select_one('a[href*="productDetail"], a[href*="ItemCode"]')
                if not link:
                    link = item.select_one('a[href]')
                
                if link:
                    href = link.get('href', '')
                    if href and 'productDetail' in href:
                        if not href.startswith('http'):
                            href = self.BASE_URL + href
                        
                        if href not in seen_urls:
                            seen_urls.add(href)
                            self.plan_urls.append({
                                'name': name,
                                'url': href,
                                'filter_name': filter_name,
                                'filter_code': list_info['filter_code']
                            })
                            count += 1
                            print(f"     ✅ {name}")
            
            print(f"   → {count}개 수집")
            time.sleep(self.delay)
        
        print(f"\n📊 총 수집된 상품: {len(self.plan_urls)}개")
        return self.plan_urls
    
    # ========== 유틸리티 ==========
    
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\d+\)', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _parse_list(self, ul_element) -> List[str]:
        items = []
        for li in ul_element.select('li'):
            text = self._clean_text(li.get_text(strip=True))
            if text:
                items.append(text)
        return items
    
    def _parse_table_with_rowspan(self, table, selectors: dict) -> List[Dict]:
        plans = []
        
        headers = []
        thead = table.select_one('thead')
        if thead:
            for th in thead.select('th'):
                colspan = int(th.get('colspan', 1))
                if colspan == 1:
                    text = self._clean_text(th.get_text(strip=True))
                    if text and text not in headers:
                        headers.append(text)
        
        tbody = table.select_one('tbody')
        if not tbody:
            return plans
        
        rows = tbody.select('tr')
        col_count = len(headers) if headers else 15
        rowspan_tracker = {}
        
        for row in rows:
            cells = row.select('th, td')
            row_data = {}
            
            col_index = 0
            cell_index = 0
            
            while col_index < col_count and (cell_index < len(cells) or col_index in rowspan_tracker):
                if col_index in rowspan_tracker and rowspan_tracker[col_index]['remaining'] > 0:
                    value = rowspan_tracker[col_index]['value']
                    rowspan_tracker[col_index]['remaining'] -= 1
                else:
                    if cell_index < len(cells):
                        cell = cells[cell_index]
                        
                        td_list_selector = selectors.get('td_list', 'ul.td-list')
                        ul = cell.select_one(td_list_selector)
                        if ul:
                            value = self._parse_list(ul)
                        else:
                            value = self._clean_text(cell.get_text(strip=True))
                        
                        rowspan = int(cell.get('rowspan', 1))
                        if rowspan > 1:
                            rowspan_tracker[col_index] = {
                                'value': value,
                                'remaining': rowspan - 1
                            }
                        
                        cell_index += 1
                    else:
                        break
                
                if col_index < len(headers):
                    key = headers[col_index]
                else:
                    key = f"col_{col_index}"
                
                row_data[key] = value
                col_index += 1
            
            if row_data:
                plans.append(row_data)
        
        return plans
    
    # ========== 타입1 파싱 ==========
    
    def _parse_type1(self, soup: BeautifulSoup, plan_info: Dict) -> Optional[Dict]:
        selectors = self.SELECTORS['type1']
        
        accordion = soup.select_one(selectors['container'])
        if not accordion:
            return None
        
        result = {
            'name': plan_info['name'],
            'url': plan_info['url'],
            'product_type': self.category.name,
            'filter_name': plan_info['filter_name'],
            'filter_code': plan_info['filter_code'],
            'html_type': 'type1_accordion',
            'sections': {}
        }
        
        columns = accordion.select(selectors['section'])
        
        for column in columns:
            title_btn = column.select_one(selectors['title'])
            if not title_btn:
                continue
            
            section_title = self._clean_text(title_btn.get_text(strip=True))
            if not section_title:
                continue
            
            content_div = column.select_one(selectors['content'])
            if not content_div:
                continue
            
            section_data = self._parse_content_area(content_div, selectors)
            
            if section_data:
                result['sections'][section_title] = section_data
        
        return result if result['sections'] else None
    
    # ========== 타입2 파싱 ==========
    
    def _parse_type2(self, soup: BeautifulSoup, plan_info: Dict) -> Optional[Dict]:
        selectors = self.SELECTORS['type2']
        
        container = None
        for selector in selectors['container'].split(', '):
            container = soup.select_one(selector)
            if container:
                break
        
        if not container:
            return None
        
        result = {
            'name': plan_info['name'],
            'url': plan_info['url'],
            'product_type': self.category.name,
            'filter_name': plan_info['filter_name'],
            'filter_code': plan_info['filter_code'],
            'html_type': 'type2_appendPriceDiv',
            'sections': {}
        }
        
        section_data = self._parse_content_area(container, selectors)
        
        if section_data:
            result['sections']['요금안내'] = section_data
        
        return result if result['sections'] else None
    
    # ========== 콘텐츠 영역 파싱 ==========
    
    def _parse_content_area(self, content_div, selectors: dict) -> Dict:
        section_data = {
            'tables': [],
            'lists': [],
            'notes': [],
            'sub_sections': []
        }
        
        tables = content_div.select(selectors['table'])
        for table in tables:
            table_data = self._parse_table_with_rowspan(table, selectors)
            if table_data:
                section_data['tables'].append(table_data)
        
        for list_class in selectors['lists']:
            lists = content_div.select(f'ul.{list_class}')
            for ul in lists:
                if ul.find_parent('table'):
                    continue
                items = self._parse_list(ul)
                if items:
                    section_data['lists'].extend(items)
        
        desc = content_div.select_one(selectors['desc'])
        if desc:
            section_data['notes'].append(self._clean_text(desc.get_text(strip=True)))
        
        sub_titles = content_div.select(selectors['sub_title'])
        for st in sub_titles:
            text = self._clean_text(st.get_text(strip=True))
            if text:
                section_data['sub_sections'].append(text)
        
        return {k: v for k, v in section_data.items() if v}
    
    # ========== 상세 페이지 파싱 ==========
    
    def parse_detail_page(self, html: str, plan_info: Dict) -> Optional[Dict]:
        soup = BeautifulSoup(html, 'html.parser')
        
        # 타입1 시도
        result = self._parse_type1(soup, plan_info)
        if result and result.get('sections'):
            return result
        
        # 타입2 시도
        result = self._parse_type2(soup, plan_info)
        if result and result.get('sections'):
            return result
        
        # 폴백
        return self._parse_fallback(soup, plan_info)
    
    def _parse_fallback(self, soup: BeautifulSoup, plan_info: Dict) -> Optional[Dict]:
        result = {
            'name': plan_info['name'],
            'url': plan_info['url'],
            'product_type': self.category.name,
            'filter_name': plan_info['filter_name'],
            'filter_code': plan_info['filter_code'],
            'html_type': 'fallback',
            'sections': {}
        }
        
        section_data = {'tables': [], 'lists': [], 'notes': []}
        
        for table_class in ['N-pdt-tbl-plan', 'pduct-tbl-plan']:
            tables = soup.select(f'table.{table_class}')
            for table in tables:
                selectors = self.SELECTORS['type1'] if 'N-pdt' in table_class else self.SELECTORS['type2']
                table_data = self._parse_table_with_rowspan(table, selectors)
                if table_data:
                    section_data['tables'].append(table_data)
        
        all_list_classes = self.SELECTORS['type1']['lists'] + self.SELECTORS['type2']['lists']
        for list_class in all_list_classes:
            lists = soup.select(f'ul.{list_class}')
            for ul in lists:
                if ul.find_parent('table'):
                    continue
                items = self._parse_list(ul)
                if items:
                    section_data['lists'].extend(items)
        
        section_data = {k: v for k, v in section_data.items() if v}
        
        if section_data:
            result['sections']['요금안내'] = section_data
            return result
        
        return None
    
    def crawl_detail_page(self, plan_info: Dict) -> Optional[Dict]:
        print(f"\n  🔍 [{plan_info['name']}] 크롤링...")
        
        html = self.fetch_page(plan_info['url'])
        if not html:
            return None
        
        result = self.parse_detail_page(html, plan_info)
        
        if result:
            sections = result.get('sections', {})
            for section_name, section_data in sections.items():
                tables = len(section_data.get('tables', []))
                total_rows = sum(len(t) for t in section_data.get('tables', []))
                lists = len(section_data.get('lists', []))
                print(f"    📂 {section_name}: 테이블 {tables}개({total_rows}행), 리스트 {lists}개")
        
        return result
    
    # ========== 메인 크롤링 ==========
    
    def crawl(self) -> List[Dict]:
        print("\n" + "="*60)
        print(f"🚀 KT {self.category.name} 요금제 크롤링 시작")
        print("="*60)
        
        try:
            # 1단계: URL 수집
            self.collect_plan_urls()
            
            if not self.plan_urls:
                print("⚠️ 수집된 URL이 없습니다.")
                return []
            
            # 2단계: 상세 페이지 크롤링
            print("\n" + "="*60)
            print("📋 2단계: 상세 페이지 크롤링")
            print("="*60)
            
            for i, plan_info in enumerate(self.plan_urls, 1):
                print(f"\n[{i}/{len(self.plan_urls)}] {plan_info['name']} ({plan_info['filter_name']})")
                
                result = self.crawl_detail_page(plan_info)
                if result and result.get('sections'):
                    self.results.append(result)
                
                time.sleep(self.delay)
            
            print("\n" + "="*60)
            print(f"✅ 크롤링 완료! 총 {len(self.results)}개 상품 정보 수집")
            print("="*60)
            
            return self.results
            
        finally:
            self._close_selenium()
    
    # ========== 저장 ==========
    
    def save_to_json(self, filename: str = None):
        if filename is None:
            filename = f'kt_{self.category.name}_plans.json'
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"💾 JSON 저장: {filename}")
        return filename
    
    def print_summary(self):
        if not self.results:
            print("⚠️ 수집된 데이터가 없습니다.")
            return
        
        print("\n" + "="*60)
        print(f"📊 {self.category.name} 요금제 수집 결과")
        print("="*60)
        
        # 필터별 그룹화
        by_filter = {}
        for result in self.results:
            filter_name = result.get('filter_name', 'unknown')
            if filter_name not in by_filter:
                by_filter[filter_name] = []
            by_filter[filter_name].append(result)
        
        for filter_name, items in by_filter.items():
            print(f"\n📁 [{filter_name}] ({len(items)}개)")
            for item in items:
                sections = item.get('sections', {})
                total_tables = sum(len(s.get('tables', [])) for s in sections.values())
                total_rows = sum(
                    sum(len(t) for t in s.get('tables', []))
                    for s in sections.values()
                )
                print(f"   • {item['name']}: 테이블 {total_tables}개({total_rows}행)")


def crawl_all() -> Dict[str, List[Dict]]:
    """모든 카테고리 크롤링"""
    all_results = {}
    
    for category_key in CATEGORIES.keys():
        print(f"\n{'#'*60}")
        print(f"# {CATEGORIES[category_key].name} 요금제 크롤링")
        print(f"{'#'*60}")
        
        crawler = KTCrawler(category_key, use_selenium=True, delay=1.0)
        results = crawler.crawl()
        
        if results:
            crawler.print_summary()
            crawler.save_to_json()
            all_results[category_key] = results
    
    return all_results


def crawl_mobile() -> List[Dict]:
    """모바일 요금제만 크롤링"""
    crawler = KTCrawler('mobile', use_selenium=True, delay=1.0)
    results = crawler.crawl()
    if results:
        crawler.print_summary()
        crawler.save_to_json()
    return results


def crawl_internet() -> List[Dict]:
    """인터넷 요금제만 크롤링"""
    crawler = KTCrawler('internet', use_selenium=True, delay=1.0)
    results = crawler.crawl()
    if results:
        crawler.print_summary()
        crawler.save_to_json()
    return results


def crawl_tv() -> List[Dict]:
    """TV 요금제만 크롤링"""
    crawler = KTCrawler('tv', use_selenium=True, delay=1.0)
    results = crawler.crawl()
    if results:
        crawler.print_summary()
        crawler.save_to_json()
    return results


def main():
    """메인 실행"""
    print("="*60)
    print("🏢 KT 요금제 크롤러 v5")
    print("="*60)
    
    print("\n📂 모바일 요금제 URL:")
    for fc in CATEGORIES['mobile'].filter_codes:
        print(f"   • {fc['name']}: FilterCode={fc['code']}")
    
    print("\n📂 인터넷 요금제 URL:")
    for fc in CATEGORIES['internet'].filter_codes:
        print(f"   • {fc['name']}: FilterCode={fc['code']}")
    
    print("\n📂 TV 요금제 URL:")
    for fc in CATEGORIES['tv'].filter_codes:
        print(f"   • {fc['name']}: CateCode=6008")
    
    print("\n" + "-"*60)
    print("카테고리 선택:")
    print("  1. 모바일 요금제")
    print("  2. 인터넷 요금제")
    print("  3. TV 요금제")
    print("  4. 전체 (모바일 + 인터넷 + TV)")
    
    choice = input("\n선택 (1/2/3/4): ").strip()
    
    if choice == '1':
        crawl_mobile()
    elif choice == '2':
        crawl_internet()
    elif choice == '3':
        crawl_tv()
    elif choice == '4':
        crawl_all()
    else:
        print("잘못된 선택입니다.")
        return
    
    print("\n✅ 완료!")


if __name__ == '__main__':
    main()