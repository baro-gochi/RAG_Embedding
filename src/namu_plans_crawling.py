"""
나무위키 KT 요금제 크롤러 v2
============================
나무위키의 KT 요금제 페이지에서 요금제 정보를 크롤링합니다.
계층 구조(h3 > h4 > h5)를 유지하여 요금제를 그룹화합니다.

URL: https://namu.wiki/w/KT/요금제

설치:
    pip install requests beautifulsoup4 selenium webdriver-manager
"""

import requests
from bs4 import BeautifulSoup, NavigableString
import json
import time
import re
from typing import List, Dict, Optional
from dataclasses import dataclass

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


class NamuWikiKTCrawler:
    """나무위키 KT 요금제 크롤러 (계층 구조 지원)"""
    
    URL = "https://namu.wiki/w/KT/%EC%9A%94%EA%B8%88%EC%A0%9C"
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        'Connection': 'keep-alive',
    }
    
    def __init__(self, use_selenium: bool = True):
        self.use_selenium = use_selenium and SELENIUM_AVAILABLE
        self.driver = None
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
    
    def _fetch_with_selenium(self, url: str) -> Optional[str]:
        try:
            if not self.driver:
                self._init_selenium()
            
            print(f"🌐 페이지 로딩 중: {url}")
            self.driver.get(url)
            
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
            time.sleep(3)
            
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            while True:
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
            
            print("✅ 페이지 로딩 완료")
            return self.driver.page_source
            
        except Exception as e:
            print(f"❌ Selenium 실패: {e}")
            return None
    
    def fetch_page(self, url: str = None) -> Optional[str]:
        if url is None:
            url = self.URL
        return self._fetch_with_selenium(url)
    
    # ========== 유틸리티 ==========
    
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _extract_cell_text(self, cell) -> str:
        content_div = cell.select_one('div.IBdgNaCn')
        if content_div:
            return self._clean_text(content_div.get_text(strip=True))
        return self._clean_text(cell.get_text(strip=True))
    
    def _get_heading_name(self, heading) -> str:
        """h3/h4/h5 태그에서 실제 이름 추출"""
        for span in heading.select('span[id]'):
            name = span.get('id', '')
            if name and not name.startswith('s-') and not name.startswith('rfn-'):
                return name
        return ''
    
    # ========== 테이블 파싱 ==========
    
    def _parse_table(self, table) -> Dict:
        """단일 테이블 - HTML 원본으로 저장"""
        
        # 테이블 이름 추출 (첫 번째 행에서)
        table_name = ''
        first_row = table.select_one('tr.R4S-40tq')
        if first_row:
            first_cell = first_row.select_one('td')
            if first_cell:
                table_name = self._extract_cell_text(first_cell)
        
        # HTML 원본 저장 (정리된 버전)
        table_html = self._clean_table_html(table)
        
        result = {
            'table_name': table_name,
            'table_html': table_html
        }
        
        return result
    
    def _clean_table_html(self, table) -> str:
        """테이블 HTML 정리 (불필요한 속성 제거)"""
        import copy
        
        # 테이블 복사본 생성
        table_copy = copy.copy(table)
        
        # 원본 HTML 반환 (문자열로)
        html_str = str(table)
        
        # 나무위키 특유의 data-v-cf63095b 등 제거 (선택적)
        html_str = re.sub(r'\s*data-v-[a-z0-9]+=""', '', html_str)
        html_str = re.sub(r'\s*data-dark-style="[^"]*"', '', html_str)
        
        return html_str
    
    def _extract_description_after_table(self, table) -> str:
        """테이블 다음의 설명 텍스트 추출"""
        descriptions = []
        parent = table.find_parent('div', class_='pCELUZmY')
        if parent:
            next_div = parent.find_next_sibling('div', class_='IBdgNaCn')
            if next_div:
                text = self._clean_text(next_div.get_text(strip=True))
                if text:
                    descriptions.append(text)
        return ' '.join(descriptions)
    
    # ========== 계층 구조 파싱 ==========
    
    def _parse_hierarchical_structure(self, soup: BeautifulSoup) -> List[Dict]:
        """h3 > h4 > h5 계층 구조 파싱
        
        HTML 구조:
        div (h3와 같은 레벨)
        div (h4와 같은 레벨) 
        div (h5와 같은 레벨)
        div (테이블 - h5와 같은 레벨의 형제 요소)
        """
        
        # 문서의 모든 주요 요소를 순서대로 수집 (헤딩 + 테이블)
        # 최상위 컨테이너에서 순차적으로 탐색
        all_elements = []
        
        # 모든 h3, h4, h5, table을 포함하는 div들을 찾음
        for elem in soup.select('h3.PVbZbzR7, h4.PVbZbzR7, h5.PVbZbzR7, table._3lpnOiRq'):
            all_elements.append(elem)
        
        # 문서 순서대로 정렬 (sourceline 기준)
        # BeautifulSoup에서는 요소 순서가 이미 문서 순서
        
        # 계층 구조로 조직화
        h3_sections = []
        current_h3 = None
        current_h4 = None
        current_h5 = None
        
        for elem in all_elements:
            if elem.name in ['h3', 'h4', 'h5']:
                # 헤딩 처리
                name = self._get_heading_name(elem)
                if not name:
                    continue
                
                level = elem.name
                
                if level == 'h3':
                    # 이전 h5 저장
                    if current_h5 and current_h4:
                        current_h4['plans'].append(current_h5)
                    # 이전 h4 저장
                    if current_h4 and current_h3:
                        current_h3['sub_categories'].append(current_h4)
                    # 이전 h3 저장
                    if current_h3:
                        h3_sections.append(current_h3)
                    
                    current_h3 = {
                        'category': name,
                        'sub_categories': [],
                        'direct_tables': []
                    }
                    current_h4 = None
                    current_h5 = None
                    
                elif level == 'h4':
                    # 이전 h5 저장
                    if current_h5 and current_h4:
                        current_h4['plans'].append(current_h5)
                    # 이전 h4 저장
                    if current_h4 and current_h3:
                        current_h3['sub_categories'].append(current_h4)
                    
                    current_h4 = {
                        'name': name,
                        'plans': [],
                        'direct_tables': []
                    }
                    current_h5 = None
                    
                elif level == 'h5':
                    # 이전 h5 저장
                    if current_h5 and current_h4:
                        current_h4['plans'].append(current_h5)
                    
                    # h4가 없으면 기본 h4 생성
                    if current_h4 is None:
                        current_h4 = {
                            'name': '기타',
                            'plans': [],
                            'direct_tables': []
                        }
                    
                    current_h5 = {
                        'plan_name': name,
                        'tables': []
                    }
                    
            elif elem.name == 'table':
                # 테이블 처리
                parsed = self._parse_table(elem)
                parsed['description'] = self._extract_description_after_table(elem)
                
                # 현재 h5가 있으면 h5에 추가
                if current_h5 is not None:
                    current_h5['tables'].append(parsed)
                # h5가 없고 h4가 있으면 h4 직속 테이블
                elif current_h4 is not None:
                    current_h4['direct_tables'].append(parsed)
                # h4도 없고 h3만 있으면 h3 직속 테이블
                elif current_h3 is not None:
                    current_h3['direct_tables'].append(parsed)
        
        # 마지막 데이터 저장
        if current_h5 and current_h4:
            current_h4['plans'].append(current_h5)
        if current_h4 and current_h3:
            current_h3['sub_categories'].append(current_h4)
        if current_h3:
            h3_sections.append(current_h3)
        
        return h3_sections
    
    # ========== 메인 크롤링 ==========
    
    def parse_page(self, html: str) -> List[Dict]:
        """HTML 파싱하여 계층 구조로 요금제 정보 추출"""
        soup = BeautifulSoup(html, 'html.parser')
        
        print("\n🔍 계층 구조 분석 중 (h3 > h4 > h5)...")
        
        results = self._parse_hierarchical_structure(soup)
        
        # 결과 요약 출력
        for h3_section in results:
            print(f"\n📁 {h3_section['category']}")
            for h4_section in h3_section.get('sub_categories', []):
                plan_count = len(h4_section.get('plans', []))
                direct_count = len(h4_section.get('direct_tables', []))
                print(f"   📂 {h4_section['name']} (요금제 {plan_count}개, 직속 테이블 {direct_count}개)")
                
                for plan in h4_section.get('plans', []):
                    table_count = len(plan.get('tables', []))
                    print(f"      ✅ {plan['plan_name']} (테이블 {table_count}개)")
        
        return results
    
    def crawl(self, html: str = None) -> List[Dict]:
        """크롤링 실행"""
        print("\n" + "="*60)
        print("🚀 나무위키 KT 요금제 크롤링 시작 (v2 - 계층구조)")
        print("="*60)
        
        try:
            if html is None:
                html = self.fetch_page()
            
            if not html:
                print("❌ HTML을 가져올 수 없습니다.")
                return []
            
            self.results = self.parse_page(html)
            
            # 통계
            total_h4 = 0
            total_plans = 0
            total_tables = 0
            
            for h3 in self.results:
                for h4 in h3.get('sub_categories', []):
                    total_h4 += 1
                    total_tables += len(h4.get('direct_tables', []))
                    for plan in h4.get('plans', []):
                        total_plans += 1
                        total_tables += len(plan.get('tables', []))
            
            print("\n" + "="*60)
            print(f"✅ 크롤링 완료!")
            print(f"   📁 대분류(h3): {len(self.results)}개")
            print(f"   📂 중분류(h4): {total_h4}개")
            print(f"   📄 요금제(h5): {total_plans}개")
            print(f"   📊 총 테이블: {total_tables}개")
            print("="*60)
            
            return self.results
            
        finally:
            self._close_selenium()
    
    def crawl_from_file(self, filepath: str) -> List[Dict]:
        """로컬 HTML 파일에서 크롤링"""
        print(f"\n📂 파일에서 HTML 로딩: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            html = f.read()
        
        return self.crawl(html)
    
    # ========== 저장 ==========
    
    def save_to_json(self, filename: str = 'namu_kt_plans.json'):
        """JSON으로 저장"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"💾 JSON 저장: {filename}")
        return filename
    
    def print_summary(self):
        """결과 요약 출력"""
        if not self.results:
            print("⚠️ 수집된 데이터가 없습니다.")
            return
        
        print("\n" + "="*60)
        print("📊 KT 요금제 수집 결과 (계층 구조)")
        print("="*60)
        
        for h3_section in self.results:
            print(f"\n{'='*50}")
            print(f"📁 {h3_section['category']}")
            print(f"{'='*50}")
            
            for h4_section in h3_section.get('sub_categories', []):
                print(f"\n  📂 {h4_section['name']}")
                print(f"  {'-'*40}")
                
                # h4 직속 테이블 (h5가 없는 경우)
                for table in h4_section.get('direct_tables', []):
                    print(f"\n    📋 {table.get('table_name', '(테이블)')}")
                    html_len = len(table.get('table_html', ''))
                    print(f"       HTML 길이: {html_len} chars")
                
                # h5 요금제들
                for plan in h4_section.get('plans', []):
                    print(f"\n    🏷️ {plan['plan_name']}")
                    
                    # 해당 요금제의 모든 테이블
                    for idx, table in enumerate(plan.get('tables', []), 1):
                        if len(plan.get('tables', [])) > 1:
                            print(f"       [테이블 {idx}] {table.get('table_name', '')}")
                        else:
                            print(f"       테이블: {table.get('table_name', '')}")
                        html_len = len(table.get('table_html', ''))
                        print(f"       HTML 길이: {html_len} chars")
                        if table.get('description'):
                            print(f"       📝 {table['description'][:50]}...")


def main():
    """메인 실행"""
    print("="*60)
    print("🏢 나무위키 KT 요금제 크롤러 v2 (계층구조)")
    print("="*60)
    print(f"\n📍 URL: {NamuWikiKTCrawler.URL}")
    print("\n📋 구조: h3(대분류) > h4(중분류) > h5(요금제) > 테이블들")
    
    print("\n" + "-"*60)
    print("실행 방법 선택:")
    print("  1. Selenium으로 직접 크롤링")
    print("  2. 로컬 HTML 파일에서 크롤링")
    
    choice = input("\n선택 (1/2): ").strip()
    
    crawler = NamuWikiKTCrawler(use_selenium=True)
    
    if choice == '1':
        results = crawler.crawl()
    elif choice == '2':
        filepath = input("HTML 파일 경로: ").strip()
        results = crawler.crawl_from_file(filepath)
    else:
        print("잘못된 선택입니다.")
        return
    
    if results:
        crawler.print_summary()
        crawler.save_to_json()
    
    print("\n✅ 완료!")


if __name__ == '__main__':
    main()