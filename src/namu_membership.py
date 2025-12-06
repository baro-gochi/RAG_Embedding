#!/usr/bin/env python3
"""
나무위키 KT 멤버십 정보 크롤러
https://namu.wiki/w/KT%20멤버십

테이블은 HTML 원본으로 저장
"""

import json
import re
import os
from datetime import datetime
from typing import Dict, List, Optional
from bs4 import BeautifulSoup

# Selenium (나무위키 직접 접근 시 필요)
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


class NamuKTMembershipCrawler:
    """나무위키 KT 멤버십 크롤러"""
    
    # 나무위키 URL
    TARGET_URL = "https://namu.wiki/w/KT%20%EB%A9%A4%EB%B2%84%EC%8B%AD"
    
    # 나무위키 CSS 셀렉터
    SELECTORS = {
        'h2': 'h2.PVbZbzR7',      # 대분류 (2. 등급, 3. 혜택 등)
        'h3': 'h3.PVbZbzR7',      # 중분류
        'h4': 'h4.PVbZbzR7',      # 소분류
        'h5': 'h5.PVbZbzR7',      # 세부항목
        'h6': 'h6.PVbZbzR7',      # 더 세부항목
        'table': 'table._3lpnOiRq',
        'table_row': 'tr.R4S-40tq',
        'cell_content': 'div.IBdgNaCn',
    }
    
    def __init__(self):
        self.driver = None
        self.soup = None
        self.results = []
        
    # ========== 드라이버/파일 로드 ==========
    
    def setup_driver(self, headless: bool = False):
        """Selenium 드라이버 설정 (봇 탐지 우회)"""
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium이 설치되지 않았습니다: pip install selenium")
        
        options = Options()
        
        # headless 모드 (선택적 - 나무위키는 headless 차단할 수 있음)
        if headless:
            options.add_argument('--headless=new')
        
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        
        # 봇 탐지 우회 설정
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option('excludeSwitches', ['enable-automation'])
        options.add_experimental_option('useAutomationExtension', False)
        
        # 실제 브라우저처럼 보이게
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        self.driver = webdriver.Chrome(options=options)
        
        # navigator.webdriver 속성 숨기기
        self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': '''
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                })
            '''
        })
        
        self.driver.implicitly_wait(10)
        print("✅ Chrome 드라이버 설정 완료 (봇 탐지 우회 적용)")
    
    def load_from_url(self, url: str = None, headless: bool = False):
        """URL에서 페이지 로드 (Selenium)"""
        if not self.driver:
            self.setup_driver(headless=headless)
        
        target_url = url or self.TARGET_URL
        print(f"🌐 페이지 로드 중: {target_url}")
        
        self.driver.get(target_url)
        
        import time
        time.sleep(3)  # 초기 로딩 대기
        
        # 여러 가지 선택자로 페이지 로드 확인
        selectors_to_try = [
            'article',
            'div.wiki-content',
            'div.content',
            'table',
            'h2'
        ]
        
        page_loaded = False
        for selector in selectors_to_try:
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                print(f"  ✓ '{selector}' 요소 발견")
                page_loaded = True
                break
            except:
                continue
        
        if not page_loaded:
            print("  ⚠️ 특정 요소를 찾지 못했지만 계속 진행합니다...")
        
        time.sleep(2)  # 추가 대기
        
        html = self.driver.page_source
        self.soup = BeautifulSoup(html, 'html.parser')
        print(f"✅ 페이지 로드 완료 (HTML 크기: {len(html):,} bytes)")
        
        # 페이지 내용 간단히 확인
        tables = self.soup.select('table')
        headings = self.soup.select('h2, h3, h4, h5')
        print(f"   발견된 테이블: {len(tables)}개, 헤딩: {len(headings)}개")
    
    def load_from_file(self, filepath: str):
        """로컬 HTML 파일 로드"""
        print(f"📂 파일 로드 중: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            html = f.read()
        
        self.soup = BeautifulSoup(html, 'html.parser')
        print(f"✅ 파일 로드 완료 (HTML 크기: {len(html):,} bytes)")
    
    def close(self):
        """드라이버 종료"""
        if self.driver:
            self.driver.quit()
            print("🔒 드라이버 종료")
    
    # ========== 텍스트 추출 ==========
    
    def _extract_heading_text(self, element) -> str:
        """헤딩 텍스트 추출 (앵커 등 제외)"""
        text = element.get_text(strip=True)
        # 앞의 번호 제거 (예: "2. 등급" -> "등급")
        text = re.sub(r'^\d+(\.\d+)*\.?\s*', '', text)
        # [편집] 등 제거
        text = re.sub(r'\[편집\]', '', text)
        return text.strip()
    
    def _extract_cell_text(self, cell) -> str:
        """셀 내부 텍스트 추출"""
        content_div = cell.select_one(self.SELECTORS['cell_content'])
        if content_div:
            return content_div.get_text(strip=True)
        return cell.get_text(strip=True)
    
    def _get_following_text(self, element) -> str:
        """요소 뒤의 일반 텍스트 추출 (다음 구조 요소 전까지)"""
        texts = []
        for sibling in element.find_next_siblings():
            # 구조 요소를 만나면 중단
            if sibling.name in ['h2', 'h3', 'h4', 'h5', 'h6'] or \
               sibling.select_one('h2, h3, h4, h5, h6'):
                break
            
            # 텍스트만 추출
            text = sibling.get_text(strip=True)
            if text:
                texts.append(text)
        
        return ' '.join(texts)[:500] if texts else ''
    
    def _get_section_content(self, start_element, stop_selectors='h2.PVbZbzR7, h3.PVbZbzR7, h4.PVbZbzR7, h5.PVbZbzR7, h6.PVbZbzR7') -> dict:
        """섹션의 전체 콘텐츠 추출 (텍스트 + 리스트 + 테이블)
        
        나무위키 구조:
        - 헤딩(h2/h3...) 다음에 div.Sr34rLtU 또는 div.woJSxwej가 콘텐츠를 감싸고 있음
        - 실제 콘텐츠는 div.SYMiuyiZ, div.IBdgNaCn, ul.TcQf+vBD 등에 있음
        """
        content = {
            'text': [],           # 일반 텍스트
            'list_items': [],     # 리스트 항목들
            'tables': []          # 테이블 (HTML)
        }
        
        # 헤딩의 부모 컨테이너를 찾아서 다음 형제들을 탐색
        # 나무위키는 div.woJSxwej 안에 헤딩이 있고, 콘텐츠는 다음 형제 div에 있음
        parent = start_element.parent
        if parent:
            parent = parent.parent  # 한 단계 더 올라감
        
        if not parent:
            parent = start_element
        
        # 현재 헤딩 이후의 모든 요소들을 순회
        current = start_element
        
        # find_all_next로 이후 모든 요소 탐색
        for next_elem in start_element.find_all_next():
            # 다음 헤딩을 만나면 중단
            if next_elem.name in ['h2', 'h3', 'h4', 'h5', 'h6']:
                if 'PVbZbzR7' in next_elem.get('class', []):
                    break
            
            # 테이블 처리 (class="_3lpnOiRq")
            if next_elem.name == 'table' and '_3lpnOiRq' in next_elem.get('class', []):
                table_data = self._parse_table(next_elem)
                content['tables'].append(table_data)
                continue
            
            # 리스트 항목 처리 (ul.TcQf+vBD 내부의 li)
            if next_elem.name == 'li':
                # 부모가 ul.TcQf+vBD인지 확인
                parent_ul = next_elem.parent
                if parent_ul and parent_ul.name == 'ul':
                    item_text = next_elem.get_text(strip=True)
                    if item_text and item_text not in content['list_items']:
                        content['list_items'].append(item_text)
                continue
            
            # 텍스트 콘텐츠 처리 (div.IBdgNaCn 내부 텍스트)
            if next_elem.name == 'div' and 'IBdgNaCn' in next_elem.get('class', []):
                # 직접 텍스트만 추출 (자식 태그의 텍스트 제외하고)
                text = next_elem.get_text(strip=True)
                if text and len(text) > 5:
                    # 이미 추가된 텍스트인지 확인
                    is_duplicate = False
                    for existing in content['text']:
                        if text in existing or existing in text:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        content['text'].append(text)
        
        return content
    
    def _get_section_content_html(self, start_element, stop_tags=['h2', 'h3', 'h4', 'h5', 'h6']) -> str:
        """섹션의 전체 HTML 콘텐츠 추출"""
        html_parts = []
        
        for next_elem in start_element.find_all_next():
            # 다음 헤딩을 만나면 중단
            if next_elem.name in stop_tags:
                if 'PVbZbzR7' in next_elem.get('class', []):
                    break
            
            # HTML 추가
            html_parts.append(str(next_elem))
        
        return '\n'.join(html_parts)
    
    # ========== 테이블 파싱 ==========
    
    def _parse_table(self, table) -> Dict:
        """테이블 - HTML 원본으로 저장"""
        
        # 테이블 이름 추출 (첫 번째 행에서)
        table_name = ''
        first_row = table.select_one('tr.R4S-40tq')
        if first_row:
            first_cell = first_row.select_one('td')
            if first_cell:
                table_name = self._extract_cell_text(first_cell)
        
        # HTML 원본 저장 (정리된 버전)
        table_html = self._clean_table_html(table)
        
        return {
            'table_name': table_name,
            'table_html': table_html
        }
    
    def _clean_table_html(self, table) -> str:
        """테이블 HTML 정리"""
        html_str = str(table)
        
        # 나무위키 특유의 data-v 속성 제거
        html_str = re.sub(r'\s*data-v-[a-z0-9]+=""', '', html_str)
        html_str = re.sub(r'\s*data-dark-style="[^"]*"', '', html_str)
        
        return html_str
    
    # ========== 메인 파싱 로직 ==========
    
    # 추출할 h2 섹션 번호 (3, 4, 5, 6번)
    TARGET_SECTIONS = ['3.', '4.', '5.', '6.']
    
    def _is_target_section(self, heading_text: str) -> bool:
        """추출 대상 섹션인지 확인 (3., 4., 5., 6.으로 시작)"""
        text = heading_text.strip()
        for prefix in self.TARGET_SECTIONS:
            if text.startswith(prefix):
                return True
        return False
    
    def _get_section_number(self, heading_text: str) -> str:
        """섹션 번호 추출 (예: '3.1.' -> '3')"""
        text = heading_text.strip()
        match = re.match(r'^(\d+)\.', text)
        if match:
            return match.group(1)
        return ''
    
    def parse_membership_content(self):
        """KT 멤버십 내용 파싱 (계층 구조) - 3,4,5,6번 섹션만"""
        if not self.soup:
            raise ValueError("먼저 load_from_url() 또는 load_from_file()을 호출하세요")
        
        print("\n📊 KT 멤버십 정보 파싱 중...")
        print(f"📌 추출 대상 섹션: {self.TARGET_SECTIONS}")
        
        # 본문 영역 찾기
        article = self.soup.select_one('article') or self.soup
        
        # 모든 구조 요소 수집 (h2, h3, h4, h5, h6, table)
        all_elements = article.select('h2.PVbZbzR7, h3.PVbZbzR7, h4.PVbZbzR7, h5.PVbZbzR7, h6.PVbZbzR7, table._3lpnOiRq')
        
        print(f"📌 발견된 구조 요소: {len(all_elements)}개")
        
        # 요소별 카운트
        counts = {'h2': 0, 'h3': 0, 'h4': 0, 'h5': 0, 'h6': 0, 'table': 0}
        for elem in all_elements:
            if elem.name == 'table':
                counts['table'] += 1
            elif elem.name in counts:
                counts[elem.name] += 1
        print(f"   h2: {counts['h2']}, h3: {counts['h3']}, h4: {counts['h4']}, h5: {counts['h5']}, h6: {counts['h6']}, table: {counts['table']}")
        
        # 계층 구조로 파싱 - 대상 섹션만
        self.results = []
        
        current_h2 = None
        current_h3 = None
        current_h4 = None
        current_h5 = None
        
        # 현재 활성 섹션 추적 (3,4,5,6번 섹션 내부인지)
        in_target_section = False
        current_main_section = ''  # 현재 메인 섹션 번호 (3, 4, 5, 6)
        
        # 헤딩 요소만 먼저 수집
        heading_elements = [e for e in all_elements if e.name in ['h2', 'h3', 'h4', 'h5', 'h6']]
        
        for i, elem in enumerate(heading_elements):
            # 다음 헤딩 찾기 (콘텐츠 범위 결정용)
            next_heading = heading_elements[i + 1] if i + 1 < len(heading_elements) else None
            
            if elem.name == 'h2':
                # H2를 만나면 새로운 섹션 시작
                heading_text = self._extract_heading_text(elem)
                raw_text = elem.get_text(strip=True)
                
                # 대상 섹션인지 확인
                if self._is_target_section(raw_text):
                    in_target_section = True
                    current_main_section = self._get_section_number(raw_text)
                    
                    # 섹션 콘텐츠 추출
                    content = self._get_section_content(elem)
                    
                    current_h2 = {
                        'section': heading_text,
                        'section_number': raw_text.split()[0] if raw_text else '',
                        'content': content,
                        'sub_sections': []
                    }
                    self.results.append(current_h2)
                    current_h3 = None
                    current_h4 = None
                    current_h5 = None
                    print(f"  ✅ H2: {heading_text} (섹션 {current_main_section}) - 텍스트:{len(content['text'])}, 리스트:{len(content['list_items'])}, 테이블:{len(content['tables'])}")
                else:
                    # 대상이 아닌 섹션
                    section_num = self._get_section_number(raw_text)
                    if section_num:
                        try:
                            num = int(section_num)
                            if num > 6 or num < 3:
                                in_target_section = False
                                current_main_section = ''
                        except:
                            pass
                    print(f"  ⏭️ H2: {heading_text} (건너뜀)")
                    current_h2 = None
                    current_h3 = None
                    current_h4 = None
                    current_h5 = None
                
            elif elem.name == 'h3' and in_target_section:
                heading_text = self._extract_heading_text(elem)
                raw_text = elem.get_text(strip=True)
                
                # 섹션 콘텐츠 추출
                content = self._get_section_content(elem)
                
                current_h3 = {
                    'name': heading_text,
                    'section_number': raw_text.split()[0] if raw_text else '',
                    'content': content,
                    'sub_sections': []
                }
                if current_h2:
                    current_h2['sub_sections'].append(current_h3)
                current_h4 = None
                current_h5 = None
                print(f"    📂 H3: {heading_text} (텍스트:{len(content['text'])}, 리스트:{len(content['list_items'])}, 테이블:{len(content['tables'])})")
                
            elif elem.name == 'h4' and in_target_section:
                heading_text = self._extract_heading_text(elem)
                
                # 섹션 콘텐츠 추출
                content = self._get_section_content(elem)
                
                current_h4 = {
                    'name': heading_text,
                    'content': content,
                    'sub_sections': []
                }
                if current_h3:
                    current_h3['sub_sections'].append(current_h4)
                elif current_h2:
                    current_h2['sub_sections'].append(current_h4)
                current_h5 = None
                print(f"      📄 H4: {heading_text} (텍스트:{len(content['text'])}, 리스트:{len(content['list_items'])})")
                
            elif elem.name == 'h5' and in_target_section:
                heading_text = self._extract_heading_text(elem)
                
                # 섹션 콘텐츠 추출
                content = self._get_section_content(elem)
                
                current_h5 = {
                    'name': heading_text,
                    'content': content,
                    'sub_sections': []
                }
                if current_h4:
                    current_h4['sub_sections'].append(current_h5)
                elif current_h3:
                    current_h3['sub_sections'].append(current_h5)
                print(f"        🏷️ H5: {heading_text}")
                
            elif elem.name == 'h6' and in_target_section:
                heading_text = self._extract_heading_text(elem)
                
                # 섹션 콘텐츠 추출
                content = self._get_section_content(elem)
                
                current_h6 = {
                    'name': heading_text,
                    'content': content
                }
                if current_h5:
                    current_h5['sub_sections'].append(current_h6)
                elif current_h4:
                    current_h4['sub_sections'].append(current_h6)
                print(f"          📌 H6: {heading_text}")
        
        print(f"\n✅ 파싱 완료: {len(self.results)}개 섹션 (3,4,5,6번만)")
        return self.results
    
    # ========== 저장 ==========
    
    def save_to_json(self, filename: str = None):
        """결과를 JSON 파일로 저장"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'kt_membership_{timestamp}.json'
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 저장 완료: {filename}")
        return filename
    
    def print_summary(self):
        """결과 요약 출력"""
        if not self.results:
            print("⚠️ 수집된 데이터가 없습니다.")
            return
        
        print("\n" + "="*60)
        print("📊 KT 멤버십 정보 요약 (3,4,5,6번 섹션)")
        print("="*60)
        
        def print_content_summary(content, indent=""):
            """콘텐츠 요약 출력"""
            if not content:
                return
            
            # 텍스트
            if content.get('text'):
                print(f"{indent}📝 텍스트: {len(content['text'])}개 단락")
                for t in content['text'][:2]:  # 처음 2개만
                    preview = t[:50] + '...' if len(t) > 50 else t
                    print(f"{indent}   • {preview}")
            
            # 리스트
            if content.get('list_items'):
                print(f"{indent}📋 리스트: {len(content['list_items'])}개 항목")
                for item in content['list_items'][:3]:  # 처음 3개만
                    preview = item[:60] + '...' if len(item) > 60 else item
                    print(f"{indent}   • {preview}")
            
            # 테이블
            if content.get('tables'):
                print(f"{indent}📊 테이블: {len(content['tables'])}개")
        
        for section in self.results:
            section_name = section.get('section', '(섹션)')
            section_num = section.get('section_number', '')
            print(f"\n{'='*50}")
            print(f"📁 {section_num} {section_name}")
            print(f"{'='*50}")
            
            # H2 콘텐츠
            if section.get('content'):
                print_content_summary(section['content'], "  ")
            
            # 하위 섹션들 (H3)
            for sub in section.get('sub_sections', []):
                sub_name = sub.get('name', '(하위)')
                sub_num = sub.get('section_number', '')
                print(f"\n  📂 {sub_num} {sub_name}")
                
                if sub.get('content'):
                    print_content_summary(sub['content'], "    ")
                
                # H4 하위
                for sub2 in sub.get('sub_sections', []):
                    sub2_name = sub2.get('name', '')
                    print(f"\n    📄 {sub2_name}")
                    if sub2.get('content'):
                        print_content_summary(sub2['content'], "      ")


def main():
    """메인 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description='나무위키 KT 멤버십 크롤러')
    parser.add_argument('--file', '-f', type=str, help='로컬 HTML 파일 경로')
    parser.add_argument('--url', '-u', type=str, help='크롤링할 URL')
    parser.add_argument('--output', '-o', type=str, help='출력 JSON 파일명')
    parser.add_argument('--headless', action='store_true', help='Headless 모드 (브라우저 창 숨김)')
    
    args = parser.parse_args()
    
    crawler = NamuKTMembershipCrawler()
    
    try:
        # 데이터 로드
        if args.file:
            crawler.load_from_file(args.file)
        else:
            # 기본: 브라우저 창 표시 (봇 탐지 우회에 유리)
            crawler.load_from_url(args.url, headless=args.headless)
        
        # 파싱
        crawler.parse_membership_content()
        
        # 요약 출력
        crawler.print_summary()
        
        # 저장
        output_file = crawler.save_to_json(args.output)
        print(f"\n✅ 완료! 결과: {output_file}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        crawler.close()


if __name__ == '__main__':
    main()