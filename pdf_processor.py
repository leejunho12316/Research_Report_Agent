# pdf_processor.py
"""
멀티모달 RAG 데이터 처리 파이프라인

PDF 파일 경로를 입력받아 아래 단계를 순서대로 실행하고
Chroma VectorDB persist 경로를 반환합니다.

[Step 1] Unstructured  : PDF → 텍스트/테이블/figure 이미지 추출
[Step 2] PyMuPDF(fitz) : PDF → 페이지별 PNG 이미지 + TXT 추출
[Step 3] LLM           : 페이지 이미지+텍스트 → 정제된 텍스트 (gpt-4.1)
[Step 4] LLM           : 연속 페이지 쌍 → QA 합성 데이터 생성
[Step 5] LLM           : figure 이미지 → 맥락 기반 설명 생성
[Step 6] Chroma        : QA 데이터 + 이미지 설명 → VectorDB 저장

모든 중간 파일은 /data/<pdf파일명>/ 하위에 저장됩니다.
  /data/<name>/fig/            Unstructured 추출 이미지
  /data/<name>/pdf_to_image/   페이지별 PNG
  /data/<name>/pdf_to_text/    페이지별 TXT
  /data/<name>/QA_result.jsonl LLM QA 원본 (스트리밍 저장)
  /data/<name>/QA_result.json  파싱된 QA 리스트
  /data/<name>/IMAGE_result.json 이미지 설명 리스트
  /data/<name>/vectordb/       Chroma DB persist 디렉터리

Unstructured 사용 위한 Tesseract 등 설치 코드

Linux
!sudo apt install tesseract-ocr
!sudo apt install libtesseract-dev
!sudo apt-get install poppler-utils

Windows
tesseract
https://github.com/UB-Mannheim/tesseract/wiki
C:\Program Files\Tesseract-OCR PATH 추가

poppler
https://github.com/oschwartz10612/poppler-windows/releases
bin/ PATH 추가

!pip install -U "unstructured[all-docs]" lxml pillow==9.5.0 pdf2image==1.16.3 layoutparser[layoutmodels,tesseract]==0.3.4
"""

import os
import re
import io
import json
import base64
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import openai
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv() # .env 파일 읽어서 환경변수로 등록

import nltk
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from unstructured.partition.pdf import partition_pdf #시간 제일 많이 걸림

# ---------------------------------------------------------------------------
# 내부 유틸리티
# ---------------------------------------------------------------------------

def _data_dir(pdf_path: str) -> str:
    """반환: /data/<pdf파일명(확장자 제외)>/  (없으면 생성)"""
    stem = Path(pdf_path).stem
    path = os.path.join('/data', stem)
    os.makedirs(path, exist_ok=True)
    return path


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def _optimize_png(img: Image.Image) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True, compress_level=6)
    buf.seek(0)
    return Image.open(buf)


# ---------------------------------------------------------------------------
# Step 1: Unstructured — figure 이미지 / 텍스트 / 테이블 추출
# ---------------------------------------------------------------------------

def _extract_with_unstructured(pdf_path: str, fig_dir: str):
    os.makedirs(fig_dir, exist_ok=True)
    raw_pdf_elements = partition_pdf(
        filename=pdf_path,
        extract_images_in_pdf=True,
        skip_infer_table_types=False,
        chunking_strategy="by_title",
        max_characters=2000,
        new_after_n_chars=2000,
        combine_text_under_n_chars=2000,
        extract_image_block_output_dir=fig_dir,
    )

    tables, texts = [], []
    for chunk in raw_pdf_elements:
        if "CompositeElement" in str(type(chunk)):
            for element in chunk.metadata.orig_elements:
                if "Table" in str(type(element)):
                    tables.append(element)
            texts.append(chunk)

    print(f'  추출된 테이블: {len(tables)}개 / 텍스트 청크: {len(texts)}개')
    return texts, tables


# ---------------------------------------------------------------------------
# Step 2: PyMuPDF — 페이지별 PNG 이미지 + TXT 추출
# ---------------------------------------------------------------------------

def _pdf_to_page_images(pdf_path: str, output_dir: str, dpi: int = 300):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        zoom = dpi / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        _optimize_png(img).save(os.path.join(output_dir, f'page_{page_num + 1}.png'))
    total = len(doc)
    doc.close()
    print(f'  페이지 이미지 {total}개 저장: {output_dir}')


def _pdf_to_page_texts(pdf_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    with fitz.open(pdf_path) as doc:
        total = doc.page_count
        for idx, page in enumerate(doc, start=1):
            filepath = os.path.join(output_dir, f'page_{idx}.txt')
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(page.get_text())
    print(f'  페이지 텍스트 {total}개 저장: {output_dir}')


# ---------------------------------------------------------------------------
# Step 3: 페이지 이미지+텍스트 쌍 만들기
# ---------------------------------------------------------------------------

def _pair_image_and_text(image_dir: str, text_dir: str) -> list:
    images = os.listdir(image_dir) if os.path.isdir(image_dir) else []
    texts  = os.listdir(text_dir)  if os.path.isdir(text_dir)  else []
    img_dict = {os.path.splitext(f)[0]: os.path.join(image_dir, f) for f in images if f.endswith('.png')}
    txt_dict = {os.path.splitext(f)[0]: os.path.join(text_dir,  f) for f in texts  if f.endswith('.txt')}
    all_bases = set(img_dict).union(set(txt_dict))
    return [(img_dict.get(b, 'no png'), txt_dict.get(b, 'no txt')) for b in sorted(all_bases)]


# ---------------------------------------------------------------------------
# LLM 프롬프트 상수
# ---------------------------------------------------------------------------

_NO_TXT_PROMPT = '''당신이 해석할 이미지는 리포트입니다.
1. 중요한 내용이므로 요약하지말고 문법에 신경쓰면서 보이는 그대로 작성해주세요.
2. 내용을 임의로 바꾸지 마세요. 그리고 보이는 모든 내용을 다 적으십시오.
3. 단, 테이블은 풀어서 평문 또는 나열식으로 작성해주세요. 이미지에 없는 말은 적지마세요.
4. 테이블 풀어서 평문 또는 나열식으로 작성할 때 다른 행과 열이랑 헷갈리지 않게 값마다 잘 구분해서 적어주세요.
5. 테이블 해석할 때 통합셀들이 존재하니 구조를 잘 해석해서 작성해주시기 바랍니다. 어떤 게 어떤 것의 하위 내용인지를 명확히 하십시오
6. 당신의 의견은 궁금하지 않습니다. 해드렸습니다. 완성했습니다. 이런 표현도 적지마십시오. 이미지에 있는 내용만 적으십시오.
7. 만약 다단으로 구성되어져 있다면 좌측 테이블부터 먼저 작성하고 우측 테이블을 작성하십시오.
8. 수식은 반드시 마크다운으로 전부 다 작성하십시오. 가장 중요한 지시사항입니다.

자, 당신이 모든 내용을 빠트리지 않으면서 테이블은 구조를 잘 해석해서 작성해주는 것을 믿습니다.
'''

_WITH_TXT_PROMPT = '''당신이 해석할 이미지는 리포트입니다.
1. 중요한 내용이므로 요약하지말고 문법에 신경쓰면서 보이는 그대로 작성해주세요.
2. 내용을 임의로 바꾸지 마세요. 그리고 보이는 모든 내용을 다 적으십시오.
3. 단, 테이블은 풀어서 평문 또는 나열식으로 작성해주세요. 이미지에 없는 말은 적지마세요.
4. 테이블 풀어서 평문 또는 나열식으로 작성할 때 다른 행과 열이랑 헷갈리지 않게 값마다 잘 구분해서 적어주세요.
5. 테이블 해석할 때 통합셀들이 존재하니 구조를 잘 해석해서 작성해주시기 바랍니다. 어떤 게 어떤 것의 하위 내용인지를 명확히 하십시오
6. 당신의 의견은 궁금하지 않습니다. 해드렸습니다. 완성했습니다. 이런 표현도 적지마십시오. 이미지에 있는 내용만 적으십시오.
7. 만약 다단으로 구성되어져 있다면 좌측 테이블부터 먼저 작성하고 우측 테이블을 작성하십시오.
8. 수식은 반드시 마크다운으로 전부 다 작성하십시오. 가장 중요한 지시사항입니다.
9. 당신에게 당신이 해석할 파일을 txt로 변경한 내용도 드리겠습니다. 페이지 해석할 때 참고하세요.
10. txt에 있는 텍스트는 반드시 해당 페이지에 존재한다는 겁니다. txt에 있는 텍스트를 빠트리지 마십시오.

자 당신이 헷갈리지 않도록 txt도 드렸습니다. 이미지를 더 잘 해석할 거라 믿습니다.
'''

_QA_SYSTEM_PROMPT = '''당신은 주어진 2개의 문서로부터 사용자의 가능한 질문과 답변의 쌍 5개를 생성해야 합니다.

1. 2개의 문서는 서로 이어지는 문서이므로 문맥이 순차적으로 이어지고 있음을 감안하세요.
2. content는 본문이고 source는 출처입니다.
3. 주어진 2개의 문서로부터 가능한 질문과 답변의 쌍을 5개 생성하세요.
4. Q1과 A1, Q2와 A2, Q3과 A3, Q4과 A4, Q5과 A5 이렇게 작성하면됩니다.
5. 각 답변의 뒤에는 출처 문서들을 리스트 형태로 작성하세요. 다수의 문서를 동시에 참고할 수 있습니다.
6. 주어진 문서에 없는 내용은 작성하지 마십시오. 오직 'content' 안에 있는 내용만 답변할 수 있습니다.
7. 답변은 오직 'content' 안에 있는 내용으로만 답변하지만 최대한 풍부하고 길게 작성하세요. 이는 매우 중요합니다.
8. 다수의 문서를 동시에 참고하여 답변하는 것이 가능합니다.
9. 답변할 때 문서에 해당 항목이 약관의 몇 조인지 명확한 상황이라면 답변할 때 무조건 이를 언급하면 좋겠습니다.
10. 너무 억지스럽게 지엽적인 질문은 금지하겠습니다. 일반 사용자가 할만한 질문을 던지십시오.
11. 주어진 문서에서 예시, 표, 마크다운 수식, 글머리 기호로 표시된 것들을 FAQ로 만드는 것을 주저하지 마십시오.
12. 마크다운 수식 또한 중요하게 취급하여 답변에 포함하십시오.
'''

#토큰 아끼기용 보류.
# '''
# 예시)
# 입력:
# 'content' : '좌측 테이블(요약 정보, 투자 지표)부터 작성:  - 투자의견(유지): 매수 - 목표주가(상향): 88,000원 - 현재주가(26/3/19): 71,800원 - 상승여력: 22.6% - 영업이익(26F, 십억원): 1,316 - Consensus 영업이익(26F, 십억원): 1,340 - EPS 성장률(26F, %): 37.2 - MKT EPS 성장률(26F, %): 118.0 - P/E(26F, x): 15.0 - MKT P/E(26F, x): 9.2 - KOSPI: 5,763.22 - 시가총액(십억원): 12,631 - 발행주식수(백만주): 176 - 유동주식비율(%): 26.1 - 외국인 보유비중(%): 6.7 - 베타(12M) 일간수익률: 0.54 - 52주 최저가(원): 44,000 - 52주 최고가(원): 79,500  - 1M 절대주가: 3.8% - 1M 상대주가: 2.2% - 6M 절대주가: 46.4% - 6M 상대주가: -12.5% - 12M 절대주가: 26.9% - 12M 상대주가: -42.1%  [운송플랫폼/에너지]   류제현   jay.ryu@miraeasset.com  ---  우측 본문 및 하단 테이블 작성:  047050 · 에너지   포스코인터내셔널   단단한 펀더멘털, 축소될 밸류에이션 갭    유가 말고도 풍부한 투자포인트: 에너지·모빌리티·희토류   최근 동사의 주가는 유가 상승과 맞물려 주목받고 있으나, 장기적인 펀더멘털 역시 개선되고 있음에 주목한다. 에너지 사업은 상류(미얀마·세넥스), 미드스트림(터미널), 다운스트림(발전) 확대를 통해 ROE 기여도를 크게 확대할 전망이다.  모빌리티 사업은 구동모터코아(2030년 매출 1조 4,900억원)를 중심으로 장기 계획이 점차 명확해지고 있다. EV 캐즘과 중국 경쟁 리스크에도 불구, 동사는 포스코 그룹 전기강판(2030년 매출 1조 2,500억원) 연계와 탈중국 희토류 전략으로 효과적으로 대응할 전망이다. 영구 자석 사업은 동남아(2026년 3천 톤)와 미국 생산(2030년 5천 톤)으로 미국·EU OEM과 2030년 2,530톤 공급계약을 이미 확보하거나 추가 성약 협의를 진행 중이다.  지속 성장을 바탕으로 ROE 개선과 주주환원 확대 기대   1Q25 동사의 영업이익은 3,102억원을 기록할 것으로 기대한다. 상류 중심(1,227억원)을 중심으로 에너지 부문 영업이익(1,797억원)이 이익 개선을 주도할 전망이다. 지난해 인수했던 인니팜으로부터의 이익 추가도 본격화될 것으로 기대된다.  동사에 대한 목표주가 상사 업체 주가 2025~2027년 영업활동 현금흐름 5조 원과 비핵심 자산 매각(0.2조원)을 바탕으로 성장투자 3.2조 원, 주주환원 0.9~1.1조 원을 병행을 계획하고 있다. 차입금도 낮출 전망이다. 이를 바탕으로 ROE는 2025년 8%에서 2027년 15%로 상승할 것으로 목표하고 있다.  목표주가 88,000원으로 상향하며 매수 의견 유지   동사에 대한 목표주가를 73,000원에서 88,000원(상사 부문 EV/EBITDA 15배에 자원개발 가치 합산)으로 상향하며 매수의견을 유지한다. 일본상사의 밸류에이션(EV/EBITDA 20배)이 지난해 이후 크게 상승한바 있다. 이에 따라 동사의 밸류에이션(EV/EBITDA 10배)은 일본 상사업체 대비 52% 저평가되어 있는 상태이다. 할인율은 5년 평균(44%)대비로도 높은 수준이다. 동사의 장기 성장성이 확인되면서 재차 밸류에이션갭 축소를 기대할 수 있다. 최근 상승한 유가와 환율이 단기 실적에 분명 도움이 되겠으나 장기적으로 보더라도 주가 추가 반등의 이유는 충분하다.  ---  결산기(12월)   2024년   매출액: 32,261십억원   영업이익: 1,158십억원   영업이익률: 3.6%   순이익: 515십억원   EPS: 2,925원   ROE: 8.1%   P/E: 13.6배   P/B: 1.1배   배당수익률: 3.9%  2025년   매출액: 32,374십억원   영업이익: 1,165십억원   영업이익률: 3.6%   순이익: 614십억원   EPS: 3,491원   ROE: 9.3%   P/E: 14.2배   P/B: 1.3배   배당수익률: 3.7%  2026년(예상, F)   매출액: 33,694십억원   영업이익: 1,316십억원   영업이익률: 3.9%   순이익: 843십억원   EPS: 4,789원   ROE: 12.1%   P/E: 15.0배   P/B: 1.7배   배당수익률: 3.6%  2027년(예상, F)   매출액: 34,632십억원   영업이익: 1,411십억원   영업이익률: 4.1%   순이익: 842십억원   EPS: 4,786원   ROE: 11.3%   P/E: 15.0배   P/B: 1.7배   배당수익률: 3.6%  2028년(예상, F)   매출액: 35,645십억원   영업이익: 1,613십억원   영업이익률: 4.5%   순이익: 987십억원   EPS: 5,609원   ROE: 12.5%   P/E: 12.8배   P/B: 1.5배   배당수익률: 3.6%  주: K-IFRS 연결 기준, 순이익은 지배주주 귀속 순이익   자료: 포스코인터내셔널, 미래에셋증권 리서치센터  ---  차트:   포스코인터내셔널 주가와 KOSPI 지수 변화 표시는 25.3, 25.7, 25.11, 26.3 구간에서 표시되어 있음   (이미지의 시계열차트는 텍스트로 해석이 어려운 관계로 수치 표기만 나열)  ---  수식 표기:   - 상승여력: \\( \\frac{목표주가 - 현재주가}{현재주가} \\times 100 = 22.6\\% \\) - EPS 성장률(26F, %): 37.2% - P/E(26F, x): 15.0 - EV/EBITDA(동사): 10배 - EV/EBITDA(일본상사): 20배 - 할인율: 5년 평균(44%) - ROE (2025): 8% - ROE (2027): 15%', 'source' : 'page_1.png'
# 'content' : '포스코인터내셔널   2026.3.20  투자의견 및 목표주가 변동추이  제시일자, 투자의견, 목표주가(원), 괴리율(%), 평균주가대비, 최고(최저)주가대비(포스코인터내셔널 047050)  2026.03.20   투자의견: 매수   목표주가(원): 88,000   괴리율(%): -   평균주가대비: -   최고(최저)주가대비: -    2025.10.27   투자의견: 매수   목표주가(원): 73,000   괴리율(%): -18.65   평균주가대비: 8.90    2025.04.24   투자의견: 매수   목표주가(원): 59,000   괴리율(%): -15.37   평균주가대비: 1.36    2025.02.04   투자의견: 매수   목표주가(원): 55,000   괴리율(%): -8.33   평균주가대비: 10.91    2024.07.26   투자의견: 매수   목표주가(원): 65,000   괴리율(%): -24.98   평균주가대비: -10.00    2024.04.26   투자의견: 매수   목표주가(원): 60,000   괴리율(%): -11.03   평균주가대비: 14.00    2023.10.31   투자의견: 매수   목표주가(원): 65,000   괴리율(%): -15.78   평균주가대비: 0.31    * 괴리율 산정: 수정주가 적용, 목표주가 대상시점은 1년이며 목표주가를 변경하는 경우 해당 조사분석자료의 공표일 전일까지 기간을 대상으로 함  투자의견 분류 및 적용기준  기업   매수: 향후 12개월 기준 절대수익률 20% 이상의 초과수익 예상   중립: 향후 12개월 기준 절대수익률 -10~10% 이내의 등락이 예상   매도: 향후 12개월 기준 절대수익률 -10% 이상의 주가하락이 예상    산업   비중확대: 향후 12개월 기준 업종지수상승률이 시장수익률 대비 높거나 상승   중립: 향후 12개월 기준 업종지수상승률이 시장수익률 수준   비중축소: 향후 12개월 기준 업종지수상승률이 시장수익률 대비 낮거나 악화    매수(▲), Trading Buy(■), 중립(●), 매도(◆), 주가(─), 목표주가(▬), Not covered(■)  * 2025년 5월 12일 기준으로 투자의견 분류기준 변경(Trading Buy 의견 삭제) * 향후 12개월 기준 절대수익률 10% 이상, 20% 미만의 주가상승이 예상되는 종목은 금융투자분석사 재량에 따라 ‘매수’ 또는 ‘중립’ 의견으로 제시함  투자의견 비율  매수(매수): 79.76%   Trading Buy(매수): 1.19%   중립(중립): 19.05%   매도: 0%    * 2025년 12월 31일 기준으로 최근 1년간 금융투자상품에 대하여 공표한 최근일 투자등급의 비율  Compliance Notice  - 당사는 자료 작성일 현재 조사분석 대상법인과 관련하여 특별한 이해관계가 없음을 확인합니다. - 당사는 본 자료를 제3자에게 사전 제공한 사실이 없습니다. - 본 자료를 작성한 애널리스트는 자료작성일 현재 조사분석 대상법인의 금융투자상품 및 권리를 보유하고 있지 않습니다. - 본 자료는 외부의 부당한 압력이나 간섭없이 애널리스트의 의견이 정확하게 반영되었음을 확인합니다.  본 조사분석자료는 당사의 리서치센터가 신뢰할 수 있는 자료 및 정보로부터 얻은 것이나, 당사가 그 정확성이나 완전성을 보장할 수 없으므로 투자자 자신의 판단과 책임하에 종목 선택이나 투자시기에 대한 최종 결정을 하시기 바랍니다. 따라서 본 조사분석자료는 어떠한 경우에도 고객의 증권투자 결과에 대한 법적 책임소재의 증빙자료로 사용될 수 없습니다. 본 조사분석자료의 지적재산권은 당사에 있으므로 당사의 허락 없이 무단 복제 및 배포할 수 없습니다.  (원) 포스코인터내셔널   0   20,000   40,000   60,000   80,000   100,000   24.3   25.3   26.3  Mirae Asset Securities Research', 'source' : 'page_10.png'
#
# 답변:
# Q1: 미래에셋증권 리서치센터에서 포스코인터내셔널의 목표주가를 상향한 이유와 그에 따른 상승여력은 어떻게 계산되었나요?
# A1: 미래에셋증권은 포스코인터내셔널의 단단한 펀더멘털 개선과 장기 성장성을 바탕으로 목표주가를 기존 73,000원에서 88,000원으로 상향 조정하며 '매수' 의견을 유지했습니다.가장 큰 상향 이유는 밸류에이션 갭 축소에 대한 기대감입니다. 일본 상사 업체들의 밸류에이션(EV/EBITDA 20배)이 지난해 이후 크게 상승한 반면, 동사의 밸류에이션은 EV/EBITDA 10배 수준으로 일본 업체 대비 52%나 저평가되어 있는 상태입니다. 미래에셋증권은 이러한 저평가 상태가 해소될 것으로 보고, 상사 부문에 EV/EBITDA 15배를 적용하고 자원개발 가치를 합산하여 새로운 목표주가를 산출했습니다. 또한 할인율 역시 5년 평균치인 44% 대비 높은 수준으로 평가되었습니다.현재 주가(2026년 3월 19일 기준 71,800원) 대비 상향된 목표주가(88,000원)를 바탕으로 한 상승여력은 보고서의 수식 표기에 따라 다음과 같이 산출됩니다.$\frac{88,000 - 71,800}{71,800} \times 100 = 22.6\%$결론적으로 최근 유가와 환율 상승이 단기 실적을 견인하는 것은 물론, 장기적인 주가 추가 반등의 근거가 충분하다는 것이 리서치센터의 분석입니다.
# 출처: ['page_1.png']
#
#
# Q2: 포스코인터내셔널의 장기적인 재무 실적 전망과 주주들을 위한 자본 배분 및 환원 계획은 어떻게 되나요?
# A2: 포스코인터내셔널은 견조한 실적 성장을 바탕으로 공격적인 투자와 주주환원을 병행할 계획입니다.
#
# 재무 실적 전망 (K-IFRS 연결 기준):
# 2024년 (실적): 매출액 32조 2,610억 원, 영업이익 1조 1,580억 원 (영업이익률 3.6%), 순이익 5,150억 원, EPS 2,925원
# 2025년 (예상): 매출액 32조 3,740억 원, 영업이익 1조 1,650억 원, 순이익 6,140억 원, EPS 3,491원
# 2026년 (예상): 매출액 33조 6,940억 원, 영업이익 1조 3,160억 원, 순이익 8,430억 원, EPS 4,789원
# 2027년 (예상): 매출액 34조 6,320억 원, 영업이익 1조 4,110억 원, 순이익 8,420억 원, EPS 4,786원
#
# 자본 배분 및 주주환원 계획:
# 회사는 2025년부터 2027년까지 약 5조 원의 영업활동 현금흐름을 창출하고, 비핵심 자산 매각으로 2,000억 원을 추가 확보할 계획입니다. 이를 재원으로 삼아 성장 투자에 3.2조 원을 투입하고, 주주환원에 9,000억 원에서 1조 1,000억 원을 배정하여 병행 추진할 예정입니다. 동시에 차입금 비율도 낮출 계획이며, 이러한 선순환을 통해 2025년 8% (세부 표 기준 9.3% 예상) 수준의 ROE를 2027년에는 15% (세부 표 기준 11.3% 예상치 상회 목표)까지 끌어올리는 것을 최종 목표로 삼고 있습니다.
# 출처: ['page_1.png']
#
#
# Q3: 보고서에 적용된 기업 투자의견(매수, 중립, 매도)의 정확한 분류 기준은 무엇이며, 최근 기준 산정에 있어 변경된 규정이 있나요?
# A3: 보고서의 **'투자의견 분류 및 적용기준'**에 따르면 기업에 대한 투자의견은 향후 12개월 기준의 예상 절대수익률을 바탕으로 세 가지로 엄격하게 분류됩니다.
#
# 매수 (Buy): 향후 12개월 기준 절대수익률이 20% 이상의 초과 수익을 거둘 것으로 예상될 때 적용됩니다.
# 중립 (Hold): 향후 12개월 기준 절대수익률이 -10%에서 10% 이내의 등락을 보일 것으로 예상될 때 적용됩니다.
# 매도 (Sell): 향후 12개월 기준 절대수익률이 -10% 이상의 주가 하락을 겪을 것으로 예상될 때 적용됩니다.
#
# 또한, 보고서의 **'2025년 5월 12일 기준으로 변경된 투자의견 분류기준'**에 따르면 중요한 변경 사항과 예외 조항이 존재합니다.
# 해당 일자를 기점으로 기존에 존재하던 'Trading Buy' 의견이 완전히 삭제되었습니다. 더불어, 향후 12개월 기준 절대수익률이 10% 이상, 20% 미만으로 예상되는 애매한 구간의 종목에 대해서는 일괄적인 기준을 적용하지 않고 금융투자분석사(애널리스트)의 재량적 판단에 따라 '매수' 또는 '중립' 의견으로 유연하게 제시할 수 있도록 규정하고 있습니다.
# 출처: ['page_10.png']
#
#
# ...생략...
#
#
# 시작!
#
# '''

_IMAGE_SYSTEM_PROMPT = '''당신은 더 작은 이미지가 어떤 맥락에서 나왔는지와 해당 이미지에 대한 상세한 설명을 해야 합니다.

1. 당신에게는 PDF 파일의 전체를 캡쳐한 이미지와 그 중 일부를 캡쳐한 이미지 두 개가 주어집니다.
2. 당신은 일부를 캡쳐한 이미지가 어떤 이미지인지 상세 설명을 해야 합니다.
3. 당신에게 전체를 캡쳐해서 드리는 이유는 해당 이미지가 어떤 맥락에서 나왔는지를 알려주기 위함입니다.
4. 전체 이미지에서의 맥락과 일부 캡쳐한 이미지의 정보를 조합하여 일부 캡쳐한 이미지에 대한 상세 설명을 전개하십시오.
5. 이 이미지에 대한 설명은 이미지와 함께 실제로 사용자의 질문에 대한 검색 결과로 주어지게 됩니다.
6. 따라서 검색 결과에 잘 나오도록 맥락과 해당 이미지가 설명하고 있는 바를 텍스트로 작성하십시오.
7. 일부 이미지에 대한 정보가 손실되어서는 안 됩니다.
8. 장황하게 설명하지는 마십시오.
9. 수식은 마크다운을 유지하면서 작성하십시오.'''


# ---------------------------------------------------------------------------
# Step 3: LLM 페이지 정제
# ---------------------------------------------------------------------------

def _refine_pages_with_llm(client: openai.OpenAI, pairs: list) -> list:
    result = []
    for image_path, text_path in tqdm(pairs, desc='  페이지 LLM 정제'):
        b64_image = _encode_image(image_path)

        if text_path == 'no txt':
            prompt = _NO_TXT_PROMPT
        else:
            with open(text_path, 'r', encoding='utf-8') as f:
                txt_content = f.read()
            prompt = _WITH_TXT_PROMPT + txt_content + '\n시작!'

        response = client.chat.completions.create(
            model="gpt-4.1",
            max_tokens=2500,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                    {"type": "text", "text": prompt}
                ]
            }]
        )
        result.append(response.choices[0].message.content)
    return result


# ---------------------------------------------------------------------------
# Step 4: QA 합성 데이터 생성
# ---------------------------------------------------------------------------

def _generate_qa(client: openai.OpenAI, refined_pages: list, pairs: list, jsonl_path: str) -> list:
    user_prompt_list = []
    for i in range(len(refined_pages) - 1):
        prompt = (
            f"'content' : '{refined_pages[i]}', 'source' : '{os.path.basename(pairs[i][0])}'\n"
            f"'content' : '{refined_pages[i+1]}', 'source' : '{os.path.basename(pairs[i+1][0])}'"
        )
        user_prompt_list.append(prompt)

    qa_raw = []
    previous_qa = ''

    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for user_prompt in tqdm(user_prompt_list, desc='  QA 데이터 생성'):
            if previous_qa:
                user_prompt += (
                    '\n단 아래의 질문과 답변과 유사한 내용은 피하십시오.\n'
                    + previous_qa
                    + '\n위 내용과 거의 중복되고 유사한 질문 답변은 필요없습니다.'
                )

            completion = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": _QA_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt}
                ],
                temperature=0
            )
            text = completion.choices[0].message.content
            qa_raw.append(text)
            json.dump(text, f, ensure_ascii=False)
            f.write('\n')
            f.flush()
            previous_qa = text

    # Q/A 라벨을 통일하여 개별 항목으로 분리
    qa_result = []
    for data in qa_raw:
        for q in ('Q1:', 'Q2:', 'Q3:', 'Q4:', 'Q5:'):
            data = data.replace(q, '문의:')
        for a in ('A1:', 'A2:', 'A3:', 'A4:', 'A5:'):
            data = data.replace(a, '내용:')
        items = data.split('문의:')
        qa_result += ['문의:' + d.strip() for d in items if len(d.strip()) > 1]

    return qa_result


# ---------------------------------------------------------------------------
# Step 5: figure 이미지 → LLM 설명 생성
# ---------------------------------------------------------------------------

def _describe_figures_with_llm(client: openai.OpenAI, fig_dir: str, page_image_dir: str) -> list:
    figure_files = [
        f for f in os.listdir(fig_dir)
        if f.startswith('figure') and f.endswith('.jpg')
    ]

    pair_list = []
    for fig_file in sorted(figure_files):
        match = re.match(r'figure-(\d+)-\d+\.jpg', fig_file)
        if match:
            page_num = match.group(1)
            page_file = os.path.join(page_image_dir, f'page_{page_num}.png')
            if os.path.exists(page_file):
                pair_list.append((page_file, os.path.join(fig_dir, fig_file)))

    image_results = []
    for page_file, figure_file in tqdm(pair_list, desc='  이미지 설명 생성'):
        page_b64   = _encode_image(page_file)
        figure_b64 = _encode_image(figure_file)

        response = client.chat.completions.create(
            model="gpt-4.1",
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{page_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{figure_b64}"}},
                    {"type": "text", "text": _IMAGE_SYSTEM_PROMPT}
                ]
            }]
        )
        description = response.choices[0].message.content
        image_results.append(description + '\n출처: [' + figure_file + ']')

    return ['## 이미지 콘텐츠\n' + item for item in image_results]


# ---------------------------------------------------------------------------
# 공개 인터페이스
# ---------------------------------------------------------------------------

def _update_progress(task_id, msg: str):
    """task_id가 주어졌을 때 DB의 progress 컬럼을 업데이트한다."""
    if task_id is None:
        return
    import sqlite3
    conn = sqlite3.connect("app.db")
    conn.execute("UPDATE files SET progress = ? WHERE id = ?", (msg, task_id))
    conn.commit()
    conn.close()


def process_pdf_to_vectordb(file_path: str, task_id=None) -> str:
    """
    PDF 파일 경로를 받아 멀티모달 RAG 파이프라인을 실행합니다.

    Parameters
    ----------
    file_path : str
        처리할 PDF 파일의 경로

    Returns
    -------
    str
        Chroma VectorDB가 저장된 디렉터리 경로 (/data/<pdf이름>/vectordb)

    Notes
    -----
    OPENAI_API_KEY 환경변수가 설정되어 있어야 합니다.
    """
    data_dir       = _data_dir(file_path)
    fig_dir        = os.path.join(data_dir, 'fig')
    page_image_dir = os.path.join(data_dir, 'pdf_to_image')
    page_text_dir  = os.path.join(data_dir, 'pdf_to_text')
    qa_jsonl_path  = os.path.join(data_dir, 'QA_result.jsonl')
    qa_json_path   = os.path.join(data_dir, 'QA_result.json')
    image_json_path= os.path.join(data_dir, 'IMAGE_result.json')
    vectordb_dir   = os.path.join(data_dir, 'vectordb')

    client = openai.OpenAI()


    # Step 1 ─ Unstructured
    print('[Step 1] Unstructured 추출 중...')
    _update_progress(task_id, '[Step 1] Unstructured 추출 중...')
    _extract_with_unstructured(file_path, fig_dir)

    # Step 2 ─ PyMuPDF
    print('[Step 2] 페이지 이미지/텍스트 추출 중...')
    _update_progress(task_id, '[Step 2] 페이지 이미지/텍스트 추출 중...')
    _pdf_to_page_images(file_path, page_image_dir)
    _pdf_to_page_texts(file_path, page_text_dir)

    # 이미지-텍스트 쌍 구성
    pairs = _pair_image_and_text(page_image_dir, page_text_dir)

    # Step 3 ─ LLM 페이지 정제
    print('[Step 3] LLM 페이지 정제 중...')
    _update_progress(task_id, '[Step 3] LLM 페이지 정제 중...')
    refined_pages = _refine_pages_with_llm(client, pairs)

    # Step 4 ─ QA 합성 데이터 생성
    print('[Step 4] QA 합성 데이터 생성 중...')
    _update_progress(task_id, '[Step 4] QA 합성 데이터 생성 중...')
    qa_result = _generate_qa(client, refined_pages, pairs, qa_jsonl_path)
    with open(qa_json_path, 'w', encoding='utf-8') as f:
        json.dump(qa_result, f, ensure_ascii=False)
    print(f'  QA 항목 수: {len(qa_result)}')

    # Step 5 ─ 이미지 설명 생성
    print('[Step 5] 이미지 설명 생성 중...')
    _update_progress(task_id, '[Step 5] 이미지 설명 생성 중...')
    image_result = _describe_figures_with_llm(client, fig_dir, page_image_dir)
    with open(image_json_path, 'w', encoding='utf-8') as f:
        json.dump(image_result, f, ensure_ascii=False)
    print(f'  이미지 설명 수: {len(image_result)}')


    # 디버깅 - Step 5까지 진행했는데 Step6에서 막혀서 진행할 때
    with open(qa_json_path, 'r', encoding='utf-8') as f:
        qa_result = json.load(f)
    with open(image_json_path, 'r', encoding='utf-8') as f:
        image_result = json.load(f)
    print(f'  QA 항목 수: {len(qa_result)} / 이미지 설명 수: {len(image_result)}')

    # Step 6 ─ VectorDB 저장
    print('[Step 6] Chroma VectorDB 저장 중...')
    _update_progress(task_id, '[Step 6] Chroma VectorDB 저장 중...')
    total_docs = qa_result + image_result
    langchain_docs = [Document(page_content=doc) for doc in total_docs]

    vectordb = Chroma.from_documents(
        documents=langchain_docs,
        embedding=OpenAIEmbeddings(),
        collection_name="multimodal_rag",
        persist_directory=vectordb_dir,
    )

    count = vectordb._collection.count()
    print(f'[완료] VectorDB 저장 완료: {vectordb_dir}')
    print(f'  총 저장 문서 수: {count}')

    return vectordb_dir
