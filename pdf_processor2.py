# pdf_processor2.py
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
"""
print('기본 package import중...')
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

print('unstructures, langchain package import중...')
import nltk
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

from unstructured.partition.pdf import partition_pdf
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

print('패키지 import 완료')

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

def process_pdf_to_vectordb(file_path: str) -> str:
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
    _extract_with_unstructured(file_path, fig_dir)

    # Step 2 ─ PyMuPDF
    print('[Step 2] 페이지 이미지/텍스트 추출 중...')
    _pdf_to_page_images(file_path, page_image_dir)
    _pdf_to_page_texts(file_path, page_text_dir)

    # 이미지-텍스트 쌍 구성
    pairs = _pair_image_and_text(page_image_dir, page_text_dir)

    # Step 3 ─ LLM 페이지 정제
    print('[Step 3] LLM 페이지 정제 중...')
    refined_pages = _refine_pages_with_llm(client, pairs)

    # Step 4 ─ QA 합성 데이터 생성
    print('[Step 4] QA 합성 데이터 생성 중...')
    qa_result = _generate_qa(client, refined_pages, pairs, qa_jsonl_path)
    with open(qa_json_path, 'w', encoding='utf-8') as f:
        json.dump(qa_result, f, ensure_ascii=False)
    print(f'  QA 항목 수: {len(qa_result)}')

    # Step 5 ─ 이미지 설명 생성
    print('[Step 5] 이미지 설명 생성 중...')
    image_result = _describe_figures_with_llm(client, fig_dir, page_image_dir)
    with open(image_json_path, 'w', encoding='utf-8') as f:
        json.dump(image_result, f, ensure_ascii=False)
    print(f'  이미지 설명 수: {len(image_result)}')

    # Step 6 ─ VectorDB 저장
    print('[Step 6] Chroma VectorDB 저장 중...')
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


process_pdf_to_vectordb(r"C:\Users\Hane\PycharmProjects\Research_Report_Agent_temp_pycharm\uploads\20260320_OCI홀딩스 (010060_매수).pdf")

