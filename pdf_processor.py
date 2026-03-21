# pdf_processor.py
import os
import uuid
from pypdf import PdfReader

def process_pdf_to_file(file_path: str) -> str:
    """
    PDF에서 텍스트를 추출해 텍스트 파일(.txt)로 저장하고, 그 경로를 반환합니다.
    (클라우드에 파일을 올리고 URL을 받아오는 과정을 흉내 냅니다.)
    """
    # 결과물을 저장할 폴더 생성 (클라우드의 버킷 역할)
    os.makedirs("outputs", exist_ok=True)

    extracted_text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"

        # 고유한 파일명 생성 (예: result_a1b2c3d4.txt)
        unique_filename = f"result_{uuid.uuid4().hex[:8]}.txt"
        result_filepath = f"outputs/{unique_filename}"

        # 추출한 텍스트를 파일로 저장
        with open(result_filepath, "w", encoding="utf-8") as f:
            f.write(extracted_text)

        # 🌟 파일 전체 데이터가 아닌, "파일이 저장된 경로"만 반환
        return result_filepath

    except Exception as e:
        raise Exception(f"PDF 처리 중 오류 발생: {str(e)}")