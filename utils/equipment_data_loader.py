"""
설비 마스터 데이터 로드 및 벡터화 유틸리티 모듈

이 모듈은 설비 마스터 데이터(Excel)를 로드하고, 이를 벡터 저장소에 저장하는
기능을 제공합니다. 설비 코드 검색 및 분석을 위한 벡터 데이터베이스를
구축하는 데 사용됩니다.

주요 기능:
- Excel 형식의 설비 마스터 데이터 로드 및 전처리
- 설비 정보를 텍스트로 변환하여 벡터화
- 벡터 저장소에 설비 데이터 저장 (메타데이터 포함)

사용 예시:
    python -m PROJECT_SKAI-NOTIASSISTANCE.utils.equipment_data_loader

참고:
- 데이터는 'PROJECT_SKAI-NOTIASSISTANCE/data/equipment_master/설비마스터.xlsx' 위치에 있어야 합니다.
- 필수 컬럼: ITEMNO, 설비명, 설비유형, 위치
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, List, Any

import pandas as pd
import numpy as np
from tqdm import tqdm

# 프로젝트 루트 디렉토리 설정
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# 프로젝트 루트를 Python 경로에 추가
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 프로젝트 내부 모듈 임포트
try:
    from lib.vector_store import VectorStore
    from utils.logger import get_logger
except ImportError as e:
    print(f"필수 모듈 임포트 실패: {e}")
    print(f"현재 Python 경로: {sys.path}")
    sys.exit(1)

# 로거 설정
logger = get_logger(__name__)

# 상수 정의
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "equipment_master" / "설비마스터.xlsx"
REQUIRED_COLUMNS = ['ITEMNO', '설비명', '설비유형', '위치']
VECTOR_COLLECTION_NAME = "equipment_codes"
BATCH_SIZE = 1000  # 벡터 저장 시 배치 크기


def load_equipment_master(
    file_path: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    설비 마스터 데이터를 로드하고 전처리합니다.

    Args:
        file_path (str, optional): 설비 마스터 Excel 파일 경로.
                                  기본값은 'data/equipment_master/설비마스터.xlsx'

    Returns:
        Optional[pd.DataFrame]: 전처리된 설비 마스터 데이터프레임.
                              오류 발생 시 None 반환.
    """
    # 파일 경로 설정 (Path 객체로 변환)
    data_path = Path(file_path) if file_path else DEFAULT_DATA_PATH
    logger.info(f"설비 마스터 데이터 로드 시작: {data_path}")

    # 파일 존재 여부 확인
    if not data_path.exists():
        logger.error(f"파일을 찾을 수 없습니다: {data_path}")
        return None

    try:
        # Excel 파일 로드
        df = pd.read_excel(data_path)
        logger.info(f"데이터 로드 완료: {len(df):,}개 행")

        # 필수 컬럼 확인
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            logger.error(f"필수 컬럼 누락: {missing_columns}")
            return None

        # 결측치 처리 및 로깅
        for col in REQUIRED_COLUMNS:
            na_count = df[col].isna().sum()
            if na_count > 0:
                logger.warning(f"'{col}' 컬럼에 {na_count:,}개의 결측치가 있습니다")
                df[col] = df[col].fillna("")

        return df

    except Exception as e:
        logger.error(f"데이터 로드 실패: {str(e)}", exc_info=True)
        return None


def create_equipment_vectors(
    df: pd.DataFrame,
    vector_store: VectorStore,
    collection_name: str = VECTOR_COLLECTION_NAME,
    batch_size: int = BATCH_SIZE
) -> bool:
    """
    설비 데이터를 벡터화하여 저장합니다.

    Args:
        df (pd.DataFrame): 설비 마스터 데이터프레임
        vector_store (VectorStore): 벡터 저장소 인스턴스
        collection_name (str): 저장할 컬렉션 이름 (기본값: "equipment_codes")
        batch_size (int): 한 번에 처리할 데이터 수 (기본값: 1000)

    Returns:
        bool: 성공 시 True, 실패 시 False
    """
    logger.info(f"설비 데이터 벡터화 시작 (총 {len(df):,}개 설비)")

    try:
        # 벡터 저장소 컬렉션 생성
        vector_store.create_collection(collection_name)
        logger.info(f"벡터 저장소 컬렉션 생성: {collection_name}")

        # 설비 데이터 텍스트화 및 메타데이터 구성
        texts: List[str] = []
        metadata_list: List[Dict[str, Any]] = []

        # tqdm으로 진행률 표시하며 데이터 처리
        for _, row in tqdm(df.iterrows(), total=len(df), desc="설비 데이터 처리"):
            # 검색용 텍스트 구성 (설비번호, 이름, 유형, 위치 조합)
            description = (f"{row['ITEMNO']} {row['설비명']} "
                         f"{row['설비유형']} {row['위치']}")

            # 메타데이터 구성 (상세 정보 저장)
            metadata = {
                "item_no": row['ITEMNO'],
                "name": row['설비명'],
                "type": row['설비유형'],
                "location": row['위치']
            }

            texts.append(description)
            metadata_list.append(metadata)

        # 배치 단위로 벡터 저장소에 추가
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for i in range(0, len(texts), batch_size):
            batch_end = min(i + batch_size, len(texts))
            batch_texts = texts[i:batch_end]
            batch_metadata = metadata_list[i:batch_end]

            vector_store.add_items(
                collection_name=collection_name,
                texts=batch_texts,
                metadata_list=batch_metadata
            )
            logger.info(f"벡터화 진행: {batch_end:,}/{len(texts):,} "
                       f"(배치 {(i//batch_size)+1}/{total_batches})")

        logger.info("설비 데이터 벡터화 완료")
        return True

    except Exception as e:
        logger.error(f"벡터화 중 오류 발생: {str(e)}", exc_info=True)
        return False


def main():
    """
    메인 실행 함수

    설비 마스터 데이터를 로드하고 벡터화하는 전체 과정을 실행합니다.
    """
    logger.info("설비 데이터 처리 시작")

    try:
        # 벡터 저장소 초기화
        vector_store = VectorStore()
        logger.info("벡터 저장소 초기화 완료")

        # 설비 마스터 데이터 로드
        df = load_equipment_master()
        if df is None:
            logger.error("설비 마스터 데이터 로드 실패")
            return

        # 벡터 저장소에 설비 데이터 추가
        success = create_equipment_vectors(df, vector_store)
        if success:
            logger.info("설비 데이터 처리 완료")
        else:
            logger.error("설비 데이터 벡터화 실패")

    except Exception as e:
        logger.error(f"처리 중 예상치 못한 오류 발생: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main() 