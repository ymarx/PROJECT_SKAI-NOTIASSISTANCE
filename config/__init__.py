"""
SKAI-NotiAssistance 설정 패키지

이 패키지는 프로젝트 실행에 필요한 모든 설정을 관리합니다.
주요 설정 파일은 다음과 같습니다:
- model_config.yaml: LLM, Embedding 모델, Vector Store 관련 설정
- prompt_templates.yaml: LLM에 전달할 프롬프트 템플릿 모음
- logging_config.yaml: 애플리케이션 로깅 방식 및 포맷 설정

이 __init__.py 파일은 설정 파일들을 쉽게 로드할 수 있는 헬퍼 함수를 제공합니다.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional

# 프로젝트 내부 유틸리티 함수 임포트
# utils 모듈이 상위 디렉토리에 있다고 가정
try:
    from lib.utils import load_yaml
    from utils.logger import get_logger # 로깅 사용
except ImportError:
    # 경로 문제 발생 시 임시 처리 (권장 방식 아님)
    import sys
    import logging
    # 현재 파일의 부모 디렉토리(=config)의 부모 디렉토리(=프로젝트 루트)를 경로에 추가
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    try:
        from lib.utils import load_yaml
        from utils.logger import get_logger
        logging.warning("상대 경로 임포트 실패, 경로 조정 시도 성공 (config/__init__.py)")
    except ImportError as final_e:
        logging.error(f"필수 모듈 임포트 실패 (config/__init__.py): {final_e}")
        # 필요한 함수를 찾을 수 없으므로, 모듈 기능 사용 불가
        raise

logger = get_logger(__name__)

# 설정 파일들이 위치한 디렉토리 경로 (이 파일이 있는 곳)
CONFIG_DIR = Path(__file__).resolve().parent

# 기본 설정 파일 이름 정의
DEFAULT_MODEL_CONFIG_FILE = "model_config.yaml"
DEFAULT_PROMPT_TEMPLATES_FILE = "prompt_templates.yaml"
DEFAULT_LOGGING_CONFIG_FILE = "logging_config.yaml"


def load_config(
    config_name: str, # 로드할 설정 파일의 기본 이름 (예: "model_config", "prompt_templates")
    config_dir: Optional[str | Path] = None, # 설정 파일이 있는 디렉토리 경로 (None이면 기본값 사용)
    file_extension: str = ".yaml" # 설정 파일 확장자
) -> Optional[Dict[str, Any]]:
    """
    지정된 이름의 설정 파일을 YAML 형식으로 로드합니다.

    Args:
        config_name (str): 로드할 설정 파일의 기본 이름 (확장자 제외).
                           예: "model_config", "prompt_templates", "logging_config"
        config_dir (str | Path, optional): 설정 파일이 위치한 디렉토리 경로.
                                           None이면 이 패키지 디렉토리(`config/`)를 사용합니다.
                                           기본값: None.
        file_extension (str): 설정 파일의 확장자. 기본값: ".yaml".

    Returns:
        Optional[Dict[str, Any]]: 로드된 설정 내용을 담은 딕셔너리.
                                  파일을 찾지 못하거나 로드 실패 시 None 반환.
    """
    # 설정 파일 경로 결정
    directory = Path(config_dir) if config_dir else CONFIG_DIR
    file_path = directory / f"{config_name}{file_extension}"

    logger.debug(f"설정 파일 로드 시도: {file_path}")

    # 파일 존재 여부 확인
    if not file_path.exists():
        logger.error(f"설정 파일을 찾을 수 없습니다: {file_path}")
        return None

    try:
        # lib.utils의 load_yaml 함수 사용하여 로드
        config_data = load_yaml(str(file_path))
        if config_data:
            logger.info(f"설정 파일 로드 성공: {file_path}")
            return config_data
        else:
            logger.warning(f"설정 파일이 비어있거나 로드에 실패했습니다: {file_path}")
            return None
    except Exception as e:
        logger.error(f"설정 파일 '{file_path}' 로드 중 오류 발생: {e}", exc_info=True)
        return None


def load_all_configs(config_dir: Optional[str | Path] = None) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    config 디렉토리 내의 모든 기본 설정 파일들을 로드합니다.

    Args:
        config_dir (str | Path, optional): 설정 파일들이 위치한 디렉토리. 기본값: None (config/ 사용).

    Returns:
        Dict[str, Optional[Dict[str, Any]]]: 각 설정 파일 이름을 키로 하고,
                                             로드된 설정 딕셔너리(또는 실패 시 None)를 값으로 하는 딕셔너리.
                                             예: {"model": {...}, "prompts": {...}, "logging": {...}}
    """
    configs = {}
    configs["model"] = load_config(DEFAULT_MODEL_CONFIG_FILE.replace(".yaml", ""), config_dir)
    configs["prompts"] = load_config(DEFAULT_PROMPT_TEMPLATES_FILE.replace(".yaml", ""), config_dir)
    configs["logging"] = load_config(DEFAULT_LOGGING_CONFIG_FILE.replace(".yaml", ""), config_dir)

    loaded_count = sum(1 for cfg in configs.values() if cfg is not None)
    logger.info(f"총 {len(configs)}개의 설정 파일 중 {loaded_count}개 로드 완료.")

    return configs

# 패키지에서 직접 접근 가능하도록 설정 (선택 사항)
# 예: from config import load_config
__all__ = [
    "load_config",
    "load_all_configs",
    "DEFAULT_MODEL_CONFIG_FILE",
    "DEFAULT_PROMPT_TEMPLATES_FILE",
    "DEFAULT_LOGGING_CONFIG_FILE",
    "CONFIG_DIR"
] 