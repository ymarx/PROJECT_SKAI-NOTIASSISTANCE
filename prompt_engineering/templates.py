"""
SKAI-NotiAssistance의 프롬프트 템플릿 처리 모듈입니다.

이 모듈은 외부 YAML 파일(`config/prompt_templates.yaml`)에 정의된
프롬프트 템플릿들을 로드하고, 필요한 변수를 채워 최종 프롬프트를 생성하는
유틸리티 함수들을 제공합니다. 프롬프트 내용을 코드와 분리하여 관리함으로써
유지보수성과 유연성을 높입니다.
"""

import os
import re # 변수 추출 등에 사용될 수 있음 (현재는 미사용)
from typing import Any, Dict, List, Optional, Union

# lib.utils 모듈에서 YAML 로딩 함수 임포트
# 프로젝트 구조에 따라 경로 조정 필요 가능성 있음
try:
    from lib.utils import load_yaml
    from utils.logger import get_logger # 로깅 사용
except ImportError:
    # 경로 문제 발생 시 임시 처리
    import sys
    import logging
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from lib.utils import load_yaml
        from utils.logger import get_logger
        logging.warning("상대 경로 임포트 실패, 경로 조정 시도 성공 (prompt_engineering/templates.py)")
    except ImportError as final_e:
        logging.error(f"필수 모듈 임포트 실패 (prompt_engineering/templates.py): {final_e}")
        # 필요한 함수를 찾을 수 없으므로, 모듈 기능 사용 불가
        # 이 경우, 프로그램 실행에 문제가 발생할 수 있습니다.
        raise

logger = get_logger(__name__) # 이 모듈용 로거 생성

# 기본 프롬프트 템플릿 파일 경로 설정 (이 파일의 상위 디렉토리 기준)
# __file__ 은 현재 파일의 절대 경로입니다.
# os.path.dirname() 으로 디렉토리 경로를 얻습니다.
DEFAULT_TEMPLATES_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config", "prompt_templates.yaml")
)
logger.debug(f"기본 프롬프트 템플릿 파일 경로: {DEFAULT_TEMPLATES_PATH}")


def load_template(
    template_name: str, # 로드할 템플릿의 이름 (YAML 파일 내 키)
    template_type: str = "user_prompts", # 템플릿 유형 (YAML 파일 내 상위 키, 예: 'system_prompts')
    template_file: Optional[str] = None # 사용할 YAML 파일 경로 (None이면 기본 경로 사용)
) -> Optional[str]:
    """
    지정된 이름과 유형의 프롬프트 템플릿을 YAML 파일에서 로드합니다.

    Args:
        template_name: 로드할 템플릿의 이름 (YAML 파일에서 해당 키를 찾음).
        template_type: 템플릿이 속한 유형 (YAML 파일 내 상위 키). 기본값은 'user_prompts'.
                       예: 'system_prompts', 'user_prompts', 'analysis_prompts' 등 YAML 파일 구조에 따라 지정.
        template_file: 프롬프트 템플릿이 정의된 YAML 파일의 경로 (선택 사항).
                       None이면 `DEFAULT_TEMPLATES_PATH`에 정의된 기본 경로를 사용합니다.

    Returns:
        성공적으로 로드된 템플릿 문자열. 템플릿을 찾지 못하거나 오류 발생 시 None을 반환합니다.
    """
    # 사용할 템플릿 파일 경로 결정 (입력값 우선, 없으면 기본값)
    file_path = template_file or DEFAULT_TEMPLATES_PATH

    # 템플릿 파일 존재 여부 확인
    if not os.path.exists(file_path):
        logger.error(f"프롬프트 템플릿 파일을 찾을 수 없습니다: {file_path}")
        return None

    try:
        # lib.utils의 load_yaml 함수를 사용하여 YAML 파일 로드
        templates = load_yaml(file_path)

        # 파일 로드 실패 또는 빈 파일 처리
        if not templates:
            logger.error(f"템플릿 파일이 비어있거나 로드에 실패했습니다: {file_path}")
            return None

        # YAML 데이터 구조 확인 및 템플릿 추출
        if template_type not in templates:
            logger.error(f"템플릿 유형 '{template_type}'을(를) 파일에서 찾을 수 없습니다: {file_path}")
            return None

        if template_name not in templates[template_type]:
            logger.error(f"템플릿 이름 '{template_name}'을(를) 유형 '{template_type}' 아래에서 찾을 수 없습니다.")
            return None

        # 최종 템플릿 문자열 가져오기
        template = templates[template_type][template_name]

        # 템플릿이 문자열인지 확인 (YAML 구조 오류 방지)
        if not isinstance(template, str):
             logger.error(f"로드된 템플릿 '{template_name}'이 문자열이 아닙니다 (타입: {type(template)}). YAML 파일 구조를 확인하세요.")
             return None

        logger.debug(f"템플릿 '{template_name}' (유형: {template_type}) 로드 성공.")
        return template.strip() # 앞뒤 공백 제거 후 반환

    except Exception as e:
        # YAML 파싱 오류 등 예외 처리
        logger.error(f"템플릿 '{template_name}' 로드 중 오류 발생: {str(e)}", exc_info=True)
        return None


def format_prompt(
    template: str, # 포맷팅할 템플릿 문자열
    variables: Dict[str, Any], # 템플릿에 채울 변수 딕셔너리
    strict: bool = False # 누락된 변수 발견 시 오류 발생 여부
) -> str:
    """
    주어진 프롬프트 템플릿 문자열에 변수 값을 채워 최종 프롬프트를 생성합니다.

    템플릿 내 `{변수명}` 형태의 플레이스홀더를 `variables` 딕셔너리의 해당 키 값으로 치환합니다.

    Args:
        template: 포맷팅할 프롬프트 템플릿 문자열.
                  예: "분석 대상: {equipment_id}, 메시지: {message}"
        variables: 템플릿 내 플레이스홀더를 채울 키-값 쌍의 딕셔너리.
                   예: {"equipment_id": "PUMP-101", "message": "진동 감지됨"}
        strict: 만약 True이면, 템플릿에 있는 변수가 `variables` 딕셔너리에 없을 경우
                오류(ValueError)를 발생시킵니다. False이면 누락된 변수는 그냥 `{변수명}` 그대로 둡니다.
                기본값: False.

    Returns:
        변수가 채워진 최종 프롬프트 문자열. 오류 발생 시 원본 템플릿 반환 (strict=False) 또는 예외 발생 (strict=True).
    """
    if not template: # 빈 템플릿 처리
        return ""

    try:
        # 템플릿 내 모든 {변수명} 형태의 플레이스홀더 찾기
        # 정규표현식을 사용하여 중괄호 안의 변수 이름만 추출
        placeholder_pattern = r"\{([a-zA-Z0-9_]+)\}" # {알파벳,숫자,_} 형태
        template_vars = set(re.findall(placeholder_pattern, template)) # 템플릿 내 모든 변수 이름 집합

        # strict 모드일 경우, 누락된 변수 확인
        if strict:
            missing_vars = template_vars - set(variables.keys()) # 템플릿 변수 중 variables에 없는 것 찾기
            if missing_vars:
                # 누락된 변수가 있으면 오류 발생
                raise ValueError(f"프롬프트 포맷팅 실패: 다음 변수가 누락되었습니다 - {', '.join(missing_vars)}")

        # 변수 치환 수행
        formatted_prompt = template
        for var_name in template_vars:
            if var_name in variables:
                # 플레이스홀더 (예: {equipment_id}) 를 실제 값으로 교체
                placeholder = "{" + var_name + "}"
                # 변수 값을 문자열로 변환하여 치환 (어떤 타입이든 안전하게 처리)
                formatted_prompt = formatted_prompt.replace(placeholder, str(variables[var_name]))
            # strict=False 이고 변수가 누락된 경우, 플레이스홀더는 그대로 남음

        logger.debug("프롬프트 템플릿 포맷팅 완료.")
        return formatted_prompt

    except Exception as e:
        # 포맷팅 중 예상치 못한 오류 발생 시
        logger.error(f"프롬프트 포맷팅 중 오류 발생: {str(e)}")
        if strict:
            raise # strict 모드면 오류 다시 발생시킴
        else:
            # strict 모드가 아니면 원본 템플릿 반환 (오류보다는 안전)
            return template


def get_system_prompt(
    prompt_name: str = "default", # 가져올 시스템 프롬프트 이름 (YAML 키)
    template_file: Optional[str] = None # 템플릿 파일 경로 (None이면 기본값 사용)
) -> str:
    """
    지정된 이름의 시스템 프롬프트를 템플릿 파일에서 로드합니다.

    Args:
        prompt_name: 로드할 시스템 프롬프트의 이름 (YAML 파일의 'system_prompts' 아래 키).
                     기본값: "default".
        template_file: 사용할 템플릿 파일 경로 (선택 사항).

    Returns:
        로드된 시스템 프롬프트 문자열. 찾지 못하면 빈 문자열("") 반환.
    """
    # load_template 함수를 사용하여 'system_prompts' 유형의 템플릿 로드
    prompt = load_template(
        template_name=prompt_name,
        template_type="system_prompts", # 시스템 프롬프트 유형 지정
        template_file=template_file
    )

    # 로드 실패 시 빈 문자열 반환, 성공 시 해당 프롬프트 반환
    return prompt or ""


# (선택적) 특정 작업에 대한 프롬프트를 로드하고 포맷팅하는 헬퍼 함수
# agent.py나 nodes.py에서 이 함수를 직접 사용할 수도 있고,
# load_template과 format_prompt를 개별적으로 사용할 수도 있습니다.
def load_and_format_prompt(
    template_name: str, # 로드할 템플릿 이름
    variables: Dict[str, Any], # 채울 변수
    template_type: str = "user_prompts", # 템플릿 유형
    template_file: Optional[str] = None, # 템플릿 파일 경로
    strict: bool = False # 변수 누락 시 오류 여부
) -> Optional[str]:
    """
    템플릿을 로드하고 주어진 변수로 포맷팅하는 과정을 한 번에 수행합니다.

    Args:
        template_name: 로드할 템플릿 이름.
        variables: 템플릿에 채울 변수 딕셔너리.
        template_type: 템플릿 유형 (기본값: 'user_prompts').
        template_file: 템플릿 파일 경로 (기본값: None).
        strict: 변수 누락 시 오류 발생 여부 (기본값: False).

    Returns:
        최종 포맷팅된 프롬프트 문자열. 템플릿 로드나 포맷팅 실패 시 None 반환.
    """
    # 1. 템플릿 로드
    template_content = load_template(
        template_name=template_name,
        template_type=template_type,
        template_file=template_file
    )

    # 템플릿 로드 실패 시 None 반환
    if template_content is None:
        return None

    # 2. 프롬프트 포맷팅
    # format_prompt는 strict=True 시 오류를 발생시킬 수 있으므로 try-except 사용
    try:
        formatted_prompt = format_prompt(
            template=template_content,
            variables=variables,
            strict=strict
        )
        return formatted_prompt
    except ValueError as e:
        # strict=True 이고 변수 누락 시 발생하는 오류 처리
        logger.error(f"프롬프트 포맷팅 실패 (strict=True, 변수 누락 가능성): {e}")
        return None
    except Exception as e:
        # 기타 예외 처리
        logger.error(f"프롬프트 포맷팅 중 예상치 못한 오류 발생: {e}")
        return None

# --- 이전 `load_prompt_by_task` 함수는 특정 작업 로직을 포함하므로,
# --- agent.py나 nodes.py로 옮기는 것이 더 적합할 수 있습니다.
# --- 여기서는 범용 템플릿 로드/포맷팅 함수만 남겨둡니다. ---

# def load_prompt_by_task(...):
#     # 이 함수는 특정 task 이름에 따라 시스템 프롬프트까지 자동으로 결정하는 로직이 포함되어 있어,
#     # templates.py의 범용적인 역할과는 다소 거리가 있습니다.
#     # 필요하다면 별도의 유틸리티 모듈이나 사용하는 곳(agent, node)에서 구현하는 것이
#     # 책임 분리 원칙에 더 맞을 수 있습니다.
#     pass 