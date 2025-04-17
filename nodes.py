"""
SKAI-NotiAssistance 에이전트의 처리 노드 정의 모듈입니다.

이 모듈은 에이전트의 작업 흐름을 구성하는 개별 처리 단위인 '노드(Node)'들을 정의합니다.
각 노드는 `BaseNode` 클래스를 상속받아 특정 작업을 수행하며,
입력(inputs)과 에이전트 상태(state)를 받아 처리 결과를 출력(outputs)으로 반환합니다.

에이전트(`agent.py`)는 이러한 노드들을 조합하여 복잡한 작업을 수행합니다.
(예: 알림 분석, 유사 사례 검색 등)
"""

import os
import re # 정규표현식 사용 (주로 LLM 응답 파싱에 활용)
import json
import time
import logging # 표준 로깅 사용 가능
from typing import Any, Dict, List, Optional, Union, Callable, Generator # 타입 힌트를 위한 모듈

# 프로젝트 내 다른 모듈 임포트
# 상위 디렉토리의 모듈을 참조하므로, 실행 환경에 따라 경로 조정이 필요할 수 있습니다.
try:
    # lib 폴더 내의 기본 클래스, 모델 관리자, 벡터 저장소, 유틸리티 함수 임포트
    from lib.base import BaseNode
    from lib.model_manager import ModelManager
    from lib.vector_store import VectorStore
    from lib.utils import load_yaml, merge_configs # 필요 시 설정 파일 로드 및 병합에 사용
    # prompt_engineering 폴더 내의 템플릿 포맷팅, 시스템 프롬프트 로드, few-shot 예제 추가 함수 임포트
    from prompt_engineering.templates import format_prompt, get_system_prompt, load_template # 수정: load_template 추가
    from prompt_engineering.few_shot import add_few_shot_to_prompt
    # utils 폴더 내의 로거 가져오기, 토큰 수 계산 및 텍스트 자르기 함수 임포트
    from utils.logger import get_logger
    from utils.token_counter import count_tokens, truncate_text_to_token_limit # 수정: truncate 추가
except ImportError as e:
    # 만약 위 경로에서 임포트 에러 발생 시, 파이썬 실행 경로 문제일 수 있습니다.
    # 프로젝트 루트 디렉토리에서 실행하거나, PYTHONPATH 환경 변수를 설정해야 할 수 있습니다.
    # 임시방편으로 sys.path에 상위 디렉토리를 추가하여 시도합니다.
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from lib.base import BaseNode
        from lib.model_manager import ModelManager
        from lib.vector_store import VectorStore
        from lib.utils import load_yaml, merge_configs
        from prompt_engineering.templates import format_prompt, get_system_prompt, load_template
        from prompt_engineering.few_shot import add_few_shot_to_prompt
        from utils.logger import get_logger
        from utils.token_counter import count_tokens, truncate_text_to_token_limit
        logging.warning(f"상대 경로 임포트 실패, 경로 조정 시도 성공: {e}")
    except ImportError as final_e:
        # 최종적으로 임포트 실패 시 에러 로깅 및 프로그램 중단 가능성 알림
        logging.error(f"필수 모듈 임포트 실패! 실행 경로 및 프로젝트 구조를 확인하세요: {final_e}")
        raise # 에러를 다시 발생시켜 프로그램 중단


# 이 모듈 전체에서 사용할 로거 생성
logger = get_logger(__name__)


class LLMNode(BaseNode):
    """
    언어 모델(LLM)과 상호작용하는 노드입니다.

    주어진 프롬프트를 ModelManager를 통해 선택된 LLM(로컬, Ollama, OpenAI 등)에게 전달하고,
    생성된 텍스트 응답을 반환합니다.
    """

    def __init__(
        self,
        model_manager: ModelManager, # ModelManager 인스턴스를 주입받음
        name: Optional[str] = None,
        system_prompt: Optional[str] = None # 노드 레벨 기본 시스템 프롬프트 (선택 사항)
    ):
        """
        LLMNode를 초기화합니다.

        Args:
            model_manager: 사용할 ModelManager 객체. LLM 호출을 담당합니다.
            name: 노드의 고유 이름 (지정하지 않으면 자동 생성).
            system_prompt: 이 노드에서 LLM 호출 시 기본으로 사용할 시스템 프롬프트 (선택 사항).
                           `process` 메서드 호출 시 `inputs`에서 재정의 가능합니다.
        """
        super().__init__(name=name) # BaseNode 초기화 (id, name, logger 생성)
        self.model_manager = model_manager
        self.system_prompt = system_prompt
        self.logger.info(f"LLMNode '{self.name}' 초기화 완료.")

    def process(self, inputs: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        입력으로 받은 프롬프트를 사용하여 LLM으로부터 텍스트 응답을 생성합니다.

        Args:
            inputs (Dict[str, Any]): 노드 처리에 필요한 입력 데이터. 다음 키를 포함할 수 있습니다:
                - prompt (str): LLM에게 전달할 필수 사용자 프롬프트.
                - system_prompt (str, Optional): LLM의 역할/맥락 설정용 시스템 프롬프트.
                                                 제공되지 않으면 노드 초기화 시 설정된 기본값을 사용합니다.
                - temperature (float, Optional): 생성 온도 (재정의 시).
                - max_tokens (int, Optional): 최대 생성 토큰 수 (재정의 시).
                - stream (bool, Optional): 스트리밍 응답 여부 (재정의 시).
                - **kwargs: ModelManager.generate에 전달될 추가 키워드 인수.
                            (예: Ollama 사용 시 `options` 딕셔너리 전달 가능)
            state (Dict[str, Any]): 현재 에이전트의 전체 상태 딕셔너리 (읽기/쓰기 가능).
                                   이 노드에서는 주로 로깅이나 상태 기반 분기 로직에 사용될 수 있으나,
                                   현재 구현에서는 직접적으로 상태를 변경하지는 않습니다.

        Returns:
            Dict[str, Any]: 처리 결과. 다음 키를 포함합니다:
                - response (str): LLM이 생성한 텍스트 응답 (스트리밍 아닐 때).
                - response_generator (Generator, Optional): 스트리밍 응답 시 텍스트 청크 제너레이터.
                - prompt_tokens (int): 입력 프롬프트의 토큰 수 (시스템 프롬프트 포함).
                - response_tokens (int, Optional): 생성된 응답의 토큰 수 (스트리밍 아닐 때).
                - total_tokens (int, Optional): 총 사용 토큰 수 (스트리밍 아닐 때).
                - elapsed_time (float): LLM 호출에 걸린 시간 (초).
                - error (str, Optional): 처리 중 오류 발생 시 오류 메시지.
        """
        prompt = inputs.get("prompt")
        if not prompt:
            self.logger.warning(f"'{self.name}': 처리할 'prompt'가 입력에 없습니다.")
            return {"response": "", "error": "프롬프트가 제공되지 않았습니다."}

        # 입력에서 파라미터 가져오기 (없으면 ModelManager의 기본값 사용됨)
        system_prompt = inputs.get("system_prompt", self.system_prompt)
        temperature = inputs.get("temperature") # None이면 ModelManager 기본값 사용
        max_tokens = inputs.get("max_tokens")   # None이면 ModelManager 기본값 사용
        stream = inputs.get("stream", False)    # 스트리밍 여부 (기본값 False)
        # inputs에서 예약된 키(prompt, system_prompt 등)를 제외한 나머지를 kwargs로 전달
        llm_kwargs = {k: v for k, v in inputs.items() if k not in
                      ['prompt', 'system_prompt', 'temperature', 'max_tokens', 'stream']}

        try:
            # LLM 호출 시작 시간 기록
            start_time = time.time()
            self.logger.debug(f"'{self.name}': LLM 호출 시작 (프롬프트 길이: {len(prompt)}자, 스트리밍: {stream})")

            # ModelManager를 통해 LLM 호출
            response_or_generator = self.model_manager.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **llm_kwargs # Ollama options 등 추가 파라미터 전달
            )

            # LLM 호출 종료 시간 기록 및 경과 시간 계산
            end_time = time.time()
            elapsed_time = end_time - start_time

            # 입력 토큰 수 계산 (항상 가능)
            prompt_tokens = count_tokens(prompt, self.model_manager.model_name)
            system_tokens = count_tokens(system_prompt, self.model_manager.model_name) if system_prompt else 0
            total_input_tokens = prompt_tokens + system_tokens

            # 스트리밍 응답 처리
            if stream and isinstance(response_or_generator, Generator):
                self.logger.info(f"'{self.name}': LLM 스트리밍 응답 수신 ({elapsed_time:.2f}초 소요, 토큰 수 계산은 완료 후 가능)")
                # 스트리밍 제너레이터를 직접 반환
                return {
                    "response_generator": response_or_generator,
                    "prompt_tokens": total_input_tokens,
                    "response_tokens": None, # 스트리밍 중에는 알 수 없음
                    "total_tokens": None,    # 스트리밍 중에는 알 수 없음
                    "elapsed_time": elapsed_time
                }
            # 일반 (비-스트리밍) 응답 처리
            elif isinstance(response_or_generator, str):
                response_text = response_or_generator
                response_tokens = count_tokens(response_text, self.model_manager.model_name)
                total_tokens = total_input_tokens + response_tokens
                self.logger.info(f"'{self.name}': LLM 응답 생성 완료 ({total_tokens} 토큰, {elapsed_time:.2f}초 소요)")
                # 결과 딕셔너리 반환
                return {
                    "response": response_text,
                    "prompt_tokens": total_input_tokens,
                    "response_tokens": response_tokens,
                    "total_tokens": total_tokens,
                    "elapsed_time": elapsed_time
                }
            else:
                # 예상치 못한 응답 타입
                self.logger.error(f"'{self.name}': ModelManager.generate()로부터 예상치 못한 타입의 응답 반환: {type(response_or_generator)}")
                return {"response": "", "error": "예상치 못한 LLM 응답 타입"}


        except Exception as e:
            self.logger.error(f"'{self.name}': LLM 처리 중 오류 발생: {str(e)}", exc_info=True)
            # 오류 발생 시에도 입력 토큰 수는 계산 가능할 수 있음
            prompt_tokens_on_error = count_tokens(prompt, self.model_manager.model_name) if prompt else 0
            system_tokens_on_error = count_tokens(system_prompt, self.model_manager.model_name) if system_prompt else 0
            return {
                "response": "",
                "prompt_tokens": prompt_tokens_on_error + system_tokens_on_error,
                "error": str(e)
                }


class PromptTemplateNode(BaseNode):
    """
    프롬프트 템플릿을 로드하고 주어진 변수로 포맷팅하는 노드입니다.

    템플릿 파일(YAML)에서 템플릿을 로드하거나, 입력으로 직접 템플릿 문자열을 받아
    변수를 채워 최종 프롬프트를 생성합니다. Few-shot 예제를 추가하는 기능도 지원합니다.
    """

    def __init__(
        self,
        template_file: Optional[str] = None, # 사용할 템플릿 파일 경로 (선택 사항)
        name: Optional[str] = None
    ):
        """
        PromptTemplateNode를 초기화합니다.

        Args:
            template_file: 프롬프트 템플릿이 정의된 YAML 파일 경로 (선택 사항).
                           지정하지 않으면 기본 경로(`config/prompt_templates.yaml`)를 사용하려 시도합니다.
            name: 노드의 고유 이름 (지정하지 않으면 자동 생성).
        """
        super().__init__(name=name)
        self.template_file = template_file
        # 실제 템플릿 로드는 process 시점에 수행 (파일 변경 반영 가능)
        self.logger.info(f"PromptTemplateNode '{self.name}' 초기화 완료 (템플릿 파일: {template_file or '기본값'})")

    def process(self, inputs: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        템플릿과 변수를 사용하여 최종 프롬프트를 생성합니다.

        Args:
            inputs (Dict[str, Any]): 노드 처리에 필요한 입력 데이터. 다음 키를 포함할 수 있습니다:
                - template_name (str, Optional): 템플릿 파일에서 로드할 템플릿의 이름. 'template' 키와 함께 사용할 수 없습니다.
                - template_type (str, Optional): 템플릿 파일 내의 템플릿 유형 (예: 'user_prompts', 'system_prompts'). 기본값: 'user_prompts'.
                - template (str, Optional): 사용할 프롬프트 템플릿 문자열 직접 제공. 'template_name' 키와 함께 사용할 수 없습니다.
                - variables (Dict[str, Any]): 템플릿 내의 플레이스홀더({변수명})를 채울 변수 딕셔너리.
                - use_few_shot (bool, Optional): Few-shot 예제를 프롬프트에 추가할지 여부. 기본값: False.
                - few_shot_task (str, Optional): Few-shot 예제를 가져올 작업 이름 (템플릿 파일 내 정의). 'use_few_shot'이 True일 때 필요.
                - few_shot_count (int, Optional): 사용할 Few-shot 예제 최대 개수.
                - few_shot_format (str, Optional): Few-shot 예제 포맷 방식 ('basic', 'markdown', 'qa'). 기본값: 'basic'.
            state (Dict[str, Any]): 현재 에이전트 상태. 사용되지 않음.

        Returns:
            Dict[str, Any]: 처리 결과. 다음 키를 포함합니다:
                - prompt (str): 최종적으로 생성된 프롬프트 문자열.
                - error (str, Optional): 처리 중 오류 발생 시 오류 메시지.
        """
        template_name = inputs.get("template_name")
        template_type = inputs.get("template_type", "user_prompts")
        template_text = inputs.get("template") # 직접 입력받은 템플릿
        variables = inputs.get("variables", {}) # 템플릿 변수
        use_few_shot = inputs.get("use_few_shot", False)
        few_shot_task = inputs.get("few_shot_task")
        few_shot_count = inputs.get("few_shot_count")
        few_shot_format = inputs.get("few_shot_format", "basic")

        try:
            final_template = ""
            # 1. 사용할 템플릿 결정 (직접 입력 또는 파일 로드)
            if template_text:
                # 입력으로 템플릿 문자열이 직접 제공된 경우
                if template_name:
                    self.logger.warning(f"'{self.name}': 'template'과 'template_name'이 모두 제공되었습니다. 'template'을 사용합니다.")
                final_template = template_text
                self.logger.debug(f"'{self.name}': 입력받은 템플릿 문자열 사용.")
            elif template_name:
                # 템플릿 파일에서 이름으로 로드
                loaded_template = load_template(
                    template_name=template_name,
                    template_type=template_type,
                    template_file=self.template_file # 초기화 시 설정된 파일 경로 사용
                )
                if not loaded_template:
                    error_msg = f"템플릿 '{template_name}'(유형: {template_type})을 찾을 수 없습니다."
                    self.logger.error(f"'{self.name}': {error_msg}")
                    return {"prompt": "", "error": error_msg}
                final_template = loaded_template
                self.logger.debug(f"'{self.name}': 파일에서 템플릿 '{template_name}' 로드 완료.")
            else:
                # 템플릿 정보가 전혀 없는 경우
                error_msg = "처리할 템플릿 정보('template' 또는 'template_name')가 없습니다."
                self.logger.error(f"'{self.name}': {error_msg}")
                return {"prompt": "", "error": error_msg}

            # 2. 템플릿 포맷팅 (변수 치환)
            # format_prompt 함수는 템플릿 내 {변수명} 부분을 variables 딕셔너리의 값으로 치환합니다.
            # strict=False 옵션은 템플릿에는 있지만 variables에 없는 변수가 있어도 오류를 발생시키지 않습니다.
            formatted_prompt = format_prompt(final_template, variables, strict=False)
            self.logger.debug(f"'{self.name}': 템플릿 포맷팅 완료.")

            # 3. Few-shot 예제 추가 (선택 사항)
            if use_few_shot and few_shot_task:
                self.logger.debug(f"'{self.name}': Few-shot 예제 추가 시도 (작업: {few_shot_task})")
                # `add_few_shot_to_prompt` 함수는 템플릿 파일에서 해당 task의 예제를 가져와
                # 지정된 형식(format_type)으로 포맷팅한 후, 기존 프롬프트 뒤에 덧붙입니다.
                formatted_prompt = add_few_shot_to_prompt(
                    prompt=formatted_prompt,
                    task=few_shot_task,
                    template_file=self.template_file, # 예제가 있는 템플릿 파일 경로
                    num_examples=few_shot_count,   # 가져올 예제 수 제한
                    format_type=few_shot_format    # 예제 포맷 방식
                )
                self.logger.debug(f"'{self.name}': Few-shot 예제 추가 완료.")

            # 최종 생성된 프롬프트 반환
            return {"prompt": formatted_prompt}

        except Exception as e:
            self.logger.error(f"'{self.name}': 프롬프트 템플릿 처리 중 오류 발생: {str(e)}", exc_info=True)
            return {"prompt": "", "error": str(e)}


class VectorSearchNode(BaseNode):
    """
    벡터 저장소(VectorStore)를 검색하여 관련 문서를 찾는 노드입니다.

    입력으로 받은 텍스트 쿼리의 임베딩을 생성하고, 이 임베딩을 사용하여
    VectorStore에서 가장 유사한 문서들을 검색합니다.
    """

    def __init__(
        self,
        vector_store: VectorStore,   # VectorStore 인스턴스를 주입받음
        model_manager: ModelManager, # 임베딩 생성을 위한 ModelManager 주입받음
        name: Optional[str] = None
    ):
        """
        VectorSearchNode를 초기화합니다.

        Args:
            vector_store: 사용할 VectorStore 객체. 벡터 검색을 담당합니다.
            model_manager: 쿼리 텍스트를 임베딩 벡터로 변환할 ModelManager 객체.
            name: 노드의 고유 이름 (지정하지 않으면 자동 생성).
        """
        super().__init__(name=name)
        self.vector_store = vector_store
        self.model_manager = model_manager
        # VectorStore의 provider 정보 로깅 (디버깅에 유용)
        vs_provider = getattr(vector_store, 'provider', '알 수 없음')
        self.logger.info(f"VectorSearchNode '{self.name}' 초기화 완료 (VectorStore Provider: {vs_provider})")

    def process(self, inputs: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        쿼리 텍스트를 사용하여 벡터 저장소에서 유사 문서를 검색합니다.

        Args:
            inputs (Dict[str, Any]): 노드 처리에 필요한 입력 데이터. 다음 키를 포함할 수 있습니다:
                - query (str): 검색할 텍스트 쿼리. 필수 항목입니다.
                - top_k (int, Optional): 반환할 최대 결과 수. 기본값: 5.
                - filter_metadata (Dict[str, Any], Optional): 검색 시 적용할 메타데이터 필터.
                                                             VectorStore provider가 지원해야 합니다 (예: Chroma, Pinecone).
            state (Dict[str, Any]): 현재 에이전트 상태. 사용되지 않음.

        Returns:
            Dict[str, Any]: 처리 결과. 다음 키를 포함합니다:
                - results (List[Dict[str, Any]]): 검색된 문서 목록. 각 문서는 'id', 'text', 'metadata', 'score' 포함.
                - error (str, Optional): 처리 중 오류 발생 시 오류 메시지.
        """
        query = inputs.get("query")
        if not query:
            self.logger.warning(f"'{self.name}': 처리할 'query'가 입력에 없습니다.")
            return {"results": [], "error": "쿼리가 제공되지 않았습니다."}

        top_k = inputs.get("top_k", 5) # 반환할 결과 수 (기본 5개)
        filter_metadata = inputs.get("filter_metadata") # 메타데이터 필터 (선택 사항)

        try:
            # 1. 쿼리 텍스트 임베딩 생성
            self.logger.debug(f"'{self.name}': 쿼리 임베딩 생성 시작 (쿼리: '{query[:50]}...')")
            # ModelManager를 사용하여 쿼리를 임베딩 벡터로 변환합니다.
            # create_embeddings는 텍스트 리스트를 받으므로, 단일 쿼리도 리스트에 담아 전달합니다.
            query_embedding = self.model_manager.create_embeddings([query])[0] # 결과 리스트의 첫 번째(유일한) 임베딩을 사용합니다.
            self.logger.debug(f"'{self.name}': 쿼리 임베딩 생성 완료 (벡터 차원: {len(query_embedding)})")

            # 2. VectorStore 검색 실행
            self.logger.debug(f"'{self.name}': VectorStore 검색 시작 (top_k: {top_k}, filter: {filter_metadata})")
            # VectorStore의 query 메서드를 호출하여 검색을 수행합니다.
            results = self.vector_store.query(
                query_embedding=query_embedding, # 생성된 쿼리 임베딩
                top_k=top_k,                     # 반환할 결과 수
                filter_metadata=filter_metadata  # 메타데이터 필터 (제공된 경우)
            )
            self.logger.info(f"'{self.name}': VectorStore 검색 완료 ({len(results)}개 결과 반환)")

            # 검색 결과 반환
            return {"results": results}

        except Exception as e:
            self.logger.error(f"'{self.name}': 벡터 검색 중 오류 발생: {str(e)}", exc_info=True)
            # 오류 발생 시 빈 결과 리스트와 에러 메시지 반환
            return {"results": [], "error": str(e)}


class DocumentFormatNode(BaseNode):
    """
    벡터 검색 결과를 사람이 읽기 좋은 형식 또는 LLM 입력에 적합한 형식의
    단일 문서(문자열)로 포맷팅하는 노드입니다.

    예를 들어, 검색된 여러 문서 조각들을 번호를 매겨 하나의 컨텍스트 문자열로 합치는 데 사용됩니다.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        # 각 검색 결과를 어떻게 표시할지 정의하는 템플릿 문자열입니다.
        # Python의 f-string 형식을 따르며, {index}, {text}, {metadata}, {score} 변수를 사용할 수 있습니다.
        format_template: str = "문서 {index}: {text}\n출처: {metadata}\n유사도: {score:.4f}\n\n", # 기본 포맷 템플릿
        max_tokens: Optional[int] = None # 포맷팅된 문서의 최대 토큰 수 제한 (선택 사항)
    ):
        """
        DocumentFormatNode를 초기화합니다.

        Args:
            name: 노드의 고유 이름 (지정하지 않으면 자동 생성).
            format_template: 각 검색 결과를 포맷팅할 Python f-string 템플릿.
                             사용 가능한 변수: {index}(1부터 시작), {text}, {metadata}(딕셔너리), {score}.
            max_tokens: 포맷팅된 최종 문서의 최대 토큰 수 제한. 초과 시 뒷부분이 잘릴 수 있습니다.
                       LLM의 컨텍스트 길이 제한을 맞추는 데 유용합니다.
        """
        super().__init__(name=name)
        self.format_template = format_template
        self.max_tokens = max_tokens
        self.logger.info(f"DocumentFormatNode '{self.name}' 초기화 완료.")

    def process(self, inputs: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        입력으로 받은 검색 결과 목록을 단일 문서 문자열로 포맷팅합니다.

        Args:
            inputs (Dict[str, Any]): 노드 처리에 필요한 입력 데이터. 다음 키를 포함할 수 있습니다:
                - results (List[Dict[str, Any]]): VectorSearchNode 등에서 반환된 검색 결과 목록. 필수 항목입니다.
                                                 각 딕셔너리는 'text', 'metadata', 'score' 키를 포함하는 것이 좋습니다.
                - max_tokens (int, Optional): 이 노드 실행 시 최대 토큰 수를 재정의합니다.
                                              None이면 초기화 시 설정된 값을 사용합니다.
            state (Dict[str, Any]): 현재 에이전트 상태. 사용되지 않음.

        Returns:
            Dict[str, Any]: 처리 결과. 다음 키를 포함합니다:
                - document (str): 포맷팅된 최종 문서 문자열. 결과가 없으면 빈 문자열.
                - error (str, Optional): 처리 중 오류 발생 시 오류 메시지.
        """
        results = inputs.get("results") # 검색 결과 리스트 가져오기
        if not results:
            # 입력 results가 없거나 비어있으면 경고 로그 남기고 빈 문서 반환
            self.logger.warning(f"'{self.name}': 포맷팅할 'results'가 입력에 없거나 비어있습니다.")
            return {"document": ""} # 오류 대신 빈 문자열 반환

        # 이 노드 실행 시 사용할 max_tokens 결정 (입력값 우선)
        current_max_tokens = inputs.get("max_tokens", self.max_tokens)

        try:
            formatted_parts = [] # 포맷팅된 각 결과 조각을 저장할 리스트
            # 검색 결과 리스트를 순회하며 각 항목 포맷팅
            for i, result in enumerate(results):
                # format_template을 사용하여 각 result 딕셔너리의 내용을 문자열로 만듭니다.
                # .get()을 사용하여 키가 없는 경우에도 오류 없이 기본값을 사용하도록 합니다.
                formatted_result = self.format_template.format(
                    index=i + 1, # 1부터 시작하는 인덱스
                    text=result.get("text", "내용 없음"),
                    metadata=result.get("metadata", {}), # 메타데이터는 딕셔너리 그대로 전달
                    score=result.get("score", 0.0)     # 점수 (소수점 4자리까지 표시 예시)
                )
                formatted_parts.append(formatted_result) # 포맷팅된 문자열을 리스트에 추가

            # 포맷팅된 모든 조각들을 하나의 긴 문자열로 결합
            document = "".join(formatted_parts)
            self.logger.debug(f"'{self.name}': {len(results)}개 검색 결과를 단일 문서로 포맷팅 완료 (문서 길이: {len(document)}자)")

            # 최대 토큰 수 제한 적용 (설정된 경우)
            if current_max_tokens is not None and current_max_tokens > 0:
                original_length = len(document)
                # utils.token_counter의 텍스트 자르기 함수 사용
                document = truncate_text_to_token_limit(
                    text=document,
                    max_tokens=current_max_tokens
                    # 사용할 모델 이름은 기본값(예: gpt-4) 또는 설정 가능
                )
                # 만약 문서 길이가 줄어들었다면 로그 남기기
                if len(document) < original_length:
                    self.logger.info(f"'{self.name}': 문서가 최대 토큰 수({current_max_tokens}) 제한에 맞춰 잘렸습니다.")

            # 최종 포맷팅된 문서 반환
            return {"document": document}

        except Exception as e:
            self.logger.error(f"'{self.name}': 문서 포맷팅 중 오류 발생: {str(e)}", exc_info=True)
            # 오류 발생 시 빈 문서와 에러 메시지 반환
            return {"document": "", "error": str(e)}


class ExtractStructuredDataNode(BaseNode):
    """
    LLM을 사용하여 비정형 텍스트에서 구조화된 데이터(JSON)를 추출하는 노드입니다.

    입력 텍스트와 추출할 데이터의 스키마(JSON 형식)를 받아, LLM에게 해당 스키마에 맞춰
    텍스트 내용을 요약/정리하여 JSON으로 출력하도록 요청합니다.
    특히 로컬 LLM은 JSON 출력이 불안정할 수 있으므로, 강력한 프롬프팅과
    LLM 응답에서 JSON 부분을 안정적으로 파싱하는 로직이 포함되어 있습니다.
    """

    def __init__(
        self,
        model_manager: ModelManager, # LLM 호출을 위한 ModelManager
        name: Optional[str] = None
    ):
        """
        ExtractStructuredDataNode를 초기화합니다.

        Args:
            model_manager: 사용할 ModelManager 객체. LLM 호출을 담당합니다.
            name: 노드의 고유 이름 (지정하지 않으면 자동 생성).
        """
        super().__init__(name=name)
        self.model_manager = model_manager
        self.logger.info(f"ExtractStructuredDataNode '{self.name}' 초기화 완료.")

    def _parse_json_from_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        LLM 응답 텍스트에서 JSON 객체를 파싱하여 딕셔너리로 반환합니다.

        LLM은 다양한 형식으로 JSON을 출력할 수 있습니다 (예: 마크다운 코드 블록 안에 넣거나,
        텍스트 설명과 함께 출력하거나, JSON 객체만 출력). 이 함수는 여러 가능성을
        처리하여 최대한 JSON 데이터를 추출하려고 시도합니다.

        Args:
            response_text: LLM이 반환한 전체 응답 문자열.

        Returns:
            파싱된 JSON 딕셔너리. 파싱에 완전히 실패하면 None을 반환합니다.
        """
        # 시도 1: JSON 마크다운 코드 블록 (```json ... ```) 찾기
        # re.DOTALL은 '.'이 개행 문자도 포함하게 하고, re.IGNORECASE는 대소문자 구분 안 함
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1) # 중괄호 안의 내용만 추출
            self.logger.debug(f"'{self.name}': 마크다운 코드 블록에서 JSON 추출 시도.")
            try:
                # 추출한 문자열을 JSON으로 파싱
                parsed_json = json.loads(json_str)
                self.logger.debug(f"'{self.name}': 마크다운 코드 블록 JSON 파싱 성공.")
                return parsed_json
            except json.JSONDecodeError as e:
                # 파싱 실패 시 경고 로그 남기고 다음 방법 시도
                self.logger.warning(f"'{self.name}': 마크다운 코드 블록 JSON 파싱 실패: {e}. 다른 방법 시도...")
                # 여기서 함수를 종료하지 않고 다음 파싱 방법을 시도합니다.

        # 시도 2: 문자열에서 첫 '{' 와 마지막 '}' 사이의 내용 추출 시도
        # (LLM이 코드 블록 없이 설명과 함께 JSON을 출력하는 경우)
        start_index = response_text.find('{')
        end_index = response_text.rfind('}')
        if start_index != -1 and end_index != -1 and start_index < end_index:
            json_str = response_text[start_index : end_index + 1]
            self.logger.debug(f"'{self.name}': '{...}' 패턴으로 JSON 추출 시도.")
            try:
                parsed_json = json.loads(json_str)
                self.logger.debug(f"'{self.name}': '{...}' 패턴 JSON 파싱 성공.")
                return parsed_json
            except json.JSONDecodeError as e:
                 # 파싱 실패 시 경고 로그 남기고 다음 방법 시도
                 self.logger.warning(f"'{self.name}': '{...}' 패턴 JSON 파싱 실패: {e}. 응답 전체 파싱 시도...")

        # 시도 3: 응답 텍스트 전체를 JSON으로 간주하고 파싱 시도
        # (LLM이 설명 없이 JSON 객체만 깔끔하게 출력한 경우)
        self.logger.debug(f"'{self.name}': 응답 전체를 JSON으로 파싱 시도.")
        try:
            # 앞뒤 공백 제거 후 파싱
            parsed_json = json.loads(response_text.strip())
            self.logger.debug(f"'{self.name}': 응답 전체 JSON 파싱 성공.")
            return parsed_json
        except json.JSONDecodeError as e:
            # 모든 방법 실패 시 에러 로그 남기고 None 반환
            self.logger.error(f"'{self.name}': 최종 JSON 파싱 실패. LLM 원본 응답(일부): '{response_text[:200]}...'")
            return None

    def process(self, inputs: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        입력 텍스트에서 주어진 스키마에 따라 구조화된 데이터를 추출합니다.

        Args:
            inputs (Dict[str, Any]): 노드 처리에 필요한 입력 데이터. 다음 키를 포함할 수 있습니다:
                - text (str): 구조화된 데이터를 추출할 원본 텍스트. 필수 항목입니다.
                - schema (Dict[str, Any]): 추출할 데이터의 JSON 스키마 정의. 필수 항목입니다.
                                          예: {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "number"}}}
                - schema_description (str, Optional): 스키마에 대한 추가적인 설명이나 지침 (LLM에게 전달).
                - extraction_prompt (str, Optional): 데이터 추출을 위한 전체 프롬프트를 직접 제공할 경우 사용.
                                                    제공하면 이 노드 내부의 프롬프트 생성 로직을 건너뜁니다.
            state (Dict[str, Any]): 현재 에이전트 상태. 사용되지 않음.

        Returns:
            Dict[str, Any]: 처리 결과. 다음 키를 포함합니다:
                - data (Dict[str, Any]): 추출된 구조화된 데이터 (JSON 딕셔너리). 추출/파싱 실패 시 빈 딕셔너리.
                - raw_response (str): LLM이 반환한 원본 응답 문자열 (JSON 파싱 전 상태, 디버깅용).
                - error (str, Optional): 처리 중 오류 발생 시 오류 메시지.
        """
        text = inputs.get("text") # 추출 대상 텍스트
        if not text:
            self.logger.warning(f"'{self.name}': 처리할 'text'가 입력에 없습니다.")
            return {"data": {}, "raw_response": "", "error": "텍스트가 제공되지 않았습니다."}

        schema = inputs.get("schema") # 추출할 JSON 스키마
        if not schema:
            self.logger.warning(f"'{self.name}': 데이터 추출을 위한 'schema'가 입력에 없습니다.")
            return {"data": {}, "raw_response": "", "error": "스키마가 제공되지 않았습니다."}

        schema_description = inputs.get("schema_description", "") # 스키마 부가 설명 (선택)
        extraction_prompt = inputs.get("extraction_prompt") # 직접 프롬프트 주입 (선택)

        try:
            # 1. 추출 프롬프트 생성 (직접 제공되지 않은 경우)
            if not extraction_prompt:
                # JSON 스키마를 보기 좋은 문자열로 변환 (들여쓰기 포함)
                try:
                    schema_str = json.dumps(schema, indent=2, ensure_ascii=False) # 한글 등 유니코드 문자 유지
                except TypeError as e:
                     # 스키마가 JSON으로 변환 불가능한 타입 포함 시 오류
                     self.logger.error(f"'{self.name}': 제공된 스키마를 JSON으로 변환할 수 없습니다: {e}")
                     return {"data": {}, "raw_response": "", "error": "스키마 직렬화 오류"}

                # 프롬프트 구성: 명확한 지시사항 + 스키마 + 텍스트 + 출력 유도
                # 로컬 모델이 JSON 형식을 잘 따르도록 지시를 강화합니다.
                prompt_lines = [
                    f"다음 텍스트에서 아래 JSON 스키마 형식에 정확히 맞는 데이터를 추출해주세요.",
                    f"응답은 반드시 유효한 JSON 객체 하나여야 합니다.", # 강조
                    f"JSON 객체 외에 어떠한 설명, 인사, 추가 텍스트도 포함하지 마세요.", # 강조
                    f"\nJSON 스키마:\n```json\n{schema_str}\n```" # 스키마를 코드 블록으로 명시
                ]
                if schema_description:
                    # 스키마 설명이 있으면 추가
                    prompt_lines.append(f"\n스키마 추가 설명: {schema_description}")

                prompt_lines.append(f"\n추출 대상 텍스트:\n```text\n{text}\n```") # 원본 텍스트 명시
                prompt_lines.append(f"\n추출된 JSON 데이터:") # 모델이 바로 JSON을 시작하도록 유도

                extraction_prompt = "\n".join(prompt_lines) # 줄바꿈으로 합쳐 최종 프롬프트 생성
                self.logger.debug(f"'{self.name}': 데이터 추출 프롬프트 생성 완료.")

            # 2. LLM 호출하여 데이터 추출 시도
            self.logger.debug(f"'{self.name}': 데이터 추출 LLM 호출 시작.")
            # 구조화된 데이터 추출에는 보통 낮은 온도가 권장됩니다 (창의성 < 정확성).
            llm_result = self.model_manager.generate(
                prompt=extraction_prompt,
                # 시스템 프롬프트로 모델 역할 명확화
                system_prompt="You are a precise data extraction assistant. Your task is to extract structured data from the given text according to the provided JSON schema. Respond ONLY with the valid JSON object and nothing else.",
                temperature=0.1, # 낮은 온도로 설정하여 일관성 및 정확성 유도
                # max_tokens는 예상되는 JSON 크기에 맞춰 적절히 설정 필요
                # 너무 작으면 JSON이 중간에 잘릴 수 있음
            )

            # 3. LLM 응답 파싱
            raw_response = ""
            extracted_data = None
            if isinstance(llm_result, str): # 일반 응답일 경우
                raw_response = llm_result
                extracted_data = self._parse_json_from_response(raw_response)
            # 스트리밍 응답에 대한 처리는 필요 시 추가 구현
            # elif isinstance(llm_result, Generator):
            #     raw_response = "".join(list(llm_result))
            #     extracted_data = self._parse_json_from_response(raw_response)
            else: # 예상치 못한 타입 처리
                 raw_response = str(llm_result) # 로깅을 위해 문자열 변환
                 self.logger.error(f"'{self.name}': LLM 응답이 예상 타입(str 또는 Generator)이 아닙니다: {type(llm_result)}")

            # 4. 파싱 결과 반환
            if extracted_data:
                # JSON 파싱 성공
                self.logger.info(f"'{self.name}': 구조화된 데이터 추출 및 파싱 성공.")
                return {"data": extracted_data, "raw_response": raw_response}
            else:
                # JSON 파싱 실패
                self.logger.error(f"'{self.name}': LLM 응답에서 유효한 JSON 데이터를 추출하지 못했습니다.")
                # 실패 시 빈 데이터와 원본 응답, 에러 메시지 반환
                return {"data": {}, "raw_response": raw_response, "error": "JSON 데이터 파싱 실패"}

        except Exception as e:
            # LLM 호출 또는 기타 처리 중 예외 발생
            self.logger.error(f"'{self.name}': 구조화 데이터 추출 중 오류 발생: {str(e)}", exc_info=True)
            # 오류 발생 시 빈 데이터와 에러 메시지 반환 (raw_response는 있을 수도 없을 수도 있음)
            raw_response_on_error = locals().get("raw_response", "") # 오류 전 raw_response가 있으면 포함
            return {"data": {}, "raw_response": raw_response_on_error, "error": str(e)}


class NotificationAnalysisNode(BaseNode):
    """
    장비 알림(Notification)을 분석하는 특화된 노드입니다.

    주어진 장비 알림 정보(메시지, ID 등)를 바탕으로, 관련 지식을 벡터 저장소에서 검색하고,
    LLM을 활용하여 문제 상황을 분석합니다. 최종적으로 사람이 이해하기 쉬운 분석 텍스트와
    기계 처리가 용이한 구조화된 데이터(JSON)를 생성하는 것을 목표로 합니다.
    """

    def __init__(
        self,
        model_manager: ModelManager, # LLM 호출을 위한 ModelManager 인스턴스
        vector_store: VectorStore,   # 관련 지식 검색을 위한 VectorStore 인스턴스
        name: Optional[str] = None,
        template_file: Optional[str] = None # 프롬프트 템플릿 파일 경로 (선택 사항)
    ):
        """
        NotificationAnalysisNode를 초기화합니다.

        Args:
            model_manager: 사용할 ModelManager 객체 (LLM 호출 및 임베딩 생성 담당).
            vector_store: 사용할 VectorStore 객체 (관련 문서 검색 담당).
            name: 노드의 고유 이름 (지정하지 않으면 자동 생성).
            template_file: 분석 및 구조화 데이터 추출 프롬프트에 사용할 템플릿 파일 경로.
                           지정하지 않으면 기본 경로(`config/prompt_templates.yaml`)를 사용 시도합니다.
        """
        super().__init__(name=name)
        self.model_manager = model_manager
        self.vector_store = vector_store
        self.template_file = template_file
        self.logger.info(f"NotificationAnalysisNode '{self.name}' 초기화 완료.")

    def process(self, inputs: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        입력받은 장비 알림 정보를 분석합니다.

        처리 절차:
        1.  **관련 지식 검색**: 알림 메시지와 장비 ID를 조합하여 VectorStore에서 유사한 과거 사례나 기술 문서를 검색합니다.
        2.  **컨텍스트 포맷팅**: 검색된 문서들을 LLM이 이해하기 쉬운 형식의 컨텍스트 문자열로 만듭니다.
        3.  **분석 프롬프트 생성**: 알림 정보, 검색된 컨텍스트, 그리고 미리 정의된 프롬프트 템플릿을 결합하여 LLM에게 전달할 최종 분석 프롬프트를 생성합니다.
        4.  **LLM 분석**: 생성된 프롬프트를 LLM에게 보내 알림에 대한 상세 분석(문제 원인, 심각도, 권장 조치 등) 텍스트를 얻습니다.
        5.  **구조화 데이터 추출**: LLM이 생성한 분석 텍스트에서 핵심 정보를 미리 정의된 스키마(JSON 형식)에 맞춰 추출합니다. (별도의 LLM 호출 또는 정규식/파싱 로직 사용 가능)

        Args:
            inputs (Dict[str, Any]): 노드 처리에 필요한 입력 데이터. 다음 키들을 포함하는 것이 좋습니다:
                - equipment_id (str): 알림이 발생한 장비의 식별자. 검색 및 분석 정확도 향상에 사용됩니다.
                - notification_type (str): 알림의 유형 (예: 'Warning', 'Error', 'Info'). 분석 시 참고 정보로 사용됩니다.
                - message (str): 실제 알림 메시지 내용. 분석의 핵심 입력입니다. 필수 항목입니다.
                - timestamp (str, Optional): 알림 발생 시간 정보.
                - additional_data (Any, Optional): 센서 값, 로그 등 추가적인 관련 데이터. 분석 프롬프트에 포함될 수 있습니다.
                - top_k_search (int, Optional): 관련 지식 검색 시 가져올 최대 문서 수. 기본값: 5.
                - analysis_template_name (str, Optional): 분석 프롬프트 생성에 사용할 템플릿 이름 (템플릿 파일 내 정의). 기본값: 'notification_analysis'.
                - extraction_schema (Dict, Optional): 구조화 데이터 추출 시 사용할 JSON 스키마. 제공되지 않으면 노드 내부에 정의된 기본 스키마를 사용합니다.
            state (Dict[str, Any]): 현재 에이전트 상태. 이 노드에서는 사용되지 않으나, 필요 시 상태 기반 로직 추가 가능.

        Returns:
            Dict[str, Any]: 처리 결과 딕셔너리. 다음 키를 포함합니다:
                - analysis (str): LLM이 생성한 자연어 분석 결과 텍스트.
                - structured_data (Dict[str, Any]): 분석 결과에서 추출된 구조화된 데이터 (JSON 형식 딕셔너리). 추출 실패 시 빈 딕셔너리.
                - references (List[Dict]): 분석에 참조된 관련 지식 문서 목록 (VectorStore 검색 결과).
                - error (str, Optional): 노드 처리 중 오류가 발생한 경우 해당 오류 메시지.
        """
        # 입력 값 가져오기 (없으면 기본값 사용)
        equipment_id = inputs.get("equipment_id", "Unknown")
        notification_type = inputs.get("notification_type", "Unknown")
        message = inputs.get("message") # 알림 메시지는 필수
        timestamp = inputs.get("timestamp", "")
        additional_data = inputs.get("additional_data", "")
        top_k_search = inputs.get("top_k_search", 5) # 벡터 검색 결과 수
        analysis_template_name = inputs.get("analysis_template_name", "notification_analysis") # 분석 프롬프트 템플릿 이름
        extraction_schema = inputs.get("extraction_schema") # 사용자 정의 추출 스키마 (선택)

        # 필수 입력인 'message' 확인
        if not message:
            self.logger.warning(f"'{self.name}': 처리할 'message'가 입력에 없습니다.")
            return {"analysis": "", "structured_data": {}, "references": [], "error": "알림 메시지가 제공되지 않았습니다."}

        self.logger.info(f"'{self.name}': 장비 '{equipment_id}'의 알림 분석 시작...")

        try:
            # === 단계 1: 관련 지식 검색 (Vector Search) ===
            self.logger.debug(f"'{self.name}': 관련 지식 검색 시작 (VectorStore)")
            # 검색 쿼리 생성 (장비 ID와 메시지를 조합하여 구체화)
            search_query = f"장비: {equipment_id}, 문제 내용: {message}"
            # ModelManager를 사용하여 검색 쿼리의 임베딩 벡터 생성
            query_embedding = self.model_manager.create_embeddings([search_query])[0]

            # 메타데이터 필터 설정 (equipment_id가 'Unknown'이 아닐 때만 필터링 시도)
            filter_metadata = {"equipment_id": equipment_id} if equipment_id != "Unknown" else None

            # VectorStore에 쿼리 실행
            search_results = self.vector_store.query(
                query_embedding=query_embedding,
                top_k=top_k_search,
                filter_metadata=filter_metadata
            )
            self.logger.debug(f"'{self.name}': 관련 지식 {len(search_results)}개 검색 완료.")

            # === 단계 2: 검색 결과를 컨텍스트로 포맷팅 ===
            # DocumentFormatNode를 직접 생성/사용하거나, 여기서 간단히 포맷팅 로직 구현
            # 여기서는 간단히 직접 구현하는 예시
            context_parts = []
            if search_results: # 검색 결과가 있을 때만 컨텍스트 생성
                self.logger.debug(f"'{self.name}': 검색 결과를 컨텍스트로 포맷팅 시작.")
                for i, result in enumerate(search_results):
                    # 각 검색 결과를 보기 좋은 형식으로 만듭니다.
                    context_parts.append(
                        f"참고 자료 {i+1}:\n"
                        f"- 내용 요약: {result.get('text', '내용 없음')[:150]}...\n" # 내용은 일부만 표시 (길이 조절 가능)
                        # f"- 전체 내용: {result.get('text', '내용 없음')}\n" # 전체 내용을 포함할 수도 있음
                        f"- 관련 장비 (추정): {result.get('metadata', {}).get('equipment_id', '알 수 없음')}\n"
                        f"- 유사도 점수: {result.get('score', 0.0):.4f}" # 소수점 4자리까지 표시
                    )
                knowledge_context = "\n\n".join(context_parts)
            else:
                knowledge_context = "관련된 과거 사례나 기술 문서를 찾지 못했습니다." # 결과 없을 때 메시지
            self.logger.debug(f"'{self.name}': 컨텍스트 포맷팅 완료.")

            # === 단계 3: 분석 프롬프트 생성 및 LLM 호출 ===
            self.logger.debug(f"'{self.name}': 분석 프롬프트 생성 시작.")
            # 템플릿 파일에서 분석용 프롬프트 템플릿 로드
            analysis_template = load_template(
                template_name=analysis_template_name,
                template_type="user_prompts", # 사용자 프롬프트 타입으로 가정
                template_file=self.template_file # 노드 초기화 시 받은 템플릿 파일 경로 사용
            )
            # 템플릿 로드 실패 시 오류 처리
            if not analysis_template:
                 raise ValueError(f"분석 프롬프트 템플릿 '{analysis_template_name}'을 로드할 수 없습니다.")

            # 템플릿에 알림 정보 변수 채우기
            analysis_prompt_base = format_prompt(
                analysis_template,
                variables={
                    "equipment_id": equipment_id,
                    "notification_type": notification_type,
                    "message": message,
                    "timestamp": timestamp,
                    "additional_data": str(additional_data) # 추가 데이터는 문자열로 변환하여 전달
                }
            )

            # 생성된 컨텍스트(검색 결과)를 프롬프트에 추가
            # LLM에게 컨텍스트 정보임을 명확히 알려주는 것이 좋습니다.
            analysis_prompt = f"{analysis_prompt_base}\n\n" \
                              f"--- 관련 지식 정보 (참고용) ---\n" \
                              f"{knowledge_context}\n" \
                              f"---------------------------------\n\n" \
                              f"**분석 요청**: 위 알림 내용과 관련 지식 정보를 바탕으로, 문제 상황을 상세히 분석하고, " \
                              f"가능한 원인, 문제의 심각도, 그리고 구체적인 권장 조치 사항을 단계별로 설명해주세요."

            # 분석 작업에 적합한 시스템 프롬프트 가져오기
            # 'technical_diagnosis' 템플릿이 있으면 사용하고, 없으면 'default' 사용
            system_prompt = get_system_prompt("technical_diagnosis", self.template_file) or \
                            get_system_prompt("default", self.template_file)

            self.logger.debug(f"'{self.name}': 분석 LLM 호출 시작.")
            # LLM 호출하여 분석 텍스트 생성
            # generate 함수는 문자열 또는 제너레이터를 반환할 수 있음 (LLMNode와 동일하게 처리)
            analysis_response = self.model_manager.generate(
                prompt=analysis_prompt,
                system_prompt=system_prompt,
                # 필요에 따라 temperature, max_tokens 등 추가 파라미터 전달
            )

            # 응답 처리 (스트리밍 여부에 따라)
            analysis_text = ""
            if isinstance(analysis_response, str):
                analysis_text = analysis_response
                self.logger.info(f"'{self.name}': 알림 분석 텍스트 생성 완료 (길이: {len(analysis_text)}자).")
            elif isinstance(analysis_response, Generator):
                # 스트리밍 응답이면 모두 모아서 텍스트로 만듦
                self.logger.info(f"'{self.name}': 알림 분석 스트리밍 응답 수신. 전체 내용 취합 중...")
                analysis_text = "".join(list(analysis_response))
                self.logger.info(f"'{self.name}': 알림 분석 스트리밍 완료 (길이: {len(analysis_text)}자).")
            else:
                # 예상치 못한 타입이면 경고 로깅
                self.logger.warning(f"'{self.name}': 분석 LLM 결과가 예상 타입(str 또는 Generator)이 아닙니다: {type(analysis_response)}")
                analysis_text = str(analysis_response) # 문자열로 강제 변환 시도

            # === 단계 4: 구조화된 데이터 추출 ===
            self.logger.debug(f"'{self.name}': 분석 결과에서 구조화된 데이터 추출 시작.")
            # 추출할 데이터의 기본 스키마 정의 (입력에서 extraction_schema가 제공되지 않은 경우 사용)
            if not extraction_schema:
                 extraction_schema = {
                     "type": "object",
                     "properties": {
                         "issue_summary": {"type": "string", "description": "분석된 문제 상황 요약"},
                         "severity": {"type": "string", "enum": ["낮음", "보통", "높음", "심각"], "description": "판단된 문제 심각도 (Low, Medium, High, Critical에 대응)"},
                         "possible_causes": {"type": "array", "items": {"type": "string"}, "description": "추정되는 가능한 원인 목록"},
                         "recommended_actions": {"type": "array", "items": {"type": "string"}, "description": "권장되는 조치 사항 목록 (구체적으로)"},
                         "confidence_score": {"type": "number", "description": "분석 결과의 전반적인 신뢰도 점수 (0.0 ~ 1.0, LLM이 판단)"} # 예시 필드 추가
                     },
                     "required": ["issue_summary", "severity", "recommended_actions"] # 필수 필드 지정
                 }
                 self.logger.debug(f"'{self.name}': 기본 추출 스키마 사용.")
            else:
                 self.logger.debug(f"'{self.name}': 입력으로 제공된 사용자 정의 추출 스키마 사용.")

            # ExtractStructuredDataNode를 인스턴스화하여 데이터 추출 수행
            # 별도의 노드로 분리하는 대신, 여기서 직접 해당 노드의 기능을 사용합니다.
            extraction_node = ExtractStructuredDataNode(self.model_manager, name=f"{self.name}_extractor")
            extraction_input = {
                "text": analysis_text, # LLM이 생성한 분석 텍스트를 입력으로 제공
                "schema": extraction_schema # 정의된 스키마 사용
            }
            # 추출 노드 실행 (state는 필요 없으므로 빈 딕셔너리 전달 가능)
            extraction_result = extraction_node.process(extraction_input, {})

            structured_data = extraction_result.get("data", {}) # 추출된 데이터 (실패 시 빈 딕셔너리)
            extraction_error = extraction_result.get("error") # 추출 중 발생한 오류

            if extraction_error:
                 # 추출 실패 시 경고 로깅
                 self.logger.warning(f"'{self.name}': 구조화 데이터 추출 실패: {extraction_error}. 원본 응답: '{extraction_result.get('raw_response', '')[:100]}...'")
                 # 오류가 발생했어도 분석 텍스트는 유효하므로 계속 진행

            self.logger.info(f"'{self.name}': 알림 분석 및 데이터 추출 완료.")

            # === 최종 결과 반환 ===
            return {
                "analysis": analysis_text,             # LLM 분석 결과 (자연어)
                "structured_data": structured_data,    # 추출된 데이터 (JSON/딕셔너리)
                "references": search_results,          # 분석에 참조된 문서 목록
                "error": extraction_error              # 데이터 추출 단계 오류 (있을 경우)
            }

        except Exception as e:
            # 노드 처리 중 예상치 못한 오류 발생 시
            self.logger.error(f"'{self.name}': 알림 분석 전체 과정 중 오류 발생: {str(e)}", exc_info=True)
            # 실패 결과 반환
            return {"analysis": "", "structured_data": {}, "references": [], "error": f"알림 분석 실패: {str(e)}"}