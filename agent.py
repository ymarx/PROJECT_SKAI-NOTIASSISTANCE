"""
SKAI-NotiAssistance의 메인 에이전트 구현 모듈입니다.

이 모듈은 산업 장비 알림 분석을 위한 핵심 에이전트 클래스인
`NotiAssistanceAgent`를 정의합니다. 이 에이전트는 다양한 처리 노드(Node)들을
조율(Orchestration)하여 입력된 작업을 수행하고 결과를 반환합니다.
"""

import os
import time
from typing import Any, Dict, List, Optional, Union, Callable

# 프로젝트 내 다른 모듈 임포트
# lib 폴더: 기본 클래스, 모델 관리자, 벡터 저장소, 유틸리티
# state.py: 상태 관리 클래스
# utils 폴더: 로거, 토큰 카운터 등
# nodes.py: 실제 작업을 수행하는 노드 클래스들
try:
    from lib.base import BaseAgent, BaseNode
    from lib.model_manager import ModelManager
    from lib.vector_store import VectorStore
    from lib.utils import load_yaml, merge_configs
    from state import State # 수정된 State 클래스 임포트
    from utils.logger import get_logger
    from utils.token_counter import count_tokens
    # nodes 모듈에서 정의된 모든 노드 클래스 임포트 (필요에 따라 선택적 임포트 가능)
    from nodes import (
        LLMNode,
        PromptTemplateNode,
        VectorSearchNode,
        DocumentFormatNode,
        ExtractStructuredDataNode,
        NotificationAnalysisNode
    )
except ImportError as e:
    # 임포트 실패 시 경로 문제 해결 시도 (상위 디렉토리 추가)
    import sys
    import logging
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from lib.base import BaseAgent, BaseNode
        from lib.model_manager import ModelManager
        from lib.vector_store import VectorStore
        from lib.utils import load_yaml, merge_configs
        from state import State
        from utils.logger import get_logger
        from utils.token_counter import count_tokens
        from nodes import (
            LLMNode, PromptTemplateNode, VectorSearchNode, DocumentFormatNode,
            ExtractStructuredDataNode, NotificationAnalysisNode
        )
        logging.warning(f"상대 경로 임포트 실패, 경로 조정 시도 성공: {e}")
    except ImportError as final_e:
        logging.error(f"에이전트 필수 모듈 임포트 실패! 실행 경로 및 프로젝트 구조를 확인하세요: {final_e}")
        raise


logger = get_logger(__name__) # 이 모듈용 로거 생성


class NotiAssistanceAgent(BaseAgent):
    """
    SKAI-NotiAssistance의 핵심 에이전트 클래스입니다.

    이 에이전트는 산업 장비 알림(Notification)을 분석하고, 관련 정보를 검색하며,
    유지보수 추천 등의 작업을 수행합니다. `nodes.py`에 정의된 다양한 처리 노드들을
    조합하여 복잡한 작업을 처리합니다.

    주요 기능:
    - 설정 파일 기반 초기화 (모델, 벡터 저장소 등)
    - 다양한 작업 유형 처리 (알림 분석, 유사 사례 검색, 직접 질의 등)
    - 상태 관리 (`State` 클래스 활용)
    - 처리 파이프라인(노드 조합) 관리
    - 벡터 저장소에 문서 추가 기능
    """

    def __init__(
        self,
        config_path: Optional[str] = None, # 설정 파일 경로 (선택 사항)
        name: str = "SKAI-NotiAssistance"  # 에이전트 이름 (기본값)
    ):
        """
        NotiAssistanceAgent를 초기화합니다.

        1.  `BaseAgent`를 초기화합니다 (이름, 로거 등).
        2.  설정 파일(`config_path`)을 로드합니다. 없으면 기본 설정을 찾습니다.
        3.  에이전트 상태(`State` 객체)를 초기화합니다.
        4.  설정에 따라 핵심 컴포넌트(`ModelManager`, `VectorStore`)를 초기화합니다.
        5.  사용할 처리 노드들을 생성하고 에이전트에 등록합니다 (`_setup_pipeline`).

        Args:
            config_path: 에이전트 설정을 담은 YAML 파일 경로 (선택 사항).
                           LLM 설정, 벡터 저장소 설정 등이 포함될 수 있습니다.
                           제공되지 않으면 `config/model_config.yaml`을 기본값으로 사용 시도합니다.
            name: 에이전트의 이름.
        """
        # 1. BaseAgent 초기화 (이름, ID, 로거, config 로드 등)
        super().__init__(name=name, config_path=config_path)
        self.logger.info(f"에이전트 '{self.name}' 초기화 시작...")

        # 2. 에이전트 상태 관리 객체 생성
        # `State` 클래스는 대화 기록, 중간 결과 등을 저장합니다.
        self.agent_state = State()
        self.logger.debug("에이전트 상태(State) 객체 생성 완료.")

        # 3. 기본 설정 파일 처리
        # config_path가 제공되지 않았거나, 해당 경로에 파일이 없어 self.config가 비어있는 경우
        if not self.config:
            # 프로젝트 루트 기준 기본 설정 파일 경로
            default_config_path = os.path.join(
                os.path.dirname(__file__), "config", "model_config.yaml"
            )
            self.logger.warning(f"제공된 설정 파일 경로가 없거나 유효하지 않습니다. 기본 설정 파일 로드 시도: {default_config_path}")
            if os.path.exists(default_config_path):
                # load_yaml 유틸리티 함수 사용 (lib/utils.py 에 있을 것으로 예상)
                self.config = load_yaml(default_config_path)
                self.logger.info(f"기본 설정 로드 완료: {default_config_path}")
            else:
                # 기본 설정 파일도 없으면 빈 설정으로 진행 (오류 발생 가능성 있음)
                self.logger.error("기본 설정 파일도 찾을 수 없습니다! 에이전트 기능이 제한될 수 있습니다.")
                self.config = {} # 빈 딕셔너리로 초기화

        # 4. 핵심 컴포넌트 초기화 (ModelManager, VectorStore)
        # 설정(self.config)을 바탕으로 컴포넌트를 설정합니다.
        try:
            self._init_components()
        except Exception as e:
            # 컴포넌트 초기화 실패는 심각한 문제이므로 에러 로깅 후 다시 발생시킴
            self.logger.critical(f"핵심 컴포넌트 초기화 실패! 에이전트를 사용할 수 없습니다. 오류: {e}", exc_info=True)
            raise

        # 5. 처리 파이프라인(노드) 설정
        # 사용할 노드들을 생성하고 에이전트에 등록합니다.
        try:
            self._setup_pipeline()
        except Exception as e:
            self.logger.critical(f"처리 파이프라인 설정 실패! 에이전트를 사용할 수 없습니다. 오류: {e}", exc_info=True)
            raise

        self.logger.info(f"에이전트 '{self.name}' 초기화 완료.")

    def _init_components(self):
        """
        에이전트의 핵심 컴포넌트(ModelManager, VectorStore)를 초기화합니다.
        `self.config` 딕셔너리에서 각 컴포넌트에 필요한 설정을 추출하여 전달합니다.
        """
        self.logger.debug("핵심 컴포넌트 초기화 시작...")
        # LLM 설정 부분 가져오기 (없으면 빈 딕셔너리)
        llm_config = self.config.get("llm", {})
        if not llm_config:
            self.logger.warning("LLM 설정('llm')이 설정 파일에 없습니다. 기본 설정으로 ModelManager 초기화 시도.")
        # ModelManager 초기화 (수정된 ModelManager 사용)
        self.model_manager = ModelManager(llm_config)
        self.logger.info(f"ModelManager 초기화 완료 (Provider: {self.model_manager.provider})")

        # 벡터 저장소 설정 부분 가져오기 (없으면 빈 딕셔너리)
        vector_config = self.config.get("vector_store", {})
        if not vector_config:
             self.logger.warning("벡터 저장소 설정('vector_store')이 설정 파일에 없습니다. 기본 설정으로 VectorStore 초기화 시도.")
        # VectorStore 초기화 (수정된 VectorStore 사용)
        self.vector_store = VectorStore(vector_config)
        self.logger.info(f"VectorStore 초기화 완료 (Provider: {self.vector_store.provider}, Collection: {self.vector_store.collection_name})")
        self.logger.debug("핵심 컴포넌트 초기화 완료.")

    def _setup_pipeline(self):
        """
        에이전트가 사용할 처리 노드(Node)들을 생성하고 등록합니다.

        `nodes.py` 에 정의된 노드 클래스들을 사용하여 필요한 노드 인스턴스를 만들고,
        `self.add_node()` 메서드 (BaseAgent로부터 상속)를 사용하여 에이전트에 추가합니다.
        이렇게 등록된 노드들은 작업 처리 메서드(`_run_...`)에서 이름으로 접근하여 사용할 수 있습니다.
        """
        self.logger.debug("처리 파이프라인(노드) 설정 시작...")

        # 템플릿 파일 경로 설정 (프로젝트 루트 기준)
        template_file_path = os.path.join(
            os.path.dirname(__file__), "config", "prompt_templates.yaml"
        )
        if not os.path.exists(template_file_path):
             self.logger.warning(f"프롬프트 템플릿 파일 경로를 찾을 수 없습니다: {template_file_path}")
             template_file_path = None # 파일 없으면 None 전달

        # 1. LLM 노드 생성 및 추가
        self.add_node(
            LLMNode( # nodes.py의 LLMNode 클래스 사용
                model_manager=self.model_manager, # 초기화된 ModelManager 전달
                name="llm_node" # 노드 이름 지정 (호출 시 사용)
            )
        )
        self.logger.debug("LLMNode 추가 완료.")

        # 2. 프롬프트 템플릿 노드 생성 및 추가
        self.add_node(
            PromptTemplateNode( # nodes.py의 PromptTemplateNode 클래스 사용
                template_file=template_file_path, # 템플릿 파일 경로 전달
                name="prompt_template_node"
            )
        )
        self.logger.debug("PromptTemplateNode 추가 완료.")

        # 3. 벡터 검색 노드 생성 및 추가
        self.add_node(
            VectorSearchNode( # nodes.py의 VectorSearchNode 클래스 사용
                vector_store=self.vector_store,     # 초기화된 VectorStore 전달
                model_manager=self.model_manager, # 쿼리 임베딩 생성 위해 ModelManager 전달
                name="vector_search_node"
            )
        )
        self.logger.debug("VectorSearchNode 추가 완료.")

        # 4. 문서 포맷팅 노드 생성 및 추가
        self.add_node(
            DocumentFormatNode( # nodes.py의 DocumentFormatNode 클래스 사용
                name="document_format_node",
                # max_tokens=2000 # 필요 시 최대 토큰 수 제한 설정
            )
        )
        self.logger.debug("DocumentFormatNode 추가 완료.")

        # 5. 구조화 데이터 추출 노드 생성 및 추가
        self.add_node(
            ExtractStructuredDataNode( # nodes.py의 ExtractStructuredDataNode 클래스 사용
                model_manager=self.model_manager, # LLM 호출 위해 ModelManager 전달
                name="extract_data_node"
            )
        )
        self.logger.debug("ExtractStructuredDataNode 추가 완료.")

        # 6. 알림 분석 특화 노드 생성 및 추가 (여러 노드를 내부적으로 활용)
        self.add_node(
            NotificationAnalysisNode( # nodes.py의 NotificationAnalysisNode 클래스 사용
                model_manager=self.model_manager,
                vector_store=self.vector_store,
                template_file=template_file_path,
                name="notification_analysis_node"
            )
        )
        self.logger.debug("NotificationAnalysisNode 추가 완료.")

        self.logger.info("처리 파이프라인(노드) 설정 완료.")

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        입력(inputs)을 받아 에이전트의 주 워크플로우를 실행합니다.

        1.  현재 상태 스냅샷 저장 (필요 시 복원 위함).
        2.  입력 데이터를 에이전트 상태에 기록.
        3.  입력의 'task' 키를 확인하여 수행할 작업 결정.
        4.  해당 작업에 맞는 내부 메서드(`_run_...`) 호출.
        5.  작업 결과를 상태에 기록하고 최종 결과 반환.
        6.  오류 발생 시 오류 정보 기록 및 반환.

        Args:
            inputs (Dict[str, Any]): 에이전트 실행에 필요한 입력 데이터.
                - task (str): 수행할 작업 유형 (예: 'notification_analysis', 'similar_cases', 'direct_query'). 필수.
                - 기타 작업별 필요한 입력 데이터 (예: 'message', 'equipment_id', 'query' 등).

        Returns:
            Dict[str, Any]: 에이전트 작업 실행 결과. 보통 작업별 결과 데이터와 함께
                            'status' ('completed' 또는 'error'), 'execution_time' 등을 포함합니다.
                            오류 발생 시 'error' 키에 오류 메시지가 포함됩니다.
        """
        self.logger.info(f"에이전트 실행 시작 (입력 작업: {inputs.get('task', '알 수 없음')})")
        # 1. 상태 스냅샷 생성 (작업 시작 전 상태 저장)
        self.agent_state.snapshot()

        # 2. 입력 데이터와 시작 시간을 상태에 기록
        start_run_time = time.time()
        self.agent_state.set("current_inputs", inputs) # 입력 저장
        self.agent_state.set("start_time", start_run_time) # 시작 시간 기록

        # 3. 수행할 작업 유형 결정
        task = inputs.get("task")
        if not task:
            error_msg = "입력에 'task'가 지정되지 않았습니다."
            self.logger.error(error_msg)
            return self._handle_error(error_msg, start_run_time)

        try:
            # 4. 작업 유형에 따라 적절한 내부 메서드 호출하여 작업 수행
            result = {}
            if task == "notification_analysis":
                self.logger.info("작업 유형: 알림 분석")
                result = self._run_notification_analysis(inputs)
            elif task == "similar_cases":
                self.logger.info("작업 유형: 유사 사례 검색")
                result = self._run_similar_cases_search(inputs)
            elif task == "maintenance_recommendation":
                 self.logger.info("작업 유형: 유지보수 추천")
                 result = self._run_maintenance_recommendation(inputs)
            elif task == "direct_query":
                self.logger.info("작업 유형: 직접 질의")
                result = self._run_direct_query(inputs)
            # --- 새로운 작업 유형 추가 시 여기에 elif 추가 ---
            # elif task == "new_task_type":
            #     result = self._run_new_task(inputs)
            else:
                # 지원하지 않는 작업 유형 처리
                error_msg = f"알 수 없거나 지원하지 않는 작업 유형입니다: {task}"
                self.logger.error(error_msg)
                return self._handle_error(error_msg, start_run_time)

            # 5. 작업 결과 처리 및 반환
            end_run_time = time.time()
            execution_time = end_run_time - start_run_time
            result["execution_time"] = execution_time # 실행 시간 추가
            result["status"] = result.get("status", "completed") # 상태 기본값 completed

            # 최종 결과를 상태에도 저장
            self.agent_state.set("last_result", result)
            self.agent_state.set("end_time", end_run_time)
            self.agent_state.set("last_status", result["status"])

            self.logger.info(f"작업 '{task}' 완료 ({result['status']}). 총 실행 시간: {execution_time:.2f}초")
            return result

        except Exception as e:
            # 6. 작업 실행 중 예외 발생 처리
            self.logger.error(f"작업 '{task}' 실행 중 예기치 않은 오류 발생: {str(e)}", exc_info=True)
            return self._handle_error(str(e), start_run_time)

    def _handle_error(self, error_message: str, start_time: float) -> Dict[str, Any]:
        """오류 발생 시 공통 처리 로직"""
        end_time = time.time()
        execution_time = end_time - start_time
        error_result = {
            "error": error_message,
            "status": "error",
            "execution_time": execution_time
        }
        # 오류 결과를 상태에 기록
        self.agent_state.set("last_result", error_result)
        self.agent_state.set("end_time", end_time)
        self.agent_state.set("last_status", "error")
        return error_result

    # --- 작업별 실행 메서드 ---
    # 각 메서드는 특정 작업을 수행하기 위해 필요한 노드들을 순서대로 호출하고
    # 노드 간 데이터 흐름을 관리합니다.

    def _run_notification_analysis(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        알림 분석 작업을 수행하는 워크플로우입니다.

        흐름:
        1. `NotificationAnalysisNode`를 호출하여 전체 분석 수행.
           (이 노드 내부에 벡터 검색, LLM 분석, 데이터 추출 로직이 포함됨)

        Args:
            inputs: `run` 메서드에서 받은 입력 딕셔너리.
                    `NotificationAnalysisNode`가 요구하는 키들을 포함해야 합니다.
                    (message, equipment_id 등)

        Returns:
            분석 결과 딕셔너리 ('analysis', 'structured_data', 'references', 'error' 키 포함).
        """
        self.logger.debug("알림 분석 워크플로우 시작...")
        # 등록된 'notification_analysis_node' 가져오기
        notification_node = self.nodes.get("notification_analysis_node")
        if not notification_node:
             raise RuntimeError("필수 노드 'notification_analysis_node'가 에이전트에 등록되지 않았습니다.")

        # 노드 실행 (입력과 현재 상태 전달)
        # NotificationAnalysisNode가 복잡한 처리를 내부적으로 수행합니다.
        result = notification_node.process(inputs, self.agent_state.state)
        self.logger.debug("알림 분석 워크플로우 완료.")

        # 노드 결과에 status 추가 (오류 발생 시 error 키 존재)
        result["status"] = "error" if result.get("error") else "completed"
        return result

    def _run_similar_cases_search(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        유사 사례 검색 작업을 수행하는 워크플로우입니다.

        흐름:
        1. `VectorSearchNode`: 입력된 문제 설명으로 벡터 저장소 검색.
        2. `DocumentFormatNode`: 검색 결과를 LLM 입력용 컨텍스트 문서로 포맷팅.
        3. `LLMNode`: 포맷팅된 문서와 문제 설명을 바탕으로 유사 사례 분석 요청.

        Args:
            inputs: `run` 메서드에서 받은 입력 딕셔너리.
                    'equipment_id', 'issue_description' 키가 필요합니다.

        Returns:
            유사 사례 검색 및 분석 결과 딕셔너리 ('similar_cases', 'analysis', 'error' 키 포함).
        """
        self.logger.debug("유사 사례 검색 워크플로우 시작...")
        # 필요한 노드 가져오기
        vector_search_node = self.nodes.get("vector_search_node")
        doc_format_node = self.nodes.get("document_format_node")
        llm_node = self.nodes.get("llm_node")
        prompt_template_node = self.nodes.get("prompt_template_node") # 