"""
SKAI-NotiAssistance 에이전트의 상태 관리 모듈입니다.

이 모듈은 에이전트 실행 중 필요한 모든 상태 정보를 저장하고 관리하는
State 클래스를 제공합니다. 대화 기록, 작업 컨텍스트, 중간 결과,
메타데이터 등을 유연하게 저장하고 접근할 수 있습니다.
"""

import os
import time
import json
import copy # 상태 복제를 위해 사용 (snapshot, revert)
from typing import Any, Dict, List, Optional, Union, Callable

# utils 패키지의 로거를 가져옵니다.
# 프로젝트 구조에 따라 상대 경로가 달라질 수 있습니다.
# 만약 state.py가 프로젝트 루트에 있다면 from utils.logger import get_logger
try:
    from utils.logger import get_logger
except ImportError:
    # utils.logger를 찾을 수 없는 경우, 표준 로깅 사용
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.warning("utils.logger 를 찾을 수 없어 표준 로깅을 사용합니다.")


class State:
    """
    에이전트의 상태를 관리하는 클래스입니다.

    딕셔너리 기반으로 상태를 유연하게 저장하며, 대화 기록 관리,
    상태 스냅샷 및 복원 기능을 제공합니다.

    주요 상태 키 (기본적으로 생성됨):
    - `created_at` (float): 상태 객체가 생성된 시간 (타임스탬프).
    - `conversation` (List[Dict]): 사용자, 어시스턴트, 시스템 간의 대화 메시지 목록.
        - 각 메시지 딕셔너리는 'role', 'content', 'timestamp' 키를 가집니다.
    - `context` (Dict): 현재 작업이나 에이전트 실행과 관련된 임시 데이터, 중간 결과 등을 저장하는 공간.
                       예: 현재 처리 중인 알림 정보, 검색된 문서, LLM 응답 등.
                       구조는 매우 유연하며, 작업별로 필요한 데이터를 자유롭게 저장할 수 있습니다.
    - `metadata` (Dict): 에이전트 실행 전반에 걸쳐 유지될 수 있는 메타 정보.
                       예: 에이전트 ID, 세션 정보, 사용자 설정 등.

    이 외에도 `set` 메서드를 통해 필요한 상태 정보를 자유롭게 추가할 수 있습니다.
    """

    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        """
        State 객체를 초기화합니다.

        Args:
            initial_state: 초기 상태로 사용할 딕셔너리 (선택 사항).
                           제공되면 이 딕셔너리를 기반으로 상태를 초기화하고,
                           제공되지 않으면 빈 상태에서 시작합니다.
        """
        # 초기 상태가 주어지면 깊은 복사하여 사용하고, 아니면 빈 딕셔너리로 시작
        # 깊은 복사를 통해 원본 initial_state 객체와의 연결을 끊습니다.
        self.state: Dict[str, Any] = copy.deepcopy(initial_state) if initial_state else {}
        # 상태 변경 이력을 저장하는 리스트 (스냅샷 저장용)
        self.history: List[Dict[str, Any]] = []
        # 로거 인스턴스 생성
        self.logger = get_logger(f"{__name__}.State")

        # 필수 상태 키가 없으면 기본값으로 초기화
        if "created_at" not in self.state:
            self.state["created_at"] = time.time() # 현재 시간 타임스탬프

        if "conversation" not in self.state:
            self.state["conversation"] = [] # 빈 대화 목록

        if "context" not in self.state:
            self.state["context"] = {} # 빈 컨텍스트 딕셔너리

        if "metadata" not in self.state:
            self.state["metadata"] = {} # 빈 메타데이터 딕셔너리

        self.logger.debug("State 객체 초기화 완료.")

    def get(self, key: str, default: Any = None) -> Any:
        """
        상태에서 특정 키에 해당하는 값을 가져옵니다.

        점(.)을 사용하여 중첩된 딕셔너리의 값에도 접근할 수 있습니다.
        예: `get('context.user_input.text')`

        Args:
            key: 값을 가져올 키 이름. 점(.)으로 중첩된 키 표현 가능.
            default: 키가 존재하지 않을 경우 반환할 기본값. 기본값: None.

        Returns:
            키에 해당하는 값 또는 기본값.
        """
        if "." in key:
            # 키에 점(.)이 포함된 경우, 중첩된 딕셔너리로 간주하고 탐색
            parts = key.split(".")
            current = self.state
            try:
                for part in parts:
                    # 현재 값이 딕셔너리이고 해당 키가 있으면 다음 단계로 이동
                    # list index 접근 등은 지원하지 않습니다. 오직 dict key 접근만 가능.
                    if isinstance(current, dict):
                        current = current[part]
                    else:
                        # 중간 경로가 딕셔너리가 아니면 값을 찾을 수 없음
                        return default
                return current
            except KeyError:
                # 경로 중간에 키가 없는 경우
                return default
            except Exception as e:
                # 예상치 못한 오류 발생 시 경고 로깅 후 기본값 반환
                self.logger.warning(f"중첩 키 '{key}' 접근 중 오류 발생: {e}")
                return default
        else:
            # 키에 점(.)이 없는 경우, 최상위 딕셔너리에서 직접 값을 가져옴
            return self.state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        상태에 특정 키와 값을 설정(추가 또는 수정)합니다.

        점(.)을 사용하여 중첩된 딕셔너리 내부에 값을 설정할 수 있습니다.
        중간 경로의 딕셔너리가 존재하지 않으면 자동으로 생성됩니다.
        예: `set('context.analysis_result.severity', 'High')`

        Args:
            key: 값을 설정할 키 이름. 점(.)으로 중첩된 키 표현 가능.
            value: 설정할 값.
        """
        if "." in key:
            # 키에 점(.)이 포함된 경우, 중첩된 딕셔너리로 간주
            parts = key.split(".")
            current = self.state
            # 마지막 부분을 제외하고 순회하며 중간 딕셔너리 생성 또는 탐색
            for i, part in enumerate(parts[:-1]):
                # 현재 경로에 해당 키가 없거나, 있더라도 딕셔너리가 아니면 새로 생성
                if part not in current or not isinstance(current.get(part), dict):
                    current[part] = {}
                current = current[part]

            # 마지막 키에 값 설정
            current[parts[-1]] = value
            self.logger.debug(f"중첩 키 '{key}'에 값 설정 완료.")
        else:
            # 키에 점(.)이 없는 경우, 최상위 딕셔너리에 직접 값 설정
            self.state[key] = value
            self.logger.debug(f"키 '{key}'에 값 설정 완료.")

    def delete(self, key: str) -> bool:
        """
        상태에서 특정 키를 삭제합니다.

        점(.)을 사용하여 중첩된 딕셔너리 내부의 키도 삭제할 수 있습니다.

        Args:
            key: 삭제할 키 이름. 점(.)으로 중첩된 키 표현 가능.

        Returns:
            키가 성공적으로 삭제되었으면 True, 그렇지 않으면 False.
        """
        if "." in key:
            # 키에 점(.)이 포함된 경우, 중첩된 딕셔너리로 간주
            parts = key.split(".")
            current = self.state
            # 삭제할 키의 부모 딕셔너리까지 이동
            try:
                for part in parts[:-1]:
                    if isinstance(current, dict):
                        current = current[part]
                    else:
                        # 중간 경로가 딕셔너리가 아니면 삭제 불가
                        return False
                # 마지막 키 삭제 시도
                if isinstance(current, dict) and parts[-1] in current:
                    del current[parts[-1]]
                    self.logger.debug(f"중첩 키 '{key}' 삭제 완료.")
                    return True
                else:
                    # 부모가 딕셔너리가 아니거나 마지막 키가 없음
                    return False
            except KeyError:
                # 경로 중간에 키가 없는 경우
                return False
            except Exception as e:
                self.logger.warning(f"중첩 키 '{key}' 삭제 중 오류 발생: {e}")
                return False
        elif key in self.state:
            # 최상위 키 삭제
            del self.state[key]
            self.logger.debug(f"키 '{key}' 삭제 완료.")
            return True
        else:
            # 최상위 키가 존재하지 않음
            return False

    def update(self, data: Dict[str, Any]) -> None:
        """
        주어진 딕셔너리의 키-값 쌍으로 상태를 업데이트합니다.
        기존에 있는 키는 값이 덮어씌워지고, 없는 키는 새로 추가됩니다.
        `dict.update()` 메서드와 동일하게 동작합니다.

        Args:
            data: 상태를 업데이트할 키-값 쌍을 담은 딕셔너리.
        """
        self.state.update(data)
        self.logger.debug(f"{len(data)}개의 키-값 쌍으로 상태 업데이트 완료.")

    def snapshot(self) -> None:
        """
        현재 상태의 복사본을 만들어 이력(history)에 저장합니다.
        `revert` 메서드를 사용하여 이 시점의 상태로 되돌릴 수 있습니다.
        상태 변경이 많은 작업 전에 호출하면 유용합니다.
        """
        # 현재 state 딕셔너리를 깊은 복사하여 history 리스트에 추가
        # 깊은 복사를 해야 이후 state 변경이 스냅샷에 영향을 주지 않습니다.
        self.history.append(copy.deepcopy(self.state))
        self.logger.debug(f"상태 스냅샷 생성 완료 (현재 이력 크기: {len(self.history)})")

    def revert(self, steps: int = 1) -> bool:
        """
        이전 상태 스냅샷으로 되돌립니다.

        Args:
            steps: 몇 단계 이전의 스냅샷으로 되돌릴지 지정. 기본값: 1 (가장 최근 스냅샷).

        Returns:
            성공적으로 되돌렸으면 True, 그렇지 않으면 (예: 되돌릴 스냅샷이 없는 경우) False.
        """
        if steps <= 0 or steps > len(self.history):
            self.logger.warning(f"{steps} 단계 이전으로 되돌릴 수 없습니다. (현재 이력 크기: {len(self.history)})")
            return False

        # 되돌아갈 스냅샷 인덱스 계산
        target_index = len(self.history) - steps
        # 해당 스냅샷을 깊은 복사하여 현재 상태로 설정
        # 깊은 복사를 해야 스냅샷 원본과 현재 상태가 독립적입니다.
        self.state = copy.deepcopy(self.history[target_index])

        # 되돌아간 시점 이후의 스냅샷들은 history에서 제거
        self.history = self.history[:target_index] # target_index 스냅샷은 남김

        self.logger.debug(f"{steps} 단계 이전 상태로 복원 완료 (현재 이력 크기: {len(self.history)})")
        return True

    def add_message(self, role: str, content: str) -> None:
        """
        대화 기록(`state['conversation']`)에 새 메시지를 추가합니다.

        Args:
            role: 메시지 발화자 역할 ('user', 'assistant', 'system' 등).
            content: 메시지 내용.
        """
        if not isinstance(self.state.get("conversation"), list):
            self.logger.warning("'conversation' 상태가 리스트가 아니어서 메시지를 추가할 수 없습니다. 리스트로 초기화합니다.")
            self.state["conversation"] = []

        message = {
            "role": role,
            "content": content,
            "timestamp": time.time() # 메시지 추가 시점 기록
        }
        self.state["conversation"].append(message)
        self.logger.debug(f"'{role}' 역할의 새 메시지 추가 완료.")

    def get_conversation_history(self, max_messages: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        대화 기록을 가져옵니다.

        Args:
            max_messages: 반환할 최대 메시지 수 (가장 최근 메시지부터).
                          None이면 전체 기록 반환. 기본값: None.

        Returns:
            대화 메시지 딕셔너리 목록.
        """
        conversation = self.get("conversation", [])
        if not isinstance(conversation, list):
            self.logger.warning("'conversation' 상태가 리스트가 아닙니다. 빈 리스트를 반환합니다.")
            return []

        if max_messages is not None and max_messages > 0:
            # 리스트 슬라이싱을 사용하여 최근 메시지 max_messages개 반환
            return conversation[-max_messages:]
        else:
            # 전체 대화 기록 반환
            return conversation

    def save(self, file_path: str) -> bool:
        """
        현재 상태를 JSON 파일로 저장합니다.

        Args:
            file_path: 상태를 저장할 파일 경로.

        Returns:
            저장에 성공하면 True, 실패하면 False.
        """
        try:
            # 파일 경로의 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # 상태 딕셔너리를 JSON 형식으로 파일에 쓰기
            # indent=2 옵션으로 가독성 좋게 저장
            # ensure_ascii=False 옵션으로 유니코드 문자(한글 등)를 그대로 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)

            self.logger.info(f"상태를 파일에 저장했습니다: {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"상태 저장 중 오류 발생 ({file_path}): {str(e)}")
            return False

    @classmethod
    def load(cls, file_path: str) -> Optional['State']:
        """
        JSON 파일에서 상태를 로드하여 새로운 State 객체를 생성합니다.

        Args:
            file_path: 상태를 로드할 파일 경로.

        Returns:
            로드된 상태를 담은 새로운 State 객체. 로드 실패 시 None 반환.
        """
        if not os.path.exists(file_path):
            logger.error(f"상태 파일이 존재하지 않습니다: {file_path}")
            return None
        try:
            # JSON 파일을 읽어 딕셔너리로 로드
            with open(file_path, 'r', encoding='utf-8') as f:
                state_data = json.load(f)

            logger.info(f"파일에서 상태를 로드했습니다: {file_path}")
            # 로드된 딕셔너리를 사용하여 새로운 State 객체 생성 및 반환
            return cls(state_data)

        except json.JSONDecodeError as e:
            logger.error(f"상태 파일 JSON 파싱 오류 ({file_path}): {str(e)}")
            return None
        except Exception as e:
            logger.error(f"상태 로드 중 오류 발생 ({file_path}): {str(e)}")
            return None

    def reset(self) -> None:
        """
        상태를 초기값으로 리셋합니다.
        `created_at` 타임스탬프는 유지하고, `conversation`, `context`, `metadata` 등은
        기본값으로 되돌립니다. 이력(`history`)도 초기화됩니다.
        """
        # 생성 시간은 유지
        created_at = self.state.get("created_at", time.time())

        # 상태 딕셔너리를 기본 구조로 재생성
        self.state = {
            "created_at": created_at,
            "conversation": [],
            "context": {},
            "metadata": {}
        }

        # 이력 리스트도 비움
        self.history = []
        self.logger.info("상태를 초기값으로 리셋했습니다.")

    # --- 파이썬 내장 메서드 오버라이딩 (딕셔너리처럼 사용 가능하게 함) ---

    def __getitem__(self, key: str) -> Any:
        """객체[키] 형식으로 상태 값에 접근할 수 있게 합니다 (예: state['context'])."""
        # get 메서드를 호출하되, 키가 없을 경우 KeyError 발생시킴 (딕셔너리 기본 동작)
        value = self.get(key)
        if value is None and key not in self.state: # get이 default=None을 반환한 경우와 실제 키가 없는 경우 구분
             # '.' 이 포함된 키에 대한 처리 추가 필요 가능성 있음
             # 여기서는 간단히 KeyError 발생
             raise KeyError(f"상태에 '{key}' 키가 존재하지 않습니다.")
        return value


    def __setitem__(self, key: str, value: Any) -> None:
        """객체[키] = 값 형식으로 상태 값을 설정할 수 있게 합니다 (예: state['context'] = {...})."""
        self.set(key, value)

    def __delitem__(self, key: str) -> None:
        """del 객체[키] 형식으로 상태 키를 삭제할 수 있게 합니다 (예: del state['last_error'])."""
        if not self.delete(key):
            # delete 메서드가 False를 반환하면 (키가 없거나 삭제 실패 시) KeyError 발생
            raise KeyError(f"상태에서 '{key}' 키를 삭제할 수 없습니다 (키가 없거나 오류 발생).")

    def __contains__(self, key: str) -> bool:
        """'키 in 객체' 형식으로 키 존재 여부를 확인할 수 있게 합니다 (예: 'context' in state)."""
        # get 메서드를 사용하여 키 존재 여부 확인 (중첩 키 포함)
        # get 메서드는 키가 없으면 None(또는 default)을 반환하므로,
        # None이 아닌지 확인하는 것으로 존재 여부를 판단합니다.
        # 만약 값으로 None이 저장될 수 있다면, 더 엄격한 확인 필요.
        # 여기서는 get의 기본 동작을 활용합니다.
        try:
            # 중첩 키 확인을 위해 get 사용
            sentinel = object() # 고유 객체
            return self.get(key, sentinel) is not sentinel
        except Exception:
            # get 실행 중 오류 발생 시 False 반환
             return False

    def __len__(self) -> int:
        """len(객체) 형식으로 최상위 상태 키의 개수를 반환합니다."""
        return len(self.state)

    def __str__(self) -> str:
        """print(객체) 시 상태 딕셔너리를 보기 좋게 출력합니다."""
        # 상태 딕셔너리를 JSON 문자열로 변환하여 반환 (가독성 좋게)
        try:
             return json.dumps(self.state, indent=2, ensure_ascii=False)
        except Exception:
            # JSON 변환 실패 시 기본 문자열 표현 반환
             return super().__str__() 