"""
SKAI-NotiAssistance 작업 발의 자동화 웹 인터페이스 (Gradio)

이 스크립트는 Gradio 라이브러리를 사용하여 SKAI-NotiAssistance 에이전트와 상호작용하는
간단한 웹 UI를 제공합니다. 사용자가 작업명과 작업 상세 내용을 입력하면,
에이전트가 설비 코드, 현상 코드 등을 추론하여 결과를 보여줍니다.

실행 방법:
1. 필요한 라이브러리를 설치합니다: pip install gradio pandas
2. 프로젝트 루트 디렉토리에서 다음 명령어를 실행합니다:
   python app.py
3. 웹 브라우저에서 http://127.0.0.1:7860 주소로 접속합니다.
"""

import os
import sys
import json
from pathlib import Path

# Gradio 라이브러리 임포트
import gradio as gr

# 프로젝트 루트 디렉토리 설정 및 sys.path 추가 (모듈 임포트 위함)
# __file__은 현재 실행 중인 스크립트 파일의 경로입니다.
SCRIPT_DIR = Path(__file__).resolve().parent
# 프로젝트 루트 디렉토리 설정 (현재 파일의 부모 디렉토리)
PROJECT_ROOT = SCRIPT_DIR

# sys.path에 프로젝트 루트가 없으면 추가합니다.
# 이렇게 하면 프로젝트 내의 다른 모듈(agent, lib 등)을 임포트할 수 있습니다.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    print(f"Project root added to sys.path: {PROJECT_ROOT}")

# 프로젝트 에이전트 임포트
try:
    from agent import NotiAssistanceAgent # 수정: 기존 에이전트 사용
    from utils.logger import get_logger # 로깅 사용
except ImportError as e:
    print("--------------------------------------------------")
    print(f"[Error] Failed to import necessary modules: {e}")
    print(f"PYTHONPATH: {sys.path}")
    print("Please ensure you are running this script from the PROJECT_SKAI-NOTIASSISTANCE directory,")
    print("and the 'agent.py' and 'utils' directory exist.")
    print("--------------------------------------------------")
    sys.exit(1)

# 로거 설정
logger = get_logger(__name__)

# --- 에이전트 초기화 --- 
# 애플리케이션 시작 시 에이전트를 한 번만 로드합니다.
# 설정 파일(config/app_config.yaml 등)을 기반으로 에이전트 초기화
# 에이전트 초기화에 필요한 설정 파일 경로 등을 지정해야 할 수 있습니다.
# 예를 들어, 기본 설정 파일 경로를 사용하도록 설정합니다.
# config_path = PROJECT_ROOT / "config" / "app_config.yaml"
# agent = NotiAssistanceAgent(config_path=str(config_path))

# 간단한 초기화 (기본 설정 사용 가정)
# 실제 사용 시에는 설정 파일을 명시적으로 로드하는 것이 좋습니다.
logger.info("Initializing NotiAssistanceAgent...")
try:
    agent = NotiAssistanceAgent()
    logger.info("NotiAssistanceAgent initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize NotiAssistanceAgent: {e}", exc_info=True)
    print("[Critical Error] Agent initialization failed. Please check configurations and logs. Exiting.")
    sys.exit(1)
# ----------------------

def process_work_order(work_name: str, work_details: str) -> tuple[str, str, str, str]:
    """
    Gradio 인터페이스에서 호출될 함수.
    작업명과 상세 내용을 받아 에이전트를 실행하고 결과를 반환합니다.

    Args:
        work_name (str): 사용자가 입력한 작업명.
        work_details (str): 사용자가 입력한 작업 상세 내용.

    Returns:
        tuple[str, str, str, str]: (추천 설비코드, 추천 현상코드, 추천 작업시기, 처리 과정 로그)
    """
    logger.info(f"Received request: work_name='{work_name}', work_details='{work_details[:50]}...'" ) # 상세 내용은 일부만 로깅

    # 입력 값 유효성 검사 (작업명은 필수)
    if not work_name or not work_name.strip():
        logger.warning("Work name is empty.")
        # 작업명이 비어있으면 오류 메시지 대신 초기 상태 반환
        return "작업명을 입력해주세요.", "", "", "작업명이 필요합니다."

    # 에이전트 실행을 위한 입력 데이터 구성
    # agent.run 메서드는 'task'와 'inputs' 딕셔너리를 받습니다.
    # 'process_work_order'라는 새로운 task를 정의하고 agent.py에 해당 로직 추가 필요
    agent_inputs = {
        "work_name": work_name,
        "work_details": work_details or "" # 상세 내용이 없으면 빈 문자열 전달
        # 필요한 경우, 후보 코드 목록 등 추가 정보 전달 가능
        # "item_no_candidates": [...] 
    }

    try:
        logger.debug(f"Running agent with task 'process_work_order' and inputs: {agent_inputs}")
        # 에이전트 실행: 'process_work_order' 태스크 호출
        # 이 태스크는 agent.py 내부에 해당 처리를 위한 _run_process_work_order 메서드가 구현되어야 합니다.
        result = agent.run(task="process_work_order", inputs=agent_inputs)
        logger.info(f"Agent run completed. Result status: {result.get('status', 'N/A')}")
        logger.debug(f"Agent raw result: {result}") # 상세 결과 로깅 (디버깅용)

        # 에이전트 실행 결과에서 필요한 정보 추출
        # 결과 딕셔너리에 해당 키가 없을 경우를 대비하여 .get() 사용 및 기본값 설정
        item_no = result.get("selected_item_no", "추론 실패")
        failure_code = result.get("selected_failure_code", "추론 실패")
        # '작업 시기'는 현재 에이전트 기능에 없으므로 임시로 "미지원" 표시
        # 추후 에이전트 기능 확장 시 이 부분 수정 필요
        work_schedule = result.get("selected_work_schedule", "미지원 기능")

        # 처리 과정 로그 생성 (에이전트 결과나 상태를 기반으로 생성)
        # 예시: 간단하게 결과 딕셔너리를 JSON 문자열로 표시
        # 더 상세한 로그는 에이전트 상태(agent_state)나 별도 로깅 메커니즘 활용 필요
        process_log = json.dumps(result, indent=2, ensure_ascii=False)

    except Exception as e:
        # 에이전트 실행 중 오류 발생 시 처리
        logger.error(f"Error processing work order: {e}", exc_info=True)
        item_no = "오류 발생"
        failure_code = "오류 발생"
        work_schedule = "오류 발생"
        process_log = f"처리 중 오류가 발생했습니다: {str(e)}"

    logger.debug(f"Returning results: item_no='{item_no}', failure_code='{failure_code}', schedule='{work_schedule}'")
    return item_no, failure_code, work_schedule, process_log

# --- Gradio 인터페이스 구성 --- 
logger.info("Setting up Gradio interface...")

# gr.Blocks()를 사용하여 UI 레이아웃 정의
with gr.Blocks(title="작업 발의 자동화 시스템 - SKAI", theme=gr.themes.Soft()) as app:
    # 앱 제목 및 설명 (Markdown 사용)
    gr.Markdown("## ⚙️ SKAI 작업 발의 자동화 시스템")
    gr.Markdown("작업명과 작업상세내용을 입력하면 AI 에이전트가 **설비코드**, **현상코드**, **작업시기** 등을 자동으로 분석하고 추천합니다.")

    # 입력과 출력을 가로로 배치 (gr.Row)
    with gr.Row():
        # 왼쪽 컬럼: 입력 필드 및 버튼
        with gr.Column(scale=1): # scale로 컬럼 너비 비율 조절 가능
            gr.Markdown("### 📝 입력 정보")
            # 작업명 입력 (필수)
            work_name_input = gr.Textbox(
                label="작업명",
                placeholder="예: 펌프 Y-PG78505B 유량 저하 발생으로 점검 요망",
                info="분석할 작업의 제목이나 핵심 내용을 입력하세요."
            )
            # 작업 상세 내용 입력 (선택)
            work_details_input = gr.Textbox(
                label="작업 상세 내용 (선택 사항)",
                placeholder="예: 최근 3일간 펌프 유량이 지속적으로 감소하였으며, 이상 소음도 간헐적으로 발생함. 긴급 점검 및 필요시 수리/교체 바람.",
                lines=4, # 여러 줄 입력 가능
                info="작업에 대한 구체적인 설명, 배경, 현상 등을 입력하면 분석 정확도가 향상됩니다."
            )
            # 분석 실행 버튼
            submit_btn = gr.Button("🚀 AI 분석 실행하기", variant="primary") # variant로 버튼 스타일 변경

        # 오른쪽 컬럼: 출력 필드
        with gr.Column(scale=1):
            gr.Markdown("### 📊 분석 결과")
            # 추천 설비 코드 출력 (수정 불가)
            item_no_output = gr.Textbox(label="✅ 추천 작업 대상 (설비 코드)", interactive=False)
            # 추천 현상 코드 출력 (수정 불가)
            failure_code_output = gr.Textbox(label="⚠️ 추천 현상 코드", interactive=False)
            # 추천 작업 시기 출력 (수정 불가)
            work_schedule_output = gr.Textbox(label="🗓️ 추천 작업 시기", interactive=False)
            # 처리 과정 로그 출력 (수정 불가)
            process_log_output = gr.Textbox(
                label="📄 처리 과정 상세 로그",
                lines=10, # 여러 줄 표시
                interactive=False,
                info="AI 에이전트의 내부 처리 단계 및 최종 결과를 보여줍니다."
            )

    # --- 이벤트 핸들러 연결 --- 
    # '분석하기' 버튼 클릭 시 process_work_order 함수 실행
    submit_btn.click(
        fn=process_work_order, # 실행할 함수
        inputs=[work_name_input, work_details_input], # 함수의 입력으로 전달될 UI 컴포넌트 리스트
        outputs=[item_no_output, failure_code_output, work_schedule_output, process_log_output] # 함수의 반환값을 표시할 UI 컴포넌트 리스트
    )

    # --- 추가 정보 섹션 --- 
    with gr.Accordion("💡 사용 가이드 및 팁", open=False): # Accordion으로 접고 펼 수 있는 섹션 생성
        gr.Markdown("""
        ### 사용 방법:
        1.  **작업명** 입력란에 분석하고자 하는 작업의 제목이나 핵심 내용을 입력합니다. (예: `Y-MOTOR-123 진동 증가`)
        2.  **(선택)** **작업 상세 내용** 입력란에 작업에 대한 부가적인 설명을 입력합니다. 상세할수록 분석 정확도가 높아집니다.
        3.  `🚀 AI 분석 실행하기` 버튼을 클릭합니다.
        4.  잠시 후 오른쪽에 AI가 분석한 **추천 설비코드**, **추천 현상코드**, **추천 작업시기** 및 **처리 과정 로그**가 표시됩니다.

        ### 팁:
        *   작업명이나 내용에 **정확한 설비 코드(ITEMNO)**를 포함하면, 해당 설비를 특정하는 데 도움이 됩니다.
        *   작업 현상(예: 누유, 소음, 진동, 온도 상승)을 구체적으로 묘사하면 현상 코드 추론에 유리합니다.
        *   처리 과정 로그를 통해 AI가 어떤 단계를 거쳐 결과를 도출했는지 확인할 수 있습니다.
        """)

logger.info("Gradio interface setup complete.")
# --- 애플리케이션 실행 --- 
if __name__ == "__main__":
    logger.info("Starting Gradio web server...")
    print("====================================================================")
    print("  SKAI 작업 발의 자동화 시스템 웹 인터페이스를 시작합니다. ")
    print("  웹 브라우저를 열고 다음 주소로 접속하세요:")
    print("  >> http://127.0.0.1:7860 <<")
    print("====================================================================")
    # Gradio 앱 실행
    # share=True 로 설정하면 외부에서 접속 가능한 임시 URL 생성 (보안 주의)
    app.launch(server_name="127.0.0.1", server_port=7860)
    logger.info("Gradio web server stopped.") 