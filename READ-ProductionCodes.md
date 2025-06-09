# 프로덕션 수준 프로젝트 이해하기



## 🏗️ **1. 프로덕션 수준 프로젝트 구성 단계**

프로덕션 수준의 AI 프로젝트는 **대형 건물을 짓는 것**
단계별로 차근차근 진행해야 할 필요가 있음

### **1단계: 기획 및 설계 (Planning & Design)**
```python
# 이 프로젝트의 목적: SKEnergy 통지 보조 시스템
# - 작업명과 상세 내용을 입력받아
# - AI가 설비코드, 현상코드, 작업시기를 추론하는 시스템
```

### **2단계: 기반 인프라 구축 (Foundation)**
- **설정 관리**: `config/` 폴더 - 건물의 설계도와 같음
- **로깅 시스템**: `utils/logger.py` - 건물의 감시 시스템과 같음
- **의존성 관리**: `requirements.txt` - 건축 자재 목록과 같음

### **3단계: 핵심 엔진 구축 (Core Engine)**
- **AI 모델 관리자**: `lib/model_manager.py` - 건물의 두뇌와 같음
- **벡터 저장소**: `lib/vector_store.py` - 건물의 기억 저장소와 같음
- **상태 관리**: `state.py` - 건물의 현재 상황을 추적하는 시스템

### **4단계: 처리 로직 구현 (Processing Logic)**
- **처리 노드들**: `nodes.py` - 건물 내 각 부서의 전문가들과 같음
- **에이전트**: `agent.py` - 건물의 총괄 관리자와 같음

### **5단계: 사용자 인터페이스 (User Interface)**
- **웹 인터페이스**: `app.py` - 건물의 접수 창구와 같음

## 🔧 **2. 각 구성요소 파일들의 기능**

### **🏠 메인 애플리케이션 계층**

#### **📱 `app.py` - 웹 인터페이스 (접수 창구)**
```python
# 사용자가 작업명과 상세내용을 입력하는 웹 페이지
# Gradio를 사용해 간단한 웹 UI 제공
# 마치 병원의 접수 창구와 같은 역할
```

**비유**: 병원의 접수 데스크
- 환자(사용자)가 증상(작업명)을 말하면
- 접수원(Gradio UI)이 정보를 받아서
- 의사(AI 에이전트)에게 전달

#### **🤖 `agent.py` - 핵심 에이전트 (총괄 관리자)**
```python
class NotiAssistanceAgent(BaseAgent):
    """
    병원의 주치의와 같은 역할
    - 환자 정보를 받아서 분석
    - 필요한 검사(노드들)를 지시
    - 최종 진단 결과를 종합
    """
```

### **🧠 핵심 엔진 계층**

#### **📊 `state.py` - 상태 관리 (의료 기록부)**
```python
# 현재 진행 중인 모든 정보를 저장
# 대화 히스토리, 중간 결과, 처리 상태 등
# 마치 환자의 의료 기록부와 같음
```

#### **🧠 `lib/model_manager.py` - AI 모델 관리자 (전문의)**
```python
# OpenAI GPT, Claude 등 다양한 AI 모델을 관리
# 각 상황에 맞는 최적의 모델을 선택하여 사용
# 마치 각 분야별 전문의를 관리하는 시스템
```

#### **🔍 `lib/vector_store.py` - 벡터 저장소 (의료 서적 라이브러리)**
```python
# 설비 정보, 과거 사례 등을 벡터로 변환하여 저장
# 유사한 사례를 빠르게 검색할 수 있도록 함
# 마치 의학 도서관에서 유사 사례를 찾는 것과 같음
```

### **⚙️ 처리 노드 계층**

#### **⚙️ `nodes.py` - 처리 노드들 (각 부서 전문가들)**
```python
# LLMNode: AI 모델과 대화하는 전문가
# VectorSearchNode: 유사 사례를 찾는 전문가  
# PromptTemplateNode: 질문을 잘 정리하는 전문가
# NotificationAnalysisNode: 통지 내용을 분석하는 전문가
```

**비유**: 병원의 각 진료과
- **내과 전문의**(LLMNode): 종합적인 진단
- **영상의학과**(VectorSearchNode): 과거 사례 검색
- **간호사**(PromptTemplateNode): 환자 정보 정리
- **검사실**(NotificationAnalysisNode): 상세 분석

### **🛠️ 유틸리티 계층**

#### **📝 `utils/logger.py` - 로깅 시스템 (병원 기록 시스템)**
```python
# 모든 처리 과정을 상세히 기록
# 오류 발생 시 원인 추적 가능
# 시스템 성능 모니터링
```

#### **🔢 `utils/token_counter.py` - 토큰 계산기 (비용 계산기)**
```python
# AI API 사용량을 정확히 계산
# 비용 최적화를 위한 모니터링
# 마치 병원의 수납 시스템과 같음
```

#### **💾 `utils/cache.py` - 캐시 시스템 (빠른 참조 카드)**
```python
# 자주 사용되는 결과를 임시 저장
# 같은 질문에 대해 빠른 답변 제공
# 마치 의사의 빠른 참조 카드와 같음
```

### **⚙️ 설정 계층**

#### **📋 `config/model_config.yaml` - 모델 설정 (병원 운영 규정)**
```yaml
# AI 모델별 상세 설정
# API 키, 모델 버전, 파라미터 등
llm:
  provider: "openai"
  model: "gpt-4"
  temperature: 0.1
```

#### **📝 `config/prompt_templates.yaml` - 프롬프트 템플릿 (질문 양식)**
```yaml
# AI에게 질문할 때 사용하는 표준 양식
# 일관된 결과를 얻기 위한 템플릿
analysis_prompt: |
  작업명: {work_name}
  상세내용: {work_details}
  
  위 정보를 바탕으로 설비코드와 현상코드를 추론.
```

## 🔗 **3. 구성요소들의 상호연결 및 작동방식**

### **🌊 데이터 흐름 (Water Flow Analogy)**

마치 **정수 처리장**의 물 흐름과 유사:

```python
# 1단계: 원수 유입 (Raw Input)
사용자 입력 → app.py → process_work_order()

# 2단계: 1차 처리 (Primary Processing)
agent.run() → NotiAssistanceAgent → State 객체에 저장

# 3단계: 전문 처리 (Specialized Processing)
각 Node들이 순차적으로 처리:
├── PromptTemplateNode: 질문 정리
├── VectorSearchNode: 유사 사례 검색
├── LLMNode: AI 분석
└── NotificationAnalysisNode: 결과 구조화

# 4단계: 최종 출력 (Final Output)
처리 결과 → app.py → Gradio UI → 사용자
```

### **🔄 실제 처리 과정 예시**

```python
# 사용자가 "펌프 Y-PG78505B 유량 저하" 입력
# ↓
# 1. app.py의 process_work_order() 함수 호출
def process_work_order(work_name: str, work_details: str):
    agent_inputs = {
        "work_name": work_name,
        "work_details": work_details
    }
    result = agent.run(task="process_work_order", inputs=agent_inputs)
    
# ↓  
# 2. agent.py의 run() 메서드에서 작업 분배
def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
    # State 객체에 현재 상황 저장
    self.agent_state.update(inputs)
    
    # 적절한 처리 메서드 호출
    if task == "process_work_order":
        return self._run_process_work_order(inputs)

# ↓
# 3. 각 노드들이 순차적으로 처리
# - 프롬프트 템플릿으로 질문 정리
# - 벡터 저장소에서 유사 사례 검색  
# - LLM으로 분석 및 추론
# - 결과를 구조화하여 반환
```

## 📋 **4. 개발 시 파일 작성 순서**

### **🏗️ 건축 순서와 같은 개발 순서**

#### **1단계: 기초 공사 (Foundation First)**
```bash
1. requirements.txt          # 필요한 라이브러리 정의
2. config/logging_config.yaml  # 로깅 설정
3. utils/logger.py           # 로깅 시스템 구축
4. utils/__init__.py         # 유틸리티 패키지 초기화
```

#### **2단계: 기본 구조 (Basic Structure)**
```bash
5. lib/base.py              # 기본 클래스 정의
6. lib/utils.py             # 공통 유틸리티 함수
7. config/model_config.yaml  # 모델 설정
8. lib/model_manager.py     # AI 모델 관리자
```

#### **3단계: 데이터 저장소 (Data Layer)**
```bash
9. lib/vector_store.py      # 벡터 저장소
10. utils/equipment_data_loader.py  # 데이터 로더
11. utils/cache.py          # 캐시 시스템
```

#### **4단계: 비즈니스 로직 (Business Logic)**
```bash
12. state.py               # 상태 관리
13. config/prompt_templates.yaml  # 프롬프트 템플릿
14. prompt_engineering/    # 프롬프트 엔지니어링
15. nodes.py              # 처리 노드들
```

#### **5단계: 핵심 에이전트 (Core Agent)**
```bash
16. agent.py              # 메인 에이전트
17. utils/token_counter.py # 토큰 계산기
```

#### **6단계: 사용자 인터페이스 (User Interface)**
```bash
18. app.py                # 웹 인터페이스
19. README.md             # 사용 설명서
```

### **🔍 각 단계별 개발 팁**

#### **기초 공사 단계에서 중요한 점:**
```python
# utils/logger.py - 모든 곳에서 사용되므로 가장 먼저 구현
def get_logger(name: str):
    """
    로거는 마치 건물의 전기 배선과 같음
    모든 방(모듈)에서 필요하므로 가장 먼저 설치
    """
    pass
```

#### **구조 설계 단계에서 중요한 점:**
```python
# lib/base.py - 모든 클래스의 부모 클래스
class BaseAgent:
    """
    마치 건물의 기본 구조와 같음
    모든 에이전트가 공통으로 가져야 할 기능 정의
    """
    pass

class BaseNode:
    """
    모든 처리 노드의 공통 인터페이스
    일관된 처리 방식 보장
    """
    pass
```

## 🏛️ **5. 시스템 아키텍처 설계 방법**

### **🎯 아키텍처 설계 5단계**

#### **1단계: 요구사항 분석 (Requirement Analysis)**
```python
"""
이 프로젝트의 요구사항:
1. 작업명과 상세내용을 입력받기
2. AI로 설비코드 추론하기
3. AI로 현상코드 추론하기  
4. 웹 인터페이스로 결과 제공하기
5. 처리 과정을 로그로 기록하기
"""
```

#### **2단계: 계층별 설계 (Layer Design)**
```python
# 계층별 분리 - 마치 건물의 층별 구조와 같음

# 5층: 사용자 인터페이스 계층 (Presentation Layer)
# - app.py (Gradio 웹 UI)

# 4층: 비즈니스 로직 계층 (Business Logic Layer)  
# - agent.py (핵심 비즈니스 로직)
# - nodes.py (세부 처리 로직)

# 3층: 서비스 계층 (Service Layer)
# - model_manager.py (AI 모델 서비스)
# - vector_store.py (검색 서비스)

# 2층: 데이터 계층 (Data Layer)
# - state.py (상태 데이터)
# - data/ 폴더 (정적 데이터)

# 1층: 인프라 계층 (Infrastructure Layer)
# - utils/ 폴더 (공통 유틸리티)
# - config/ 폴더 (설정 관리)
```

#### **3단계: 모듈간 의존성 설계 (Dependency Design)**
```python
# 의존성 방향: 상위 계층 → 하위 계층
# 마치 건물에서 위층이 아래층에 의존하는 것과 같음

app.py → agent.py → nodes.py → lib/ → utils/
  ↓         ↓         ↓        ↓       ↓
사용자    비즈니스   세부처리   핵심엔진  기반시설
```

#### **4단계: 데이터 흐름 설계 (Data Flow Design)**
```python
# 데이터가 시스템을 통과하는 경로 설계
# 마치 건물의 파이프라인과 같음

입력 데이터 → 전처리 → AI 분석 → 후처리 → 출력 데이터
    ↓           ↓        ↓        ↓         ↓
  app.py → PromptNode → LLMNode → AnalysisNode → app.py
```

#### **5단계: 에러 처리 및 모니터링 설계**
```python
# 각 계층별 에러 처리 방식 정의
# 마치 건물의 안전 시스템과 같음

try:
    result = agent.run(inputs)
except ModelError:
    # AI 모델 오류 처리
    logger.error("AI 모델 오류 발생")
except VectorStoreError:  
    # 벡터 저장소 오류 처리
    logger.error("검색 시스템 오류 발생")
except Exception as e:
    # 예상치 못한 오류 처리
    logger.critical(f"심각한 오류: {e}")
```

### **🏗️ 아키텍처 설계 실전 팁**

#### **1. 단일 책임 원칙 (Single Responsibility Principle)**
```python
# 각 모듈은 하나의 책임만 가져야 함
# 마치 각 방이 하나의 용도만 갖는 것과 같음

class ModelManager:
    """오직 AI 모델 관리만 담당"""
    pass

class VectorStore:
    """오직 벡터 검색만 담당"""  
    pass

class Logger:
    """오직 로깅만 담당"""
    pass
```

#### **2. 의존성 역전 원칙 (Dependency Inversion Principle)**
```python
# 구체적인 구현이 아닌 추상화에 의존
# 마치 건물이 특정 브랜드가 아닌 표준 규격에 맞춰 설계되는 것과 같음

class BaseAgent:
    def __init__(self, model_manager: BaseModelManager):
        # 특정 구현체가 아닌 인터페이스에 의존
        self.model_manager = model_manager
```

#### **3. 설정 기반 설계 (Configuration-Driven Design)**
```python
# 하드코딩 대신 설정 파일 활용
# 마치 건물의 설계도를 변경하여 다양한 건물을 지을 수 있는 것과 같음

# config/model_config.yaml
llm:
  provider: "openai"  # 언제든 "anthropic"으로 변경 가능
  model: "gpt-4"      # 언제든 "gpt-3.5-turbo"로 변경 가능
```

## 🎯 **핵심 개발 원칙 요약**

### **🏗️ 건축 원칙과 같은 개발 원칙**

1. **견고한 기초**: 로깅, 설정, 유틸리티부터 구축
2. **계층별 구조**: 각 계층의 역할을 명확히 분리
3. **모듈화**: 각 모듈은 독립적으로 테스트 가능해야 함
4. **확장성**: 새로운 기능 추가가 쉬워야 함
5. **유지보수성**: 코드 읽기와 수정이 쉬워야 함

### **🔄 실제 개발 프로세스**

```python
# 1. 최소 기능으로 시작 (MVP - Minimum Viable Product)
# 2. 점진적 기능 추가 (Incremental Development)  
# 3. 지속적 리팩토링 (Continuous Refactoring)
# 4. 철저한 테스트 (Comprehensive Testing)
# 5. 문서화 (Documentation)
```

체계적으로 접근하면 복잡한시스템을 **블록을 쌓듯이** 차근차근 구축 가능


```python
# 이 프로젝트의 목적: SKEnergy 통지 보조 시스템
# - 작업명과 상세 내용을 입력받아
# - AI가 설비코드, 현상코드, 작업시기를 추론하는 시스템
```

```python
# 사용자가 작업명과 상세내용을 입력하는 웹 페이지
# Gradio를 사용해 간단한 웹 UI 제공
# 마치 은행의 접수 창구와 같은 역할
```

```python
class NotiAssistanceAgent(BaseAgent):
    """
    병원의 주치의와 같은 역할
    - 환자 정보를 받아서 분석
    - 필요한 검사(노드들)를 지시
    - 최종 진단 결과를 종합
    """
```

```python
# 현재 진행 중인 모든 정보를 저장
# 대화 히스토리, 중간 결과, 처리 상태 등
# 마치 환자의 의료 기록부와 같음
```

```python
# OpenAI GPT, Claude 등 다양한 AI 모델을 관리
# 각 상황에 맞는 최적의 모델을 선택하여 사용
# 마치 각 분야별 전문의를 관리하는 시스템
```

```python
# 설비 정보, 과거 사례 등을 벡터로 변환하여 저장
# 유사한 사례를 빠르게 검색할 수 있도록 함
# 마치 의학 도서관에서 유사 사례를 찾는 것과 같음
```

```python
# LLMNode: AI 모델과 대화하는 전문가
# VectorSearchNode: 유사 사례를 찾는 전문가  
# PromptTemplateNode: 질문을 잘 정리하는 전문가
# NotificationAnalysisNode: 통지 내용을 분석하는 전문가
```

```python
# 모든 처리 과정을 상세히 기록
# 오류 발생 시 원인 추적 가능
# 시스템 성능 모니터링
```

```python
# AI API 사용량을 정확히 계산
# 비용 최적화를 위한 모니터링
# 마치 병원의 수납 시스템과 같음
```

```python
# 자주 사용되는 결과를 임시 저장
# 같은 질문에 대해 빠른 답변 제공
# 마치 의사의 빠른 참조 카드와 같음
```

```yaml
# AI 모델별 상세 설정
# API 키, 모델 버전, 파라미터 등
llm:
  provider: "openai"
  model: "gpt-4"
  temperature: 0.1
```

```yaml
# AI에게 질문할 때 사용하는 표준 양식
# 일관된 결과를 얻기 위한 템플릿
analysis_prompt: |
  작업명: {work_name}
  상세내용: {work_details}
  
  위 정보를 바탕으로 설비코드와 현상코드를 추론하세요.
```

```python
# 1단계: 원수 유입 (Raw Input)
사용자 입력 → app.py → process_work_order()

# 2단계: 1차 처리 (Primary Processing)
agent.run() → NotiAssistanceAgent → State 객체에 저장

# 3단계: 전문 처리 (Specialized Processing)
각 Node들이 순차적으로 처리:
├── PromptTemplateNode: 질문 정리
├── VectorSearchNode: 유사 사례 검색
├── LLMNode: AI 분석
└── NotificationAnalysisNode: 결과 구조화

# 4단계: 최종 출력 (Final Output)
처리 결과 → app.py → Gradio UI → 사용자
```

```python
# 사용자가 "펌프 Y-PG78505B 유량 저하" 입력
# ↓
# 1. app.py의 process_work_order() 함수 호출
def process_work_order(work_name: str, work_details: str):
    agent_inputs = {
        "work_name": work_name,
        "work_details": work_details
    }
    result = agent.run(task="process_work_order", inputs=agent_inputs)
    
# ↓  
# 2. agent.py의 run() 메서드에서 작업 분배
def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
    # State 객체에 현재 상황 저장
    self.agent_state.update(inputs)
    
    # 적절한 처리 메서드 호출
    if task == "process_work_order":
        return self._run_process_work_order(inputs)

# ↓
# 3. 각 노드들이 순차적으로 처리
# - 프롬프트 템플릿으로 질문 정리
# - 벡터 저장소에서 유사 사례 검색  
# - LLM으로 분석 및 추론
# - 결과를 구조화하여 반환
```

```shellscript
1. requirements.txt          # 필요한 라이브러리 정의
2. config/logging_config.yaml  # 로깅 설정
3. utils/logger.py           # 로깅 시스템 구축
4. utils/__init__.py         # 유틸리티 패키지 초기화
```

```shellscript
5. lib/base.py              # 기본 클래스 정의
6. lib/utils.py             # 공통 유틸리티 함수
7. config/model_config.yaml  # 모델 설정
8. lib/model_manager.py     # AI 모델 관리자
```

```shellscript
9. lib/vector_store.py      # 벡터 저장소
10. utils/equipment_data_loader.py  # 데이터 로더
11. utils/cache.py          # 캐시 시스템
```

```shellscript
12. state.py               # 상태 관리
13. config/prompt_templates.yaml  # 프롬프트 템플릿
14. prompt_engineering/    # 프롬프트 엔지니어링
15. nodes.py              # 처리 노드들
```

```shellscript
16. agent.py              # 메인 에이전트
17. utils/token_counter.py # 토큰 계산기
```

```shellscript
18. app.py                # 웹 인터페이스
19. README.md             # 사용 설명서
```

```python
# utils/logger.py - 모든 곳에서 사용되므로 가장 먼저 구현
def get_logger(name: str):
    """
    로거는 마치 건물의 전기 배선과 같음
    모든 방(모듈)에서 필요하므로 가장 먼저 설치
    """
    pass
```

```python
# lib/base.py - 모든 클래스의 부모 클래스
class BaseAgent:
    """
    마치 건물의 기본 구조와 같음
    모든 에이전트가 공통으로 가져야 할 기능 정의
    """
    pass

class BaseNode:
    """
    모든 처리 노드의 공통 인터페이스
    일관된 처리 방식 보장
    """
    pass
```

```python
"""
이 프로젝트의 요구사항:
1. 작업명과 상세내용을 입력받기
2. AI로 설비코드 추론하기
3. AI로 현상코드 추론하기  
4. 웹 인터페이스로 결과 제공하기
5. 처리 과정을 로그로 기록하기
"""
```

```python
# 계층별 분리 - 마치 건물의 층별 구조와 같음

# 5층: 사용자 인터페이스 계층 (Presentation Layer)
# - app.py (Gradio 웹 UI)

# 4층: 비즈니스 로직 계층 (Business Logic Layer)  
# - agent.py (핵심 비즈니스 로직)
# - nodes.py (세부 처리 로직)

# 3층: 서비스 계층 (Service Layer)
# - model_manager.py (AI 모델 서비스)
# - vector_store.py (검색 서비스)

# 2층: 데이터 계층 (Data Layer)
# - state.py (상태 데이터)
# - data/ 폴더 (정적 데이터)

# 1층: 인프라 계층 (Infrastructure Layer)
# - utils/ 폴더 (공통 유틸리티)
# - config/ 폴더 (설정 관리)
```

```python
# 의존성 방향: 상위 계층 → 하위 계층
# 마치 건물에서 위층이 아래층에 의존하는 것과 같음

app.py → agent.py → nodes.py → lib/ → utils/
  ↓         ↓         ↓        ↓       ↓
사용자    비즈니스   세부처리   핵심엔진  기반시설
```

```python
# 데이터가 시스템을 통과하는 경로 설계
# 마치 건물의 파이프라인과 같음

입력 데이터 → 전처리 → AI 분석 → 후처리 → 출력 데이터
    ↓           ↓        ↓        ↓         ↓
  app.py → PromptNode → LLMNode → AnalysisNode → app.py
```

```python
# 각 계층별 에러 처리 방식 정의
# 마치 건물의 안전 시스템과 같음

try:
    result = agent.run(inputs)
except ModelError:
    # AI 모델 오류 처리
    logger.error("AI 모델 오류 발생")
except VectorStoreError:  
    # 벡터 저장소 오류 처리
    logger.error("검색 시스템 오류 발생")
except Exception as e:
    # 예상치 못한 오류 처리
    logger.critical(f"심각한 오류: {e}")
```

```python
# 각 모듈은 하나의 책임만 가져야 함
# 마치 각 방이 하나의 용도만 갖는 것과 같음

class ModelManager:
    """오직 AI 모델 관리만 담당"""
    pass

class VectorStore:
    """오직 벡터 검색만 담당"""  
    pass

class Logger:
    """오직 로깅만 담당"""
    pass
```

```python
# 구체적인 구현이 아닌 추상화에 의존
# 마치 건물이 특정 브랜드가 아닌 표준 규격에 맞춰 설계되는 것과 같음

class BaseAgent:
    def __init__(self, model_manager: BaseModelManager):
        # 특정 구현체가 아닌 인터페이스에 의존
        self.model_manager = model_manager
```

```python
# 하드코딩 대신 설정 파일 활용
# 마치 건물의 설계도를 변경하여 다양한 건물을 지을 수 있는 것과 같음

# config/model_config.yaml
llm:
  provider: "openai"  # 언제든 "anthropic"으로 변경 가능
  model: "gpt-4"      # 언제든 "gpt-3.5-turbo"로 변경 가능
```

```python
# 1. 최소 기능으로 시작 (MVP - Minimum Viable Product)
# 2. 점진적 기능 추가 (Incremental Development)  
# 3. 지속적 리팩토링 (Continuous Refactoring)
# 4. 철저한 테스트 (Comprehensive Testing)
# 5. 문서화 (Documentation)
```

