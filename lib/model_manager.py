"""
Model manager for SKAI-NotiAssistance.

This module provides functionality for managing language model interactions.
"""

import os
import json
import yaml
import requests
from typing import Any, Dict, List, Optional, Union, Callable, Generator

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..utils.logger import get_logger
from ..utils.token_counter import count_tokens

logger = get_logger(__name__)

class ModelManager:
    """
    다양한 제공자(로컬, Ollama, OpenAI 등)의 LLM 상호작용을 관리하는 클래스입니다.
    로컬 모델 지원이 강화되었습니다.

    이 클래스는 다양한 언어 모델 제공자(OpenAI, Anthropic, Hugging Face, 로컬 모델, Ollama API)와의
    통합된 인터페이스를 제공하여 모델 관리를 용이하게 합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        모델 관리자를 초기화합니다.

        설정 딕셔너리를 사용하여 모델 설정을 로드하고,
        선택된 제공자에 따라 클라이언트를 초기화합니다.

        Args:
            config: 모델 설정을 담고 있는 딕셔너리. 이 딕셔너리는 다음 키들을 포함할 수 있습니다:
                - provider: 사용할 모델 제공자 ('local', 'ollama', 'openai', 'anthropic', 'huggingface'). 기본값: 'local'.
                - model_name: 사용할 모델 이름 (예: 'llama3:8b', 'gpt-4', 로컬 모델 경로).
                - model_path: 로컬 모델 사용 시 모델 파일 경로. 'provider'가 'local'일 때 사용됩니다.
                - temperature: 생성 텍스트의 창의성 조절 (0.0 ~ 1.0). 기본값: 0.1.
                - max_tokens: 생성할 최대 토큰 수. 기본값: 2000.
                - top_p: 확률적 샘플링 기법 파라미터. 기본값: 1.0.
                - frequency_penalty: 반복 패널티. 기본값: 0.
                - presence_penalty: 새로운 주제 장려 패널티. 기본값: 0.
                - timeout: API 요청 타임아웃 (초). 기본값: 60.
                - api_key: OpenAI, Anthropic 등 API 키. 환경 변수에서 읽어올 수도 있습니다.
                - api_url: Ollama API 엔드포인트 URL. 기본값: 'http://localhost:11434/api/generate'.
        """
        self.config = config
        # 'provider' 설정이 없으면 기본값으로 'local'을 사용합니다.
        self.provider = config.get("provider", "local")
        # 사용할 모델 이름 또는 경로를 설정합니다.
        self.model_name = config.get("model_name", "llama3:8b")
        # 생성 관련 파라미터들을 설정합니다.
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 2000)
        self.top_p = config.get("top_p", 1.0)
        self.frequency_penalty = config.get("frequency_penalty", 0)
        self.presence_penalty = config.get("presence_penalty", 0)
        self.timeout = config.get("timeout", 60)
        # API 키와 클라이언트 객체를 초기화합니다.
        self.api_key = None
        self.client = None
        # 로거를 설정합니다. 로그 이름에 provider를 포함하여 구분합니다.
        self.logger = get_logger(f"{__name__}.{self.provider}")
        
        # Ollama API 엔드포인트 URL을 설정합니다.
        self.api_url = config.get("api_url", "http://localhost:11434/api/generate")
        
        # 설정된 provider에 따라 적절한 클라이언트를 초기화합니다.
        self._initialize_client()
    
    def _initialize_client(self):
        """
        설정된 제공자('provider')에 따라 적절한 LLM 클라이언트를 초기화합니다.

        - 'local': 로컬에 저장된 모델을 Transformers 라이브러리를 사용하여 로드합니다.
                   `config`에서 'model_path'를 읽어 모델 경로를 결정합니다.
                   GPU가 사용 가능하면 자동으로 할당하며, 메모리 사용량을 최적화합니다.
        - 'ollama': Ollama API 사용을 준비합니다. 실제 API 호출은 `generate` 메소드에서 수행됩니다.
                    `config`에서 'api_url'을 읽어 API 엔드포인트를 결정합니다.
        - 'openai': OpenAI API 클라이언트를 초기화합니다. `config`나 환경 변수에서 API 키를 가져옵니다.
        - 'anthropic': Anthropic API 클라이언트를 초기화합니다. (주석 처리됨, 필요시 구현)
        - 'huggingface': Hugging Face 모델을 로컬에서 사용하거나 Inference API를 사용합니다. (주석 처리됨, 필요시 구현)
        """
        self.logger.info(f"{self.provider} 클라이언트를 초기화합니다.")
        
        if self.provider == "local":
            # 로컬 모델을 사용하는 경우
            try:
                # 설정에서 로컬 모델 경로를 가져옵니다. 없으면 model_name을 경로로 사용합니다.
                model_path = self.config.get("model_path", self.model_name)
                self.logger.info(f"로컬 모델 로드 중: {model_path}")
                
                # Transformers 라이브러리를 사용하여 토크나이저와 모델을 로드합니다.
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",  # 사용 가능한 경우 GPU 자동 할당
                    torch_dtype="auto",  # 모델에 맞는 데이터 타입 자동 설정
                    low_cpu_mem_usage=True  # CPU 메모리 사용량 줄이기 (모델 로딩 시 유용)
                )
                self.logger.info(f"로컬 모델 로드 완료: {model_path}")
                
            except ImportError:
                self.logger.error("필수 패키지(transformers, torch)가 설치되지 않았습니다. 'pip install transformers torch'로 설치해주세요.")
                raise
            except Exception as e:
                self.logger.error(f"로컬 모델 초기화 중 오류 발생: {str(e)}")
                raise
                
        elif self.provider == "ollama":
            # Ollama API를 사용하는 경우
            self.logger.info(f"Ollama API 클라이언트 설정 완료 (API URL: {self.api_url})")
            # 실제 API 요청은 generate 메소드에서 수행하므로 여기서는 별도 클라이언트 객체 생성 없음
            
        elif self.provider == "openai":
            # OpenAI API를 사용하는 경우
            try:
                from openai import OpenAI
                
                # 설정 또는 환경 변수에서 OpenAI API 키를 가져옵니다.
                self.api_key = self.config.get("api_key", os.environ.get("OPENAI_API_KEY"))
                if not self.api_key:
                    raise ValueError("OpenAI API 키가 제공되지 않았습니다. 설정 파일이나 환경 변수를 확인하세요.")
                
                # OpenAI 클라이언트를 생성합니다.
                self.client = OpenAI(api_key=self.api_key)
                self.logger.info(f"OpenAI 클라이언트 초기화 완료 (모델: {self.model_name})")
                
            except ImportError:
                self.logger.error("openai 패키지가 설치되지 않았습니다. 'pip install openai'로 설치해주세요.")
                raise
            except Exception as e:
                self.logger.error(f"OpenAI 클라이언트 초기화 중 오류 발생: {str(e)}")
                raise

        # --- 이하 다른 제공자(anthropic, huggingface) 초기화 코드는 생략 ---
        # elif self.provider == "anthropic":
        #     # ... (Anthropic 초기화 코드)
        # elif self.provider == "huggingface":
        #     # ... (Hugging Face 초기화 코드)

        else:
            # 지원하지 않는 제공자인 경우 에러 발생
            raise ValueError(f"지원하지 않는 LLM 제공자입니다: {self.provider}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
        **kwargs: Any  # 추가적인 Ollama 파라미터 등을 받기 위함
    ) -> Union[str, Generator[str, None, None]]:
        """
        언어 모델을 사용하여 텍스트를 생성합니다.

        provider 설정에 따라 로컬 모델, Ollama API, OpenAI API 등을 호출합니다.

        Args:
            prompt: 모델에게 전달할 사용자 프롬프트.
            system_prompt: 모델의 역할이나 맥락을 설정하는 시스템 프롬프트 (선택 사항).
            temperature: 생성 온도를 재정의합니다 (선택 사항).
            max_tokens: 최대 생성 토큰 수를 재정의합니다 (선택 사항).
            stream: 응답을 스트리밍할지 여부 (선택 사항). Ollama에서는 지원하지 않을 수 있습니다.
            **kwargs: 각 provider별 추가 파라미터를 전달할 수 있습니다. (예: Ollama의 'options')

        Returns:
            - stream=False 인 경우: 생성된 전체 텍스트 문자열.
            - stream=True 인 경우: 텍스트 조각(chunk)을 생성하는 제너레이터.
        """
        # 함수 인자로 받은 값이 있으면 사용하고, 없으면 클래스 기본값 사용
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        should_stream = stream if stream is not None else self.config.get("stream", False)
        
        # 입력 토큰 수 계산 (로깅용)
        prompt_tokens = count_tokens(prompt, model=self.model_name)
        system_tokens = count_tokens(system_prompt, model=self.model_name) if system_prompt else 0
        total_input_tokens = prompt_tokens + system_tokens
        
        self.logger.info(f"{self.provider}를 사용하여 텍스트 생성 시작 (입력 토큰: {total_input_tokens})")
        
        try:
            if self.provider == "local":
                # 로컬 모델을 사용하는 경우
                if not hasattr(self, 'hf_model') or not hasattr(self, 'tokenizer'):
                    raise RuntimeError("로컬 모델 또는 토크나이저가 초기화되지 않았습니다.")
                
                # 시스템 프롬프트와 사용자 프롬프트를 결합
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                
                # 텍스트를 토큰 ID로 변환하고 적절한 장치(CPU 또는 GPU)로 이동
                inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.hf_model.device)
                
                # 스트리밍 미지원 (Transformers generate는 기본적으로 스트리밍이 복잡)
                if should_stream:
                    self.logger.warning("로컬 모델에서는 스트리밍이 현재 지원되지 않습니다.")
                
                # 모델을 사용하여 텍스트 생성 (추론 모드 사용)
                with torch.no_grad():
                    outputs = self.hf_model.generate(
                        **inputs, # input_ids와 attention_mask 포함
                        max_new_tokens=tokens,
                        temperature=temp,
                        top_p=self.top_p,
                        do_sample=(temp > 0), # temperature가 0보다 크면 샘플링 사용
                        pad_token_id=self.tokenizer.eos_token_id # 패딩 토큰 설정
                    )
                
                # 생성된 토큰 ID를 다시 텍스트로 변환 (입력 프롬프트 제외)
                input_length = inputs["input_ids"].shape[1]
                result = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
                
                output_tokens = count_tokens(result, model=self.model_name)
                self.logger.debug(f"로컬 모델 응답 생성 완료 (출력 토큰: {output_tokens})")
                return result

            elif self.provider == "ollama":
                # Ollama API를 사용하는 경우
                # Ollama API는 기본적으로 스트리밍을 지원하지만, 이 구현에서는 False로 고정
                # 필요하다면 stream=True 처리 로직 추가 가능
                if should_stream:
                     self.logger.warning("Ollama API 스트리밍이 현재 구현되지 않았습니다.")

                # API 요청 페이로드 설정
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": False, # Ollama API 스트리밍 사용 여부
                    "options": { # Ollama 모델의 세부 옵션 설정
                        "temperature": temp,
                        "num_predict": tokens, # Ollama에서는 max_tokens 대신 num_predict 사용
                        "top_p": self.top_p,
                        "frequency_penalty": self.frequency_penalty,
                        "presence_penalty": self.presence_penalty,
                        **(kwargs.get("options", {})) # 추가적인 options 파라미터 병합
                    }
                }
                # system_prompt가 없으면 payload에서 제거
                if not system_prompt:
                    del payload["system"]

                self.logger.debug(f"Ollama API 요청: {self.api_url} 페이로드: {payload}")
                # requests 라이브러리를 사용하여 Ollama API 호출
                response = requests.post(self.api_url, json=payload, timeout=self.timeout)
                response.raise_for_status() # HTTP 오류 발생 시 예외 발생

                # API 응답에서 생성된 텍스트 추출
                response_data = response.json()
                result = response_data.get("response", "")

                # 응답 데이터에서 토큰 수 정보 추출 (Ollama 응답 형식에 따라 조정 필요)
                output_tokens = response_data.get("eval_count", 0) # 예시: eval_count 필드 사용
                self.logger.debug(f"Ollama API 응답 수신 완료 (출력 토큰: {output_tokens})")
                return result

            elif self.provider == "openai":
                # OpenAI API를 사용하는 경우 (기존 로직 유지)
                if not self.client:
                     raise RuntimeError("OpenAI 클라이언트가 초기화되지 않았습니다.")

                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    stream=should_stream,
                    timeout=self.timeout
                )

                if should_stream:
                    # 스트리밍 처리 제너레이터
                    def stream_generator():
                        collected_chunks = []
                        for chunk in response:
                            content = chunk.choices[0].delta.content
                            if content:
                                collected_chunks.append(content)
                                yield content
                        # 스트리밍 완료 후 전체 응답 로깅 (선택적)
                        # full_response = "".join(collected_chunks)
                        # output_tokens = count_tokens(full_response, model=self.model_name)
                        # self.logger.debug(f"OpenAI 스트리밍 응답 완료 (출력 토큰: {output_tokens})")
                    return stream_generator()
                else:
                    # 일반 응답 처리
                    result = response.choices[0].message.content
                    output_tokens = count_tokens(result, model=self.model_name)
                    self.logger.debug(f"OpenAI 응답 생성 완료 (출력 토큰: {output_tokens})")
                    return result

            # --- 이하 다른 제공자(anthropic, huggingface) 생성 코드는 생략 ---

        except Exception as e:
            # 모든 종류의 오류 처리
            self.logger.error(f"{self.provider} 텍스트 생성 중 오류 발생: {str(e)}", exc_info=True)
            # 오류 발생 시 사용자에게 오류 메시지 반환 (스트리밍 아닐 때)
            if not should_stream:
                return f"오류 발생: {str(e)}"
            else:
                # 스트리밍 중 오류 발생 시 처리 (예: 빈 제너레이터 반환 또는 예외 발생)
                raise # 또는 return iter([])

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        제공된 텍스트 목록에 대한 임베딩 벡터를 생성합니다.

        현재 구현은 OpenAI와 HuggingFace(로컬 Sentence Transformers)만 지원합니다.
        로컬 모델이나 Ollama를 사용한 임베딩 생성은 지원하지 않습니다.

        Args:
            texts: 임베딩을 생성할 텍스트 문자열 목록.

        Returns:
            각 텍스트에 해당하는 임베딩 벡터 목록.

        Raises:
            ValueError: 현재 provider가 임베딩 생성을 지원하지 않는 경우.
            ImportError: 필요한 라이브러리가 설치되지 않은 경우.
        """
        self.logger.info(f"{len(texts)}개의 텍스트에 대한 임베딩 생성 시작")

        try:
            if self.provider == "openai":
                # OpenAI 임베딩 API 사용
                if not self.client:
                     raise RuntimeError("OpenAI 클라이언트가 초기화되지 않았습니다.")

                # 설정에서 임베딩 모델 이름 가져오기 (기본값: text-embedding-ada-002)
                embedding_model_name = self.config.get("embedding", {}).get("model_name", "text-embedding-ada-002")
                batch_size = self.config.get("embedding", {}).get("batch_size", 100) # 배치 크기 설정
                all_embeddings = []

                # 배치 단위로 처리하여 API 제한 및 메모리 문제 방지
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    response = self.client.embeddings.create(
                        model=embedding_model_name,
                        input=batch
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)

                self.logger.debug(f"OpenAI 임베딩 생성 완료 ({len(all_embeddings)}개 벡터)")
                return all_embeddings

            elif self.provider == "huggingface":
                 # 로컬 Sentence Transformers 모델 사용 (기존 로직 유지)
                 # 이 부분은 provider가 'huggingface'이고 use_local=True일 때 동작합니다.
                 # provider='local'과 구분 필요. 설정 파일을 통해 명확히 해야 함.
                embedding_model_name = self.config.get("embedding", {}).get("model_name", "sentence-transformers/all-mpnet-base-v2")

                try:
                    from sentence_transformers import SentenceTransformer
                    # 임베딩 모델 인스턴스 생성 (캐싱 가능)
                    if not hasattr(self, '_embedding_model_instance'):
                         self._embedding_model_instance = SentenceTransformer(embedding_model_name)

                    embeddings = self._embedding_model_instance.encode(texts, show_progress_bar=True)
                    embeddings_list = embeddings.tolist() # Numpy 배열을 리스트로 변환

                    self.logger.debug(f"로컬 Sentence Transformer 임베딩 생성 완료 ({len(embeddings_list)}개 벡터)")
                    return embeddings_list

                except ImportError:
                    self.logger.error("sentence-transformers 패키지가 설치되지 않았습니다. 'pip install sentence-transformers'로 설치해주세요.")
                    raise

            # --- 로컬 모델 및 Ollama 임베딩 지원 ---
            # elif self.provider == "local":
            #     self.logger.warning("현재 로컬 모델을 사용한 임베딩 생성은 지원되지 않습니다.")
            #     raise ValueError("로컬 모델 임베딩은 지원되지 않습니다.")
            # elif self.provider == "ollama":
            #     # Ollama API가 임베딩을 지원하는 경우 관련 로직 추가
            #     # 예: requests.post(ollama_embedding_endpoint, json=...)
            #     self.logger.warning("현재 Ollama API를 사용한 임베딩 생성은 구현되지 않았습니다.")
            #     raise ValueError("Ollama 임베딩은 지원되지 않습니다.")

            else:
                # 지원하지 않는 provider인 경우
                self.logger.error(f"'{self.provider}' 제공자에서는 임베딩 생성을 지원하지 않습니다.")
                raise ValueError(f"임베딩 미지원 제공자: {self.provider}")

        except Exception as e:
            # 임베딩 생성 중 발생한 모든 오류 처리
            self.logger.error(f"임베딩 생성 중 오류 발생: {str(e)}", exc_info=True)
            raise # 오류를 다시 발생시켜 호출자에게 알림 