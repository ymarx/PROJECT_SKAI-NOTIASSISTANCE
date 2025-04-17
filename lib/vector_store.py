"""
SKAI-NotiAssistance를 위한 벡터 저장소 관리 모듈입니다.

이 모듈은 텍스트 데이터의 벡터 임베딩을 효율적으로 저장하고 검색하는 기능을 제공합니다.
다양한 벡터 데이터베이스 백엔드(Chroma, Pinecone, FAISS)를 지원하며,
설정 파일을 통해 사용할 백엔드와 관련 설정을 지정할 수 있습니다.

주요 기능:
- 텍스트와 메타데이터, 임베딩 벡터 저장
- 벡터 유사도 기반 검색
- 다양한 백엔드 지원 (설정 기반 선택)
- 영속성 관리 (로컬 파일 또는 클라우드 서비스)

참고: 임베딩 벡터 생성 자체는 이 모듈의 책임이 아니며, 
      `ModelManager` 등 외부에서 생성된 임베딩 벡터를 받아 저장/검색합니다.
"""

import os
from typing import Any, Dict, List, Optional, Union, Tuple

# FAISS 사용 시 필요할 수 있는 라이브러리 임포트 (ImportError 방지)
try:
    import faiss
    import numpy as np
    import pickle
except ImportError:
    faiss = None # FAISS 라이브러리가 없어도 다른 provider 사용 가능
    np = None
    pickle = None

# Chroma 사용 시 필요할 수 있는 라이브러리 임포트
try:
    import chromadb
except ImportError:
    chromadb = None

# Pinecone 사용 시 필요할 수 있는 라이브러리 임포트
try:
    import pinecone
except ImportError:
    pinecone = None


from ..utils.logger import get_logger

logger = get_logger(__name__)

class VectorStore:
    """
    텍스트 임베딩 저장을 위한 벡터 데이터베이스 인터페이스입니다.

    Chroma, Pinecone, FAISS 등 다양한 벡터 저장소 백엔드를 지원하는
    통합 인터페이스를 제공합니다. 설정(`config`)에 따라 적절한 백엔드를 초기화하고 사용합니다.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        벡터 저장소를 초기화합니다.

        설정 딕셔너리를 기반으로 사용할 provider(백엔드)와 관련 설정을 로드하고,
        해당 provider의 클라이언트 또는 인덱스를 설정합니다.

        Args:
            config: 벡터 저장소 설정을 담은 딕셔너리. 필수 및 선택적 키는 다음과 같습니다:
                - provider (str): 사용할 벡터 저장소 종류 ('chroma', 'pinecone', 'faiss'). 기본값: 'chroma'.
                - collection_name (str): 데이터를 저장할 컬렉션(테이블 또는 인덱스)의 이름. 기본값: 'default'.
                - distance_metric (str): 벡터 간 거리 계산 방식 ('cosine', 'l2', 'ip'). 기본값: 'cosine'.
                                       FAISS는 'l2'만 지원할 수 있습니다.
                - persist_directory (str): 로컬 파일 기반 provider(Chroma, FAISS) 사용 시
                                          데이터를 저장할 디렉토리 경로. 기본값: './data/embeddings'.
                - dimensions (int): FAISS 또는 Pinecone 사용 시 벡터의 차원 수.
                                   사용하는 임베딩 모델의 출력 차원과 일치해야 합니다. (예: OpenAI ada-002는 1536)
                - api_key (str): Pinecone 사용 시 필요한 API 키. 환경 변수 `PINECONE_API_KEY` 에서도 읽어옵니다.
                - environment (str): Pinecone 사용 시 필요한 환경 이름. 환경 변수 `PINECONE_ENVIRONMENT` 에서도 읽어옵니다.
        """
        self.config = config
        # 사용할 벡터 저장소 제공자 설정 (기본값: 'chroma')
        self.provider = config.get("provider", "chroma")
        # 데이터 컬렉션 이름 설정 (기본값: 'default')
        self.collection_name = config.get("collection_name", "default")
        # 벡터 거리 계산 방식 설정 (기본값: 'cosine')
        self.distance_metric = config.get("distance_metric", "cosine")
        # 로컬 데이터 저장 경로 설정 (기본값: './data/embeddings')
        self.persist_directory = config.get("persist_directory", "./data/embeddings")
        # 백엔드 클라이언트 또는 인덱스 객체 초기화
        self.client = None
        self.collection = None # Chroma에서 사용
        self.metadata = None # FAISS에서 사용
        self.index_file = None # FAISS 인덱스 파일 경로
        self.metadata_file = None # FAISS 메타데이터 파일 경로
        # 로거 설정
        self.logger = get_logger(f"{__name__}.{self.provider}")

        # 데이터 저장 디렉토리가 없으면 생성 (Chroma, FAISS 사용 시)
        if self.provider in ["chroma", "faiss"]:
            if not os.path.exists(self.persist_directory):
                try:
                    os.makedirs(self.persist_directory, exist_ok=True)
                    self.logger.info(f"데이터 저장 디렉토리 생성: {self.persist_directory}")
                except OSError as e:
                    self.logger.error(f"데이터 저장 디렉토리 생성 실패: {e}")
                    raise

        # 설정된 provider에 따라 클라이언트 또는 인덱스 초기화
        self._initialize_client()

    def _initialize_client(self):
        """설정된 provider에 따라 적절한 벡터 저장소 클라이언트/인덱스를 초기화합니다."""
        self.logger.info(f"{self.provider} 벡터 저장소를 초기화합니다.")

        if self.provider == "chroma":
            if chromadb is None:
                self.logger.error("chromadb 패키지가 설치되지 않았습니다. 'pip install chromadb'로 설치해주세요.")
                raise ImportError("ChromaDB 사용을 위해 chromadb 패키지가 필요합니다.")
            try:
                # ChromaDB PersistentClient를 사용하여 로컬 디스크에 데이터 저장
                self.client = chromadb.PersistentClient(path=self.persist_directory)
                # 지정된 이름의 컬렉션을 가져오거나 생성합니다.
                # metadata에 distance_function을 지정하여 거리 계산 방식 설정 (Chroma 0.4.x 기준)
                # 최신 버전의 ChromaDB는 컬렉션 생성 시 distance metric 지정 방식이 다를 수 있습니다.
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": self.distance_metric} # 최신 Chroma 방식 확인 필요
                )
                self.logger.info(f"ChromaDB 컬렉션에 연결되었습니다: {self.collection_name}")
            except Exception as e:
                self.logger.error(f"ChromaDB 클라이언트 초기화 중 오류 발생: {str(e)}")
                raise

        elif self.provider == "pinecone":
            if pinecone is None:
                 self.logger.error("pinecone-client 패키지가 설치되지 않았습니다. 'pip install pinecone-client'로 설치해주세요.")
                 raise ImportError("Pinecone 사용을 위해 pinecone-client 패키지가 필요합니다.")
            try:
                # 설정 또는 환경 변수에서 Pinecone API 키와 환경 이름 가져오기
                api_key = self.config.get("api_key", os.environ.get("PINECONE_API_KEY"))
                environment = self.config.get("environment", os.environ.get("PINECONE_ENVIRONMENT"))

                if not api_key or not environment:
                    raise ValueError("Pinecone 사용을 위한 API 키 또는 환경 정보가 부족합니다.")

                # Pinecone 클라이언트 초기화
                pinecone.init(api_key=api_key, environment=environment)

                # 지정된 이름의 인덱스가 존재하는지 확인하고, 없으면 새로 생성
                if self.collection_name not in pinecone.list_indexes():
                    # 인덱스 생성 시 벡터 차원 수 필요
                    dimension = self.config.get("dimensions")
                    if dimension is None:
                        raise ValueError("Pinecone 인덱스 생성을 위해 'dimensions' 설정이 필요합니다.")
                    self.logger.info(f"Pinecone 인덱스 생성 시작: {self.collection_name} (차원: {dimension})")
                    pinecone.create_index(
                        name=self.collection_name,
                        dimension=dimension,
                        metric=self.distance_metric # Pinecone에서 지원하는 메트릭 사용
                    )
                    self.logger.info(f"Pinecone 인덱스 생성 완료: {self.collection_name}")

                # 생성되었거나 기존에 있던 인덱스에 연결
                self.client = pinecone.Index(self.collection_name)
                self.logger.info(f"Pinecone 인덱스에 연결되었습니다: {self.collection_name}")
            except Exception as e:
                self.logger.error(f"Pinecone 클라이언트 초기화 중 오류 발생: {str(e)}")
                raise

        elif self.provider == "faiss":
            if faiss is None or np is None or pickle is None:
                self.logger.error("FAISS 사용을 위해 faiss-cpu(또는 faiss-gpu), numpy, pickle 패키지가 필요합니다. 'pip install faiss-cpu numpy'로 설치해주세요.")
                raise ImportError("FAISS, numpy, pickle 라이브러리가 필요합니다.")
            try:
                # FAISS 인덱스 파일과 메타데이터 파일 경로 설정
                self.index_file = os.path.join(self.persist_directory, f"{self.collection_name}.index")
                self.metadata_file = os.path.join(self.persist_directory, f"{self.collection_name}.meta")

                # 기존 인덱스 파일이 있으면 로드, 없으면 새로 생성
                if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
                    self.logger.info(f"저장된 FAISS 인덱스 로드 시도: {self.index_file}")
                    self.client = faiss.read_index(self.index_file)
                    with open(self.metadata_file, 'rb') as f:
                        self.metadata = pickle.load(f)
                    self.logger.info(f"FAISS 인덱스 로드 완료 ({self.client.ntotal} 항목)")
                else:
                    self.logger.info("저장된 FAISS 인덱스 없음. 새로 생성합니다.")
                    # 새 인덱스 생성 시 벡터 차원 수 필요
                    dimension = self.config.get("dimensions")
                    if dimension is None:
                        raise ValueError("FAISS 인덱스 생성을 위해 'dimensions' 설정이 필요합니다.")

                    # FAISS 인덱스 생성 (IndexFlatL2는 L2 거리(유클리드 거리) 사용)
                    # 다른 인덱스 유형 (예: IndexFlatIP - 내적)도 사용 가능하나, 거리 계산 방식 통일 필요
                    # self.distance_metric에 따라 다른 Index 유형 사용하도록 확장 가능
                    self.client = faiss.IndexFlatL2(dimension)
                    # 메타데이터는 리스트들의 딕셔너리로 관리 (ID, 원본 텍스트, 메타데이터)
                    self.metadata = {"ids": [], "documents": [], "metadata": []}
                    self.logger.info(f"새 FAISS 인덱스 생성 완료 (차원: {dimension})")
            except Exception as e:
                self.logger.error(f"FAISS 인덱스 초기화 중 오류 발생: {str(e)}")
                raise
        else:
            # 지원하지 않는 provider인 경우 오류 발생
            raise ValueError(f"지원하지 않는 벡터 저장소 제공자입니다: {self.provider}")

    def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        텍스트, 해당 임베딩 벡터, 그리고 메타데이터를 벡터 저장소에 추가합니다.

        Args:
            texts: 저장할 원본 텍스트 문서 목록.
            embeddings: 각 텍스트에 해당하는 임베딩 벡터 목록 (ModelManager 등에서 생성).
                        벡터의 차원은 초기화 시 설정된 차원과 일치해야 합니다.
            ids: 각 문서에 할당할 고유 ID 목록 (선택 사항). 제공되지 않으면 UUID로 자동 생성됩니다.
                 Chroma, Pinecone은 ID를 직접 사용하지만, FAISS는 내부 인덱스와 매핑하여 관리합니다.
            metadata: 각 문서에 대한 추가 정보(딕셔너리) 목록 (선택 사항).
                      Chroma, Pinecone은 메타데이터 필터링을 지원합니다. FAISS는 메타데이터를 별도로 저장합니다.

        Returns:
            추가된 문서들의 ID 목록.

        Raises:
            ValueError: 입력 데이터(texts, embeddings, ids, metadata)의 길이가 맞지 않는 경우.
            RuntimeError: 벡터 저장소 클라이언트/인덱스가 초기화되지 않은 경우.
            Exception: 각 provider별 API 호출 또는 파일 작업 중 오류 발생 시.
        """
        import uuid

        # 입력 데이터 길이 검증
        if not (len(texts) == len(embeddings)):
            raise ValueError("텍스트 목록과 임베딩 목록의 길이가 일치해야 합니다.")
        if ids is not None and len(texts) != len(ids):
            raise ValueError("ID 목록이 제공된 경우, 텍스트 목록과 길이가 일치해야 합니다.")
        if metadata is not None and len(texts) != len(metadata):
             raise ValueError("메타데이터 목록이 제공된 경우, 텍스트 목록과 길이가 일치해야 합니다.")

        # ID가 제공되지 않으면 UUID 생성
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]

        # 메타데이터가 제공되지 않으면 빈 딕셔너리 리스트 생성
        if metadata is None:
            metadata = [{} for _ in range(len(texts))]

        self.logger.info(f"{self.provider}에 {len(texts)}개의 문서를 추가합니다.")

        try:
            if self.provider == "chroma":
                if self.collection is None:
                    raise RuntimeError("ChromaDB 컬렉션이 초기화되지 않았습니다.")
                # ChromaDB에 문서 추가
                self.collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadata,
                    ids=ids
                )
            elif self.provider == "pinecone":
                if self.client is None:
                     raise RuntimeError("Pinecone 인덱스가 초기화되지 않았습니다.")
                # Pinecone에 저장할 형식으로 데이터 변환
                vectors_to_upsert = []
                for i in range(len(texts)):
                    # 메타데이터에 원본 텍스트 포함 (Pinecone은 text 필드를 메타데이터로 관리)
                    pinecone_meta = {"text": texts[i], **metadata[i]}
                    vectors_to_upsert.append((ids[i], embeddings[i], pinecone_meta))

                # Pinecone 인덱스에 데이터 upsert (update or insert)
                self.client.upsert(vectors=vectors_to_upsert)
            elif self.provider == "faiss":
                if self.client is None or self.metadata is None or np is None or pickle is None:
                    raise RuntimeError("FAISS 인덱스 또는 관련 라이브러리가 초기화되지 않았습니다.")

                # 임베딩 리스트를 numpy 배열로 변환 (FAISS는 numpy 배열 필요)
                # FAISS는 float32 타입을 요구합니다.
                embeddings_array = np.array(embeddings, dtype=np.float32)

                # FAISS 인덱스에 벡터 추가
                self.client.add(embeddings_array)

                # 메타데이터 저장소 업데이트
                self.metadata["ids"].extend(ids)
                self.metadata["documents"].extend(texts)
                self.metadata["metadata"].extend(metadata)

                # 변경된 인덱스와 메타데이터를 파일에 저장 (영속성 확보)
                faiss.write_index(self.client, self.index_file)
                with open(self.metadata_file, 'wb') as f:
                    pickle.dump(self.metadata, f)

            self.logger.debug(f"{len(texts)}개의 문서가 성공적으로 추가되었습니다.")
            return ids

        except Exception as e:
            self.logger.error(f"문서 추가 중 오류 발생: {str(e)}")
            raise

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        주어진 쿼리 임베딩과 가장 유사한 문서를 벡터 저장소에서 검색합니다.

        Args:
            query_embedding: 검색할 쿼리의 임베딩 벡터 (ModelManager 등에서 생성).
                             벡터의 차원은 저장된 벡터들의 차원과 일치해야 합니다.
            top_k: 반환할 최대 결과 수. 기본값: 5.
            filter_metadata: 메타데이터 필터링 조건 (딕셔너리 형태). (선택 사항)
                             Chroma, Pinecone에서 지원됩니다. FAISS는 현재 필터링 미지원.
                             예: {"category": "report", "year": 2023}

        Returns:
            검색된 문서 목록. 각 문서는 다음 키를 포함하는 딕셔너리입니다:
            - id (str): 문서 ID
            - text (str): 원본 텍스트 (FAISS, Pinecone의 경우 메타데이터에서 추출)
            - metadata (dict): 추가 메타데이터
            - score (float): 쿼리와의 유사도 점수 (높을수록 유사).
                             Chroma/FAISS는 거리를 반환하므로 유사도로 변환될 수 있습니다.
                             Pinecone은 유사도 점수를 직접 반환합니다.

        Raises:
            RuntimeError: 벡터 저장소 클라이언트/인덱스가 초기화되지 않은 경우.
            ValueError: 지원하지 않는 provider 이거나, FAISS에서 필터링 시도 시.
            Exception: 각 provider별 API 호출 또는 파일 작업 중 오류 발생 시.
        """
        self.logger.debug(f"{self.provider}에서 상위 {top_k}개의 유사 문서를 검색합니다.")

        try:
            if self.provider == "chroma":
                if self.collection is None:
                    raise RuntimeError("ChromaDB 컬렉션이 초기화되지 않았습니다.")

                # ChromaDB 쿼리 실행
                # where 필터 사용 가능 (filter_metadata 형식은 ChromaDB 문서 참조)
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=filter_metadata # 메타데이터 필터 적용
                )

                # ChromaDB 결과를 표준 형식으로 변환
                formatted_results = []
                if results and results.get("ids") and results["ids"][0]:
                    for i in range(len(results["ids"][0])):
                        # ChromaDB는 거리를 반환할 수 있음. 유사도 점수로 변환 필요 시 로직 추가
                        score = results["distances"][0][i] if results.get("distances") else 0.0
                        formatted_results.append({
                            "id": results["ids"][0][i],
                            "text": results["documents"][0][i] if results.get("documents") else "",
                            "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                            "score": score # 필요시 1/(1+distance) 등으로 변환
                        })
                return formatted_results

            elif self.provider == "pinecone":
                if self.client is None:
                     raise RuntimeError("Pinecone 인덱스가 초기화되지 않았습니다.")

                # Pinecone 쿼리 실행
                # filter 파라미터 사용 가능 (filter_metadata 형식은 Pinecone 문서 참조)
                results = self.client.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True, # 메타데이터(텍스트 포함) 반환 요청
                    filter=filter_metadata # 메타데이터 필터 적용
                )

                # Pinecone 결과를 표준 형식으로 변환
                formatted_results = []
                if results and results.get("matches"):
                    for match in results["matches"]:
                        # Pinecone 메타데이터에서 'text' 필드 추출
                        text = match["metadata"].pop("text", "") if match.get("metadata") else ""
                        formatted_results.append({
                            "id": match["id"],
                            "text": text,
                            "metadata": match.get("metadata", {}),
                            "score": match["score"] # Pinecone은 유사도 점수 반환
                        })
                return formatted_results

            elif self.provider == "faiss":
                if self.client is None or self.metadata is None or np is None:
                     raise RuntimeError("FAISS 인덱스 또는 관련 라이브러리가 초기화되지 않았습니다.")

                # FAISS는 메타데이터 필터링을 직접 지원하지 않음
                if filter_metadata:
                     self.logger.warning("FAISS provider는 메타데이터 필터링을 지원하지 않습니다. 필터는 무시됩니다.")
                     # 필터링 필요 시, 검색 후 결과에서 직접 필터링하는 로직 추가 가능

                # 쿼리 임베딩을 numpy 배열로 변환 (FAISS는 numpy 배열 필요)
                query_array = np.array([query_embedding], dtype=np.float32)

                # FAISS 인덱스 검색 (거리와 인덱스 반환)
                distances, indices = self.client.search(query_array, top_k)

                # FAISS 결과를 표준 형식으로 변환
                formatted_results = []
                if len(indices) > 0:
                    for i in range(len(indices[0])):
                        idx = indices[0][i] # 검색된 내부 인덱스
                        dist = distances[0][i] # 거리 (L2)

                        # 유효한 인덱스이고, 메타데이터 범위 내에 있는지 확인
                        if idx != -1 and idx < len(self.metadata["ids"]):
                            # 거리를 유사도 점수로 변환 (예: 1 / (1 + 거리))
                            # 거리가 0일 경우 유사도는 1, 거리가 무한대일 경우 유사도는 0에 가까워짐
                            similarity = 1.0 / (1.0 + float(dist))

                            formatted_results.append({
                                "id": self.metadata["ids"][idx],
                                "text": self.metadata["documents"][idx], # 저장된 원본 텍스트
                                "metadata": self.metadata["metadata"][idx], # 저장된 메타데이터
                                "score": similarity
                            })
                return formatted_results

            else:
                raise ValueError(f"지원하지 않는 벡터 저장소 제공자입니다: {self.provider}")

        except Exception as e:
            self.logger.error(f"쿼리 실행 중 오류 발생: {str(e)}")
            raise

    def delete(self, ids: List[str]) -> bool:
        """
        주어진 ID 목록에 해당하는 문서를 벡터 저장소에서 삭제합니다.

        Args:
            ids: 삭제할 문서 ID 목록.

        Returns:
            삭제 작업 성공 여부 (True/False).
            FAISS의 경우, 삭제는 인덱스 재구성을 의미할 수 있어 비용이 클 수 있습니다.

        Raises:
            RuntimeError: 벡터 저장소 클라이언트/인덱스가 초기화되지 않은 경우.
            NotImplementedError: FAISS provider에서 삭제 기능이 복잡하여 아직 구현되지 않았을 경우.
            Exception: 각 provider별 API 호출 또는 파일 작업 중 오류 발생 시.
        """
        self.logger.info(f"{self.provider}에서 {len(ids)}개의 문서를 삭제합니다.")

        try:
            if self.provider == "chroma":
                if self.collection is None:
                    raise RuntimeError("ChromaDB 컬렉션이 초기화되지 않았습니다.")
                # ChromaDB에서 ID 기반 삭제
                self.collection.delete(ids=ids)
                self.logger.debug(f"ChromaDB에서 {len(ids)}개 문서 삭제 완료.")
                return True

            elif self.provider == "pinecone":
                if self.client is None:
                     raise RuntimeError("Pinecone 인덱스가 초기화되지 않았습니다.")
                # Pinecone에서 ID 기반 삭제
                self.client.delete(ids=ids)
                self.logger.debug(f"Pinecone에서 {len(ids)}개 문서 삭제 완료.")
                return True

            elif self.provider == "faiss":
                # FAISS는 직접적인 ID 기반 삭제를 효율적으로 지원하지 않습니다.
                # 삭제를 구현하려면, 삭제할 ID를 제외한 모든 벡터와 메타데이터를 이용해
                # 인덱스를 새로 생성하고 저장해야 합니다. 이는 매우 비효율적일 수 있습니다.
                self.logger.warning("FAISS provider의 효율적인 문서 삭제는 현재 구현되지 않았습니다. "
                                  "삭제가 필요하면 전체 인덱스를 재생성해야 할 수 있습니다.")
                # 실제 삭제 로직 (비효율적):
                # 1. self.metadata에서 삭제할 ids에 해당하는 인덱스 찾기
                # 2. 해당 인덱스를 제외한 벡터들을 self.client에서 추출 (reconstruct 사용 가능)
                # 3. 제외된 메타데이터 생성
                # 4. 새로운 FAISS 인덱스 생성 및 벡터 추가
                # 5. 새로운 인덱스 및 메타데이터 저장
                raise NotImplementedError("FAISS provider의 삭제 기능은 현재 지원되지 않습니다.")
                # return False # 임시 반환

            else:
                raise ValueError(f"지원하지 않는 벡터 저장소 제공자입니다: {self.provider}")

        except Exception as e:
            self.logger.error(f"문서 삭제 중 오류 발생: {str(e)}")
            raise # 또는 return False

    def get_collection_info(self) -> Dict[str, Any]:
         """
         현재 연결된 컬렉션(인덱스)의 정보를 반환합니다.

         Returns:
             컬렉션 정보를 담은 딕셔너리 (예: 문서 수).
             provider 마다 반환되는 정보가 다를 수 있습니다.
         """
         info = {"provider": self.provider, "collection_name": self.collection_name}
         try:
             if self.provider == "chroma" and self.collection:
                 info["count"] = self.collection.count()
             elif self.provider == "pinecone" and self.client:
                 # Pinecone 클라이언트의 describe_index_stats() 사용
                 stats = self.client.describe_index_stats()
                 info["count"] = stats.get('total_vector_count', 0)
                 info["dimensions"] = stats.get('dimension', 0)
             elif self.provider == "faiss" and self.client:
                 info["count"] = self.client.ntotal
                 info["dimensions"] = self.client.d
         except Exception as e:
             self.logger.warning(f"컬렉션 정보 조회 중 오류: {e}")
             info["error"] = str(e)
         return info 