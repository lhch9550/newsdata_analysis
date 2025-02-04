# Market Intelligence:프로젝트 개요
Market Intelligence는 에듀테크 시장의 거시적 및 미시적 동향 정보를 **데이터 기반으로 제공**하기 위한 기술 연구 프로젝트입니다. 기존 마켓 분석에서 가장 큰 걸림돌이었던 **데이터 양과 라벨링 문제**를 해결하기 위해, 초기 단계에서는 **비정형 텍스트 데이터**를 중심으로 데이터를 수집하고 분석하는 방식을 채택하고 있습니다.

## 주요 기능
Market Intelligence는 다음 **3가지 핵심 기능**을 중심으로 연구 및 개발이 진행됩니다.

### 1. 개체명 인식 (Named Entity Recognition, NER)
- 사전 학습된 **개체명 인식 모델**을 활용하여 텍스트에서 **인물, 기관, 기업 등의 정보를 추출**합니다.
- 수집된 데이터를 구조화하여, 보다 정밀한 인사이트를 도출할 수 있도록 설계합니다.

### 2. 마켓 애널리틱스 (Market Analytics)
- **토픽 모델링, 워드 임베딩, 노드 임베딩** 등 다양한 분석 기법을 활용하여 **대규모 텍스트 데이터에서 핵심 키워드 네트워크를 구축**합니다.
- 추출된 데이터를 **시각화**하여, 시장의 흐름과 트렌드를 쉽게 파악할 수 있도록 합니다.

### 3. RAG 기반 챗봇 서비스
- **Retrieval-Augmented Generation (RAG)** 기법을 활용하여, **대형 언어 모델(LLM)과 프롬프트 엔지니어링을 결합**한 **챗봇 서비스**를 제공합니다.
- 유저의 질문에 대해 **즉각적이고 정확한 응답**을 생성할 수 있도록 설계합니다.

## 기술 스택
- **자연어 처리(NLP):** Named Entity Recognition (NER), 토픽 모델링, 워드 임베딩
- **데이터 분석:** 네트워크 분석, 시각화 기법 적용
- **대형 언어 모델(LLM):** RAG 기반 검색 및 질의응답 시스템 구축

```python
brotlipy==0.7.0
ConfigParser==7.1.0
cryptography==44.0.0
Cython==0.29.21
dl==0.1.0
HTMLParser==0.0.2
ipaddr==2.2.0
JPype1==1.5.0
keyring==21.2.1
kiwipiepy==0.20.3
kss.core==1.6.5
langchain_core==0.3.33
langchain_openai==0.3.3
lxml==4.5.2
numpy==1.24.4
ordereddict==1.1
pandas==2.0.3
protobuf==3.20.3
pyOpenSSL==25.0.0
python-dotenv==1.0.1
pytorch_lightning==2.4.0
scikit_learn==1.3.2
seqeval==1.2.2
streamlit==1.35.0
thread==2.0.5
torch==2.4.1
tqdm==4.66.1
transformers==4.35.2
wincertstore==0.2.1
xmlrpclib==1.0.1
```

## 기대 효과
- 에듀테크 시장의 거시적 및 미시적 동향을 데이터 기반으로 분석하여 **의사결정 지원**
- **비정형 데이터를 활용한 인사이트 도출**로 기존 마켓 분석의 한계 극복
- **최신 AI 기술을 활용한 정보 탐색 및 분석 자동화**

## 진행 상황
프로젝트는 단계적으로 개발 및 테스트를 진행 중이며, 지속적으로 업데이트될 예정입니다. 최신 개발 사항과 연구 결과는 문서화하여 공유할 예정입니다.

