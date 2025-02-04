## 1. BERTSum 뉴스기사 모델 설명

BERTSum 모델은 BERT의 구조 위에 두 개의 inter-sentence Transformer 레이어를 추가한 형태로 설계되었다. 이를 더욱 최적화하여 BertSumExt 요약 모델로 활용할 수 있다.
Pre-trained BERT를 문서 요약(task-specific) 모델로 활용하기 위해서는 여러 개의 문장을 하나의 입력으로 처리할 수 있어야 하며, 각 문장에 대한 개별적인 정보를 효과적으로 추출할 수 있도록 입력 형식을 조정해야 한다. 
입력 문서의 각 문장 앞에 [CLS] 토큰을 삽입하고, 문장마다 고유한 segment embeddings을 부여하는 interval segment embeddings 기법을 적용한다. 이를 통해 BERT가 문서 내 개별 문장의 관계를 보다 정교하게 학습할 수 있도록 한다.
기존의 BERT를 summariaztion에 바로 적용하기에 BERT는 MLM으로 훈련되기 때문에 출력 벡터가 토큰단위로 출력되게 된다. 
이러한 한계를 극복하고자 요약 task에서는 문장 수준의 표현을 다루기 위해 BERT의 입력 데이터 형태를 수정하여 사용합니다.

![Image](https://github.com/user-attachments/assets/6c55ebc3-640b-48da-bb10-3f5e1f49d26c)

본 실험에서는 해당 모델을 통해 요약된 뉴스 기사를 한국언론진흥재단에서 구축한 방대한 뉴스기사 말뭉치로 학습한 KPF-BERT를 이용했다
특히 뉴스기사 요약에 특화된 모델로 한국어 데이터 셋은 AI-HUB에서 제공하는 문서요약 텍스트를 사용하였다.
뉴스 기사 3줄 요약인 KPF-BERTSum은 KPF-BERT Text Summarization의 준말로 BERT의 사전학습 모델을 이용한 텍스트 요약 모델이다.

### 2. 실험 방법

1. 데이터셋 및 전처리
- AI-HUB 한국어 요약 데이터셋을 활용하여 학습 진행.
- AI-HUB 문서요약 데이터셋을 뉴스 기사 데이터셋 형식에 맞춰 변환.

2. 토크나이저 및 인코딩
- kpfBERT 토크나이저 사용.
- PreSumm 방식으로 문장 인코딩 진행.

3. 모델 학습 및 예측
- 사전 훈련된 Kpf-BERT 모델을 기반으로 문장 추출 모델 생성.
- 후처리 레이어 추가하여 문장 중요도를 평가.
- pytorch-lightning을 이용하여 학습 진행. 학습된 모델을 활용하여 기사 내에서 가장 중요한 3개 문장을 자동 추출.
