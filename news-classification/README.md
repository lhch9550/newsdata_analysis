# KPF-BERT 사전학습 모델을 활용한 뉴스 주제 분류 모델 구현

### 1. 모델 및 데이터

한국언론진흥재단이 개발한 kpf-BERT 모델의 cls 토큰을 기반으로 뉴스기사에 대한 주제 분류 task를 수행할 수 있는 분류 모델을 위한 코드를 작성
- 기존 kpf-BERT 모델과 토크나이저는 git을 통해 다운로드: [빅카인즈 참조](https://github.com/KPF-bigkinds/BIGKINDS-LAB/blob/main/KPF-BERT-CLS/README.md)
- 학습데이터는 '국립국어원의 뉴스 분류 데이터'를 활용하였음
- 학습데이터는 기사내용과 분류명을 넣어 제작하였음. 본 실험에서는 세분류에 대해서만 학습을 진행함
<img width="658" alt="Image" src="https://github.com/user-attachments/assets/7db9664f-ec5f-41ee-89bb-ad9b508d8214" /> 

### 2. 실험 방법

두 가지 방식으로 모델 학습시켰음
1. top3가 아닌 하나의 라벨로 분류해주는 모델 학습(multiclass classification)
- 분류 모델의 정확도를 위해서 2년도 1~3월 뉴스기사를 활용(백만건) 중 라벨이 하나인 기사만을 필터링하여 2 epoch 학습을 진행함 
2. 한 기사를 여러 라벨로 분류해주는 모델 학습(multilabel classification)
- 학습 시간이 방대한 관계로 22년도 1~3월 뉴스기사를 활용(백만건)하여 2 epoch 학습을 진행함
