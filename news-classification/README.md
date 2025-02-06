# KPF-BERT 사전학습 모델을 활용한 뉴스 주제 분류 모델

## 1. 모델 및 데이터
KPF-BERT 모델의 CLS 토큰을 기반으로 뉴스 기사에 대한 주제 분류(Task)를 수행할 수 있는 분류 모델입니다.

- 기존 KPF-BERT 모델과 토크나이저는 허깅페이스를 통해 다운로드:[빅카이즈 허깅페이스 링크](https://huggingface.co/KPF/KPF-bert-cls2) 참조
- 학습 데이터는 **국립국어원의 모두의 말뭉치의 신문 말뭉치** 데이터[모두의 말뭉치 링크](https://kli.korean.go.kr/corpus/main/requestMain.do)를 활용하였음
- 기사 내용과 분류명을 포함하여 학습 데이터를 제작
- 본 실험에서는 **세분류(Task-specific classification)**에 대해서만 학습을 진행
<img width="500" alt="Image" src="https://github.com/user-attachments/assets/7db9664f-ec5f-41ee-89bb-ad9b508d8214" /> 

## 2. 실험 방법
본 연구에서는 두 가지 방식으로 모델을 학습시켰습니다.

### 2.1 Multiclass Classification (단일 라벨 분류)
- 뉴스 기사를 하나의 라벨로 분류하는 모델
- 모델의 정확도를 높이기 위해 **2020년도 1~3월 뉴스 기사** 중 **단일 라벨 기사만을 필터링**하여 학습
- **2 Epoch** 동안 학습 진행(추가 학습 필요)

### 2.2 Multilabel Classification (다중 라벨 분류)
- 한 기사를 여러 개의 라벨로 분류하는 모델
- 학습 데이터가 방대하여 **2022년도 1~3월 뉴스 기사 1백만 건**만을 활용
- **2 Epoch** 동안 학습 진행(추가 학습 필요)

<img width="500" alt="image" src="https://github.com/user-attachments/assets/df851425-144d-4f7d-b232-2190ba4314d6" />

## 3. 기대사항
KPF-BERT 모델을 활용한 뉴스 주제 분류 모델을 통해 한국어 뉴스 기사의 주제별 자동 분류가 가능할 것으로 기대됩니다.
특히 '교육-시험' 분야에 대한 뉴스 기사들을 필터링할 수 있는 모델로서 활용할 예정입니다. 

