# BERT Finetuing 이용한 뉴스기사 요약 + 핵심 키워드 추출

### 1. 모델 및 데이터

- 뉴스 기사 3줄 요약
    - KPF-BERTSum은 KPF-BERT Text Summarization의 준말로 BERT의 사전학습 모델을 이용한 텍스트 요약 모델이며 [PRESUMM모델](https://github.com/nlpyang/PreSumm)을 참조하여 한국어 문장의 요약추출을 구현한 한국어 요약 모델임
        
        ![image.png](attachment:07ad5850-a1e0-474e-8265-aaa253af3d9b:image.png)
        
    - 한국언론진흥재단에서 구축한 방대한 뉴스기사 말뭉치로 학습한 KPF-BERT를 이용하여 특히 뉴스기사 요약에 특화된 모델로 한국어 데이터 셋은 AI-HUB에서 제공하는 문서요약 텍스트를 사용하였음
- 핵심 키워드 도출
    - 

### 2. 실험 방법
