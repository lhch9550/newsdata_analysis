### 1. 요약 모델

BERTSum 모델은 BERT의 구조 위에 두 개의 inter-sentence Transformer 레이어를 추가한 형태로 설계되었다. 이를 더욱 최적화하여 BertSumExt 요약 모델로 활용할 수 있다.
Pre-trained BERT를 문서 요약(task-specific) 모델로 활용하기 위해서는 여러 개의 문장을 하나의 입력으로 처리할 수 있어야 하며, 각 문장에 대한 개별적인 정보를 효과적으로 추출할 수 있도록 입력 형식을 조정해야 한다. 이를 위해, 입력 문서의 각 문장 앞에 [CLS] 토큰을 삽입하고, 문장마다 고유한 segment embeddings을 부여하는 interval segment embeddings 기법을 적용한다. 이를 통해 BERT가 문서 내 개별 문장의 관계를 보다 정교하게 학습할 수 있도록 한다.
기존의 BERT를 summariaztion에 바로 적용하기에 BERT는 MLM으로 훈련되기 때문에 출력 벡터가 토큰단위로 출력되게 됩니다. 이러한 한계를 극복하고자 요약 task에서는 문장 수준의 표현을 다루기 위해 BERT의 입력 데이터 형태를 수정하여 사용합니다.

![Image](https://github.com/user-attachments/assets/6c55ebc3-640b-48da-bb10-3f5e1f49d26c)

본 실험에서는 해당 모델을 통해 요약된 뉴스 기사를 한국언론진흥재단에서 구축한 방대한 뉴스기사 말뭉치로 학습한 KPF-BERT를 이용하여 특히 뉴스기사 요약에 특화된 모델로 한국어 데이터 셋은 AI-HUB에서 제공하는 문서요약 텍스트를 사용하였음
뉴스 기사 3줄 요약인 KPF-BERTSum은 KPF-BERT Text Summarization의 준말로 BERT의 사전학습 모델을 이용한 텍스트 요약 모델이다.


### 2. KeyBERT를 통한 키워드 추출 모델

KeyBERT는 BERT를 기반으로 문서의 주제를 효과적으로 파악하고, 문서 내에서 의미적으로 중요한 키워드를 추출하는 모델이다. 
이를 위해 문서의 임베딩을 생성하고, 다양한 N-gram 단위를 고려하여 단어 및 구(phrase)를 벡터화한 후, 문서와의 코사인 유사도를 계산하여 가장 관련성이 높은 키워드를 선정한다. 
즉, 문서의 의미를 가장 잘 대표하는 단어들을 도출함으로써, 효과적인 키워드 추출을 가능하게 한다.

관련하여 [다음 페이지](https://heeya-stupidbutstudying.tistory.com/entry/DL-keyword-extraction-with-KeyBERT-%EA%B0%9C%EC%9A%94%EC%99%80-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-1)를 참조하였다. 

KeyBERT의 General Flow는 다음과 같다. 

<img width="832" alt="image" src="https://github.com/user-attachments/assets/6cbad1c6-bbc9-47e4-8968-403e8878aefb" /># BERT를 이용한 뉴스기사 요약 및 핵심 키워드 추출

1) Document-level representation (by document embeddings extracted with BERT)
2) Phrase-level representation (by word embeddings extracted for N-gram words/phrases and BERT)
3) Use of cosine similarity to find the words/phrases that are most similar to the document
  - (optional) MMR or Max Sum Similarity
4) Extraction of words/phrases that best describe the entire document

### MMR (Maximal Marginal Relevance)

MMR은 검색 시스템에서 검색 결과의 다양성과 연관성을 균형 있게 조정하는 기법으로, 텍스트 요약 및 정보 검색에서 중복성을 줄이고 결과의 다양성을 극대화하는 데 활용된다.
KeyBERT에서 MMR을 적용하는 과정은 다음과 같다:

문서와 가장 유사한 키워드를 우선 선택한다.
이후, 문서와 유사하면서도 이미 선택된 키워드와 중복되지 않는 새로운 후보 키워드를 반복적으로 선택하여 최종 키워드 집합을 구성한다.

MMR 방식은 키워드의 다양성을 확보하는 동시에, 문서와의 연관성을 유지하는 데 유용하다.

### Max Sum Similarity

Max Sum Similarity 기법은 키워드-문서 간의 유사도를 최대화하면서도, 키워드 간의 유사성을 최소화하는 방식으로 의미적으로 풍부한 키워드 집합을 생성하는 방법이다.
문서와 가장 유사한 키워드 후보군을 선정한다.
상위 n개의 키워드를 조합하여 가능한 모든 키워드 쌍에 대해 코사인 유사도를 계산한다.
이 중에서 키워드-키워드 간의 유사도가 가장 낮은 조합을 최종 키워드로 선택한다.
이를 통해 서로 중복되는 키워드를 제거하고, 보다 다양한 의미를 포함하는 키워드 세트를 구축할 수 있다.

### 2. 실험 방법
