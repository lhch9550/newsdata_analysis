### 1. KeyBERT를 통한 키워드 추출 모델

KeyBERT는 BERT를 기반으로 문서의 주제를 효과적으로 파악하고, 문서 내에서 의미적으로 중요한 키워드를 추출하는 모델이다. 
이를 위해 문서의 임베딩을 생성하고, 다양한 N-gram 단위를 고려하여 단어 및 구(phrase)를 벡터화한 후, 문서와의 코사인 유사도를 계산하여 가장 관련성이 높은 키워드를 선정한다. 
즉, 문서의 의미를 가장 잘 대표하는 단어들을 도출함으로써ㅜ효과적인 키워드 추출을 가능하게 하는 원리이다. 

알고리즘 관련한 설명은 [다음 페이지](https://heeya-stupidbutstudying.tistory.com/entry/DL-keyword-extraction-with-KeyBERT-%EA%B0%9C%EC%9A%94%EC%99%80-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-1)를 참조하였다. 

KeyBERT의 General Flow는 다음과 같다. 

<img width="832" alt="image" src="https://github.com/user-attachments/assets/6cbad1c6-bbc9-47e4-8968-403e8878aefb" />

1) Document-level representation (by document embeddings extracted with BERT)
2) Phrase-level representation (by word embeddings extracted for N-gram words/phrases and BERT)
3) Use of cosine similarity to find the words/phrases that are most similar to the document - (optional) MMR 또는 Max Sum Similarity
4) Extraction of words/phrases that best describe the entire document

### MMR (Maximal Marginal Relevance)

MMR이란 검색 엔진 내에서 본문 검색(text retrieval) 관련하여 검색(query)에 따른 결과(document)의 다양성과 연관성을 control하는 방법론이다.

<img width="788" alt="image" src="https://github.com/user-attachments/assets/e963940b-0abc-40de-9b20-0e8674eb5b2f" />

### Max Sum Similarity

Max Sum Similarity 기법은 키워드-문서 간의 유사도를 최대화하면서도, 키워드 간의 유사성을 최소화하는 방식으로 의미적으로 풍부한 키워드 집합을 생성하는 방법이다.
1) 문서와 가장 유사한 키워드 후보군을 선정한다.
2) 상위 n개의 키워드를 조합하여 가능한 모든 키워드 쌍에 대해 코사인 유사도를 계산한다.
3) 이 중에서 키워드-키워드 간의 유사도가 가장 낮은 조합을 최종 키워드로 선택한다.
4) 이를 통해 서로 중복되는 키워드를 제거하고, 보다 다양한 의미를 포함하는 키워드 세트를 구축할 수 있다.

### 2. Key-BERT 실험방법

1. 실험 환경
모델: 사용자 정의 BERT 모델 (Kpf-BERT)
토크나이저: 사용자 정의 BERT 토크나이저
형태소 분석기: Kiwi (한국어 형태소 분석기)

2. 전처리 (Preprocessing)
Kiwi 형태소 분석기를 이용하여 명사(NNG, NNP)만 추출.
CountVectorizer를 활용하여 키워드 후보군 생성.

3. BERT 임베딩 적용
사용자 정의 BERT 모델 및 토크나이저를 로드하여 텍스트를 벡터(임베딩)로 변환.
문서 및 키워드 후보군을 BERT 임베딩 벡터로 변환 (Mean Pooling 적용).

4. 코사인 유사도 계산
문서 임베딩과 키워드 후보군 임베딩 간 코사인 유사도 계산.
키워드 후보군 간의 코사인 유사도도 함께 계산하여 MMR 알고리즘 적용.

5. MMR (Maximal Marginal Relevance) 알고리즘 적용
문서와 가장 유사한 대표 키워드 1개 선정.
이후, 선정된 키워드와 유사하지 않으면서 문서와 연관성이 높은 키워드를 반복 선정.
다양성 조절 파라미터 (diversity=0.2) 적용하여 특정 키워드에 집중되지 않도록 제어.
