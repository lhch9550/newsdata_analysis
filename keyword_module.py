import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer
from kiwipiepy import Kiwi

# Kiwi 형태소 분석기 초기화
kiwi = Kiwi()

# 사용자 정의 BERT 모델 로드
model_name_or_path = "/home/lhch9550/공모전"  # 사용자의 BERT 모델 경로
custom_model = BertModel.from_pretrained(model_name_or_path, add_pooling_layer=False)
custom_tokenizer = BertTokenizer.from_pretrained(model_name_or_path)

def encode_text(text):
    """텍스트를 임베딩 벡터로 변환하는 함수 (2D 배열로 변환)"""
    inputs = custom_tokenizer(text, 
                              return_tensors="pt", 
                              padding=True, 
                              truncation=True,
                              max_length=512)
    outputs = custom_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # 평균 풀링
    return embeddings.detach().numpy().reshape(1, -1)  # 2D 배열로 변환 (1, hidden_size)

def keyword_ext(text):
    """주어진 텍스트에서 키워드를 추출하는 함수"""
    
    # 예외 처리: 입력이 비어 있으면 빈 리스트 반환
    if not text or not isinstance(text, str) or text.strip() == "":
        return []

    # Kiwi를 활용한 형태소 분석 (명사 추출)
    tokenized_doc = kiwi.tokenize(text)
    tokenized_nouns = ' '.join([word.form for word in tokenized_doc if word.tag in ['NNG', 'NNP']])

    # 예외 처리: 명사가 없으면 빈 리스트 반환
    if not tokenized_nouns.strip():
        return []

    # CountVectorizer를 실행하기 전에 단어가 없으면 빈 리스트 반환
    if len(tokenized_nouns.split()) == 0:
        return []

    # 키워드 후보군 생성 (1-gram 기준)
    try:
        count = CountVectorizer(ngram_range=(1, 1))
        count.fit([tokenized_nouns])
        candidates = count.get_feature_names_out()
    except ValueError:  # 빈 단어 목록이면 예외 발생
        return []

    # 예외 처리: 키워드 후보가 없으면 빈 리스트 반환
    if len(candidates) == 0:
        return []

    # 문서 및 키워드 후보군을 벡터로 변환 (2D 배열 유지)
    doc_embedding = encode_text(text)
    candidate_embeddings = np.vstack([encode_text(candidate) for candidate in candidates]) if len(candidates) > 0 else np.array([])

    # 예외 처리: 임베딩이 비어 있으면 빈 리스트 반환
    if candidate_embeddings.size == 0 or doc_embedding.size == 0:
        return []

    return mmr(doc_embedding, candidate_embeddings, candidates, top_n=20, diversity=0.2)

def mmr(doc_embedding, candidate_embeddings, words, top_n, diversity):
    """MMR 알고리즘을 활용하여 최적의 키워드 추출"""

    # 예외 처리: 빈 배열일 경우 즉시 반환
    if candidate_embeddings.shape[0] == 0:
        return []

    # 문서와 각 키워드의 유사도 계산
    word_doc_similarity = cosine_similarity(candidate_embeddings, doc_embedding)

    # 키워드 간 유사도 계산
    word_similarity = cosine_similarity(candidate_embeddings)

    # 예외 처리: 유사도 값이 비어 있으면 반환
    if word_doc_similarity.size == 0:
        return []

    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    # 나머지 키워드 선정 (top_n 개수만큼 반복)
    for _ in range(top_n - 1):
        if len(candidates_idx) == 0:
            break

        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        mmr = (1 - diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # 업데이트
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]

# 테스트 실행 코드
if __name__ == "__main__":
    text = """
    한덕수 국무총리는 30일 난방비 인상 등과 관련해 시장 상황에 맞게 가격을 조정하지 않는 정책은 '포퓰리즘'이라고 규정했다.

    한 총리는 이날 오전 정부서울청사에서 주재한 국무회의 모두발언에서 "한파와 가스비 등 공공요금 인상이 겹쳐 국민들이 느끼는 고통에 마음이 무겁다"고 말했다.
    
    이어 "그러나 국민들이 불편해한다고 해서 장기간 조정해야 할 가격을 시장에 맞서 조정하지 않고 억누르는 정책은, 추후 국민들께 더 큰 부담을 드리고 우리 경제에 악영향을 끼치는 포퓰리즘 정책에 다름 아니다"라고 지적했다.
    """
    
    print(keyword_ext(text))



