#####################################################################################
"""
    Config 클래스.
    학습에 필요한 하이퍼파라미터.
    
    model_name (str): load할 kpf-bert model과 tokenizer 이름 (model과 tokenizer가 들어있는 폴더명)
    train_data (str): 학습할 데이터의 설명
    test_data (str): 테스트할 데이터의 설명
    epoch (int): 학습 횟수
    max_seq_len (int): kpf-bert-ner 에서 인식할 최대 토큰 수 (문장 길이와 연관, 최대 512)
    batch_size (int): 학습시 한번에 처리하는 데이터 수
    learning_rate (float): 학습 관련 파라미터, 학습률
    adam_epsilon (float): 학습 관련 파라미터, 최적화 변수
    max_grad_norm (float): 학습 관련 파라미터, 학습 안정화 clipping
    seed (int): 랜덤값 고정 변수, 학습 데이터셋 관련하여 고정된 랜덤값 출력
    intermediate_hidden_size (int): 모델의 hidden layer
    stride (int) : 겹칠 문장의 수
"""

from dataclasses import dataclass

#class Config():
#    model_name: str = "" # 모델 경로
#    train_data: str = "21_150tags_EntityLinking"
#    test_data: str = "21_150tags_EntityLinking"
#    epoch: int = 1
#    max_seq_len: int = 10
#    batch_size: int = 12
#    learning_rate: float = 5e-3
#    adam_epsilon: float = 1e-8
#    device: str = "cuda"
#    max_grad_norm: float = 1.0
#    seed: int = 2023
#    intermediate_hidden_size: int = 768
#    stride = 5

class Config():
    # 모델 설정
    model_name: str = "/home/lhch9550/공모전/KPF-BERT-CLS/cls_model"
    
    # 데이터 설정
    train_data: str = "21_150tags_EntityLinking"
    test_data: str = "21_150tags_EntityLinking"

    # 학습 파라미터
    epoch: int = 2  # 멀티라벨 학습에서는 더 많은 epoch가 필요할 수도 있음
    max_seq_len: int = 256  # 멀티라벨 문제는 문장이 더 길어질 가능성이 높음
    batch_size: int = 16  # 배치 크기 증가 가능
    learning_rate: float = 3e-5  # 일반적으로 3e-5 ~ 5e-5 사용
    adam_epsilon: float = 1e-8
    device: str = "cuda"
    max_grad_norm: float = 1.0
    seed: int = 2023
    intermediate_hidden_size: int = 768
    stride: int = 5

    # 멀티라벨 설정
    num_labels: int = 58  # 사용할 태그(라벨)의 개수
    loss_function: str = "BCEWithLogitsLoss"  # 멀티라벨 손실 함수
    activation_function: str = "sigmoid"  # 마지막 활성화 함수
