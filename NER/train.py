import config
import torch
import numpy as np
from transformers import BertConfig, AutoTokenizer, BertForTokenClassification, AdamW, logging
logging.set_verbosity_error()
from seqeval.metrics import f1_score, classification_report
from torch.utils.data import DataLoader
import json, sys
from tqdm import tqdm, trange
import Dataset
import label as label_mapping
from collections import defaultdict

# 누락된 레이블을 처리하기 위해 label_mapping.label2id를 defaultdict로 변경하여,
# lookup 시 존재하지 않는 키가 있으면 기본값으로 "O" 레이블의 id를 반환하도록 함.
label_mapping.label2id = defaultdict(lambda: label_mapping.label2id["O"], label_mapping.label2id)

############################################################################################################
"""
  train_epoch(epoch, model, dataloader, optimizer, con) : 학습 함수
  - 반복문을 돌면서 데이터를 학습.
  - 데이터셋에서 데이터들을 불러와 input 형태에 맞게 가공 후 원본 데이터와 비교하며 학습.
"""
############################################################################################################

def train_epoch(epoch, model, dataloader, optimizer, con):

    model.train()
    total_loss = 0.0
    
    # 학습 진행을 보여주는 tqdm 사용
    tepoch = tqdm(dataloader, unit="batch", position=1, leave=True)
    for batch in tepoch:
        tepoch.set_description(f"Train")
        model.zero_grad()
        
        # input data
        input_ids = batch[0].to(con.device)
        token_type_ids = batch[1].to(con.device)
        attention_mask = batch[2].to(con.device)
        labels = batch[3].to(con.device)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }
        
        # output
        outputs = model(**inputs)

        # loss (출력의 첫번째 값)
        loss = outputs[0]
        loss.backward()
    
        # gradient clipping 및 optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), con.max_grad_norm)
        optimizer.step()
        total_loss += loss.item()

        tepoch.set_postfix(loss=loss.mean().item())
    tepoch.set_postfix(loss=total_loss / len(dataloader))
    return total_loss / len(dataloader)

############################################################################################################
"""
    valid_epoch(epoch, dataloader, model, con) : 검증 함수.
    - 반복문을 돌면서 학습한 모델을 검증.
    - 학습 과정과 유사하게 진행됨.
"""
############################################################################################################

def valid_epoch(epoch, dataloader, model, con):
    total_loss = 0.0

    model.eval()
    all_token_predictions = []
    all_token_labels = []

    tepoch = tqdm(dataloader, unit="batch", leave=False)
    for batch in tepoch:
        tepoch.set_description(f"Valid")
        with torch.no_grad():
            input_ids = batch[0].to(con.device)
            token_type_ids = batch[1].to(con.device)
            attention_mask = batch[2].to(con.device)
            labels = batch[3].to(con.device)
            inputs = {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

            outputs = model(**inputs)

            loss, logits = outputs[:2]
            total_loss += loss.item()

            token_predictions = logits.argmax(dim=2)
            token_predictions = token_predictions.detach().cpu().numpy()

            for token_prediction, label in zip(token_predictions, labels):
                filtered = []
                filtered_label = []
                for i in range(len(token_prediction)):
                    if label[i].tolist() == -100:
                        continue
                    # missing label 처리: lookup 시 defaultdict가 "O"의 id를 반환함
                    filtered.append(label_mapping.id2label[token_prediction[i]])
                    filtered_label.append(label_mapping.id2label[label[i].tolist()])
                assert len(filtered) == len(filtered_label)
                all_token_predictions.append(filtered)
                all_token_labels.append(filtered_label)

        tepoch.set_postfix(loss=loss.mean().item())

    token_f1 = f1_score(all_token_labels, all_token_predictions, average="macro")
    return total_loss / len(dataloader), token_f1

############################################################################################################
"""
    test_epoch(dataloader, model, con) : 테스트 함수.
    학습한 모델의 성능을 평가.
"""
############################################################################################################
def test_epoch(dataloader, model, con):
    total_loss = 0.0

    model.eval()
    all_token_predictions = []
    all_token_labels = []

    tepoch = tqdm(dataloader, unit="batch")
    for batch in tepoch:
        tepoch.set_description(f"Test")
        with torch.no_grad():
            input_ids = batch[0].to(con.device)
            token_type_ids = batch[1].to(con.device)
            attention_mask = batch[2].to(con.device)
            labels = batch[3].to(con.device)

            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "labels": labels,
            }

            outputs = model(**inputs)

            loss, logits = outputs[:2]
            total_loss += loss.item()

            token_predictions = logits.argmax(dim=2)
            token_predictions = token_predictions.detach().cpu().numpy()

            for token_prediction, label in zip(token_predictions, labels):
                filtered = []
                filtered_label = []
                for i in range(len(token_prediction)):
                    if label[i].tolist() == -100:
                        continue
                    filtered.append(label_mapping.id2label[token_prediction[i]])
                    filtered_label.append(label_mapping.id2label[label[i].tolist()])
                assert len(filtered) == len(filtered_label)
                all_token_predictions.append(filtered)
                all_token_labels.append(filtered_label)

            tepoch.set_postfix(loss=loss.mean().item())
    
    # 평가 보고서 작성
    token_result = classification_report(all_token_labels, all_token_predictions)
    token_f1 = f1_score(all_token_labels, all_token_predictions, average="macro")

    print(token_result)

    tepoch.set_postfix(loss=total_loss / len(dataloader), token_f1=token_f1)
    return total_loss / len(dataloader), token_f1

############################################################################################################
"""
    set_optimizer(model, con) : 옵티마이저 설정 (AdamW 사용)
"""
############################################################################################################
def set_optimizer(model, con):
    optimizer_grouped_parameters = [
        {'params': model.bert.parameters(), 'lr': con.learning_rate / 100},
        {'params': model.classifier.parameters(), 'lr': con.learning_rate}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=con.learning_rate, eps=con.adam_epsilon)
    return optimizer

############################################################################################################
"""
    load_dataset(data, tokenizer, con) : 학습, 검증, 테스트 데이터셋 생성.
"""
############################################################################################################
def load_dataset(data, tokenizer, con):
    # 데이터셋 생성 (Dataset.load_data 내부 구현에 따라 처리)
    dataset = Dataset.load_data(data, tokenizer)
    
    # train dataset (70%)
    index = int(len(dataset) * 0.7)
    train_dataset = Dataset.NerDataset(
        tokenizer,
        dataset[:index],
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=Dataset.collate_fn
    )
    
    # valid dataset (20%)
    index2 = int(len(dataset) * 0.9)
    valid_dataset = Dataset.NerDataset(
        tokenizer,
        dataset[index:index2],
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=Dataset.collate_fn
    )
    
    # test dataset (10%)
    test_dataset = Dataset.NerDataset(
        tokenizer,
        dataset[index2:],
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=Dataset.collate_fn
    )
    
    return train_dataloader, valid_dataloader, test_dataloader

############################################################################################################
"""
    train(model, train_dataloader, valid_dataloader, test_dataloader, con) : 모델 학습 함수.
    설정한 epoch만큼 학습 및 검증 후 최종 테스트 수행.
"""
############################################################################################################
def train(model, train_dataloader, valid_dataloader, test_dataloader, con):
    model.to("cuda")

    best_f1 = 0.0
    best_model = None 

    tepoch = trange(con.epoch, position=0, leave=True)
    for epoch in tepoch:
        tepoch.set_description(f"Epoch {epoch}")

        # 학습
        train_loss = train_epoch(epoch, model, train_dataloader, optimizer, con)
        # 검증
        valid_loss, token_f1 = valid_epoch(epoch, valid_dataloader, model, con)

        if best_f1 < token_f1:
            best_f1 = token_f1
            best_model = model

        tepoch.set_postfix(valid_f1=token_f1)
    # 테스트
    test_loss, token_f1 = test_epoch(test_dataloader, model, con)
    
    return best_model

############################################################################################################
"""
    save_model(model, model_name) : 학습된 모델 저장 함수.
"""
############################################################################################################
def save_model(model, model_name):
    model.save_pretrained(model_name)

if __name__ == "__main__":
   
    if len(sys.argv) < 5:
        print("학습할 데이터셋과 모델명을 입력해주세요.")
        sys.exit()
    if sys.argv[1] != "-s":
        print("train -s TRAIN_FILE -o MODEL_NAME")
        sys.exit()
    if sys.argv[3] != "-o":
        print("train -s TRAIN_FILE -o MODEL_NAME")
        sys.exit()
    if sys.argv[4] == "kpfbert":
        print("kpfbert 외의 다른 이름을 입력해주세요")
        sys.exit()
        
    print("Start train!")
    
    # 학습할 모델의 환경 설정
    _config = config.Config()

    max_length = _config.max_seq_len
    batch_size = _config.batch_size

    # 랜덤 seed 고정
    torch.manual_seed(_config.seed)
    np.random.seed(_config.seed)

    # config에 정의된 레이블 수에 맞춰 BertConfig 생성 (checkpoint의 classifier는 300개 출력)
    bertconfig = BertConfig.from_pretrained(_config.model_name, num_labels=len(label_mapping.label2id))
    bertconfig.update(_config.__dict__)
    
    model_path = "kpfbert"  # pretrained 모델 경로

    # 모델 및 tokenizer 로드
    model = BertForTokenClassification.from_pretrained(_config.model_name, config=bertconfig)
    tokenizer = AutoTokenizer.from_pretrained(_config.model_name)
    
    # 데이터셋 로드 (json 파일)
    data = json.load(open(sys.argv[2], 'rt', encoding='UTF8'))
    
    # DataLoader 생성
    train_dataloader, valid_dataloader, test_dataloader = load_dataset(data, tokenizer, _config)
    
    # 옵티마이저 설정
    optimizer = set_optimizer(model, _config)
    
    # 모델 학습
    out_model = train(model, train_dataloader, valid_dataloader, test_dataloader, _config)
    
    # 모델 저장
    save_model(out_model, sys.argv[4])
    
    print("Train Done")