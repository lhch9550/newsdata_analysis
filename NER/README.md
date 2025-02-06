## 1. 개체명 인식(NER) 모델 소개

- 한국언론진흥재단이 개발한 kpf-BERT를 기반으로, 신문기사에서 150개 클래스를 분류하는 NER 모델인 kpf-BERT-ner은 문장의 각 단어에 적절한 태그를 부여하는 다중 분류 작업을 수행하며, 모두의 말뭉치의 [개체명 사전 데이터](https://kli.korean.go.kr/corpus/main/requestMain.do?lang=ko)를 활용합니다.
- kpf-BERT 토크나이저로 문장을 토큰화하고, 특수 토큰들을 추가한 후 BIO 표기법을 적용해 개체의 시작, 내부, 비개체를 구분합니다.
- 모델과 토크나이저는 [빅카인즈 허깅페이스](https://huggingface.co/KPF/KPF-bert-ner) 통해 다운로드 가능합니다.

## 2. 데이터 전처리

- 모델 전처리 과정은 학습 데이터(json 파일)를 불러오고 해당 파일 중에 필요한 정보로 파싱한다. **단, 국립국어원 데이터셋의 경우, 현재는 csv 형태로만 제공하고 있어 별도로 json 형식으로 변환 필요**
- 해당 정보들은 토크나이저를 통해 문장 내에서의 토큰의 위치, 세그먼트, ID 등으로 다시 구분하며 토큰별로 BIO 표기법 형태에 맞춰 표기한다. 그 후 학습을 위해 tensor 형태로 변환
- Dataset.py에서 데이터 전처리 과정을 진행
  - NerDataset : ner dataset 클래스. (torch의 dataset 라이브러리 사용, 해당 문서 참고)
  - load_data : 말뭉치 데이터(dict)에 맞게 필요한 정보를 추출하는 함수. 정보를 추출하고 BIO 표기법으로 분류 후 모델의 input 형태로 변형함.
  - collate_fn : 학습에 사용할 수 있도록 torch 라이브러리를 사용하여 타입을 변형시키고 매칭시켜줌.

```sentence : 문장 (ex. "아디다스의 대표 운동화 '스탠스미스'가 연간 800만 켤레 팔리는 것과 비교하면 놀랄 만한 실적이다")
token_label : 토큰의 클래스 (ex. ['B-OGG_ECONOMY', 'I-OGG_ECONOMY', 'I-OGG_ECONOMY', 'O', 'O', 'O', 'O', 'B-AFW_OTHER_PRODUCTS', 'I-AFW_OTHER_PRODUCTS',
                                'I-AFW_OTHER_PRODUCTS', 'O', 'O', 'O', 'B-QT_COUNT', 'I-QT_COUNT', 'I-QT_COUNT', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'])
char_label : 단어의 클래스 <BIO표기법> (ex. ['B-OGG_ECONOMY', 'I-OGG_ECONOMY', 'I-OGG_ECONOMY', 'I-OGG_ECONOMY', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 
                                            'B-AFW_OTHER_PRODUCTS', 'I-AFW_OTHER_PRODUCTS', 'I-AFW_OTHER_PRODUCTS', 'I-AFW_OTHER_PRODUCTS', 'I-AFW_OTHER_PRODUCTS',
                                            'O', 'O', 'O', 'O', 'O', 'O', 'B-QT_COUNT', 'I-QT_COUNT', 'I-QT_COUNT', 'I-QT_COUNT', 'I-QT_COUNT', 'I-QT_COUNT', 'I-QT_COUNT',
                                            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'])
offset_mapping : 토큰의 위치정보 (ex. [(0, 0), (0, 2), (2, 3), (3, 4), (4, 5), (6, 8), (9, 12), (13, 14), (14, 16), (16, 17), (17, 19), (19, 20), (20, 21), (22, 24), (25, 28),
                                (28, 29), (30, 32), (33, 35), (35, 36), (37, 38), (38, 39), (40, 42), (42, 44), (45, 47), (48, 50), (51, 53), (53, 54), (54, 55), (55, 56), (0, 0)])
```

## 3. 모델 학습

```
python train.py -s TRAIN_FILE -o MODEL_NAME
```
- kpf-bert-ner : KPF-BERT-NER 모델의 저장 위치
- train.py : 학습 관련 코드.
- 실행에 필요한 파일 : label.py, config.py, Dataset.py, kpfbert 모델 폴더가 있어야 합니다. 
