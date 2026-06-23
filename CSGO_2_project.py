import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup, logging
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm


def main():
    # 1. 학습 시 경고 메세지 제거 및 디바이스 설정
    logging.set_verbosity_error()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"현재 사용 중인 디바이스: {device}")

    # 2. 스팀 데이터 로드 (파일명 및 컬럼명 수정)
    path = "cs2_steam_ready.csv"  # 전처리가 완료된 스팀 데이터 파일명

    # 스팀 리뷰는 줄바꿈(\n)이 많아 데이터가 깨지는 것을 막기 위해 lineterminator 추가
    df = pd.read_csv(path, encoding="utf-8", lineterminator='\n')

    text = list(df["text"].values)  # 전처리된 리뷰 텍스트 컬럼
    labels = df["label"].values  # 스팀 유저들의 추천(1)/비추천(0) 라벨

    print("\n=== 데이터 확인 ===")
    print(" 문장 샘플 : ", text[:3])
    print(" 라벨 샘플 : ", labels[:3])

    # 3. 텍스트 데이터의 토큰화 (패딩 길이 효율화)
    tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')

    # 스팀 리뷰 길이에 맞춰 max_length를 512에서 128로 줄여 학습 속도를 대폭 향상
    inputs = tokenizer(text, truncation=True, max_length=128, add_special_tokens=True, padding="max_length")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    num_to_print = 3
    print("\n=== 토큰화 샘플 ===")
    for j in range(num_to_print):
        print(f"\n{j + 1}번째 데이터")
        print("토큰: ", input_ids[j][:10], "... (이하 생략)")  # 출력 가독성을 위해 앞 10개만 출력
        print("어텐션 마스크: ", attention_mask[j][:10], "... (이하 생략)")

    # 4. 데이터 분리 (2026년 기준 시드 유지)
    tx, vx, ty, vy = train_test_split(input_ids, labels, test_size=0.2, random_state=2026)
    tm, vm, _, _ = train_test_split(attention_mask, labels, test_size=0.2, random_state=2026)

    # 5. torch 학습용 데이터 세팅 (배치 사이즈 최적화)
    # MobileBERT 특성에 맞춰 배치 사이즈를 16으로 상향 조정 (메모리 여유 확보)
    batch_size = 16

    train_inputs = torch.tensor(tx)
    train_labels = torch.tensor(ty)
    train_masks = torch.tensor(tm)
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    valid_inputs = torch.tensor(vx)
    valid_labels = torch.tensor(vy)
    valid_masks = torch.tensor(vm)
    valid_data = TensorDataset(valid_inputs, valid_masks, valid_labels)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)

    # 6. 사전학습 언어모델 설정 (정식 허깅페이스 주소로 수정)
    model = MobileBertForSequenceClassification.from_pretrained("google/mobilebert-uncased", num_labels=2)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epoch = 4
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_dataloader) * epoch)

    # 7. 학습 및 검증
    epoch_results = []

    for e in range(epoch):
        model.train()
        total_train_loss = 0.0

        process_bar = tqdm(train_dataloader, desc=f"Training epoch {e + 1}", leave=False)

        for batch in process_bar:
            batch = tuple(t.to(device) for t in batch)
            batch_ids, batch_masks, batch_labels = batch
            model.zero_grad()

            outputs = model(batch_ids, attention_mask=batch_masks, labels=batch_labels)
            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            process_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = total_train_loss / len(train_dataloader)

        # --- 학습 정확도 평가 ---
        model.eval()
        train_preds, train_true = [], []

        progress_bar_t = tqdm(train_dataloader, desc=f"Evaluation Train Epoch {e + 1}", leave=False)
        for batch in progress_bar_t:
            batch = tuple(t.to(device) for t in batch)
            batch_ids, batch_masks, batch_labels = batch
            with torch.no_grad():
                outputs = model(batch_ids, attention_mask=batch_masks)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            train_preds.extend(preds.cpu().numpy())
            train_true.extend(batch_labels.cpu().numpy())
        train_acc = np.sum(np.array(train_preds) == np.array(train_true)) / len(train_preds)

        # --- 검증 정확도 평가 ---
        valid_preds, valid_true = [], []

        progress_bar_v = tqdm(valid_dataloader, desc=f"Evaluation Epoch {e + 1}", leave=False)
        for batch in progress_bar_v:
            batch = tuple(t.to(device) for t in batch)
            batch_ids, batch_masks, batch_labels = batch
            with torch.no_grad():
                outputs = model(batch_ids, attention_mask=batch_masks)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            valid_preds.extend(preds.cpu().numpy())
            valid_true.extend(batch_labels.cpu().numpy())
        valid_acc = np.sum(np.array(valid_preds) == np.array(valid_true)) / len(valid_preds)

        epoch_results.append([avg_train_loss, train_acc, valid_acc])

    # 8. 결과 저장
    print("\n=== 학습 및 검증 결과 ===")
    for idx, (loss, tacc, vacc) in enumerate(epoch_results, start=1):
        print(f"Epoch {idx}: 학습오차 - {loss:.4f}, 학습정확도 - {tacc:.4f}, 검증정확도 - {vacc:.4f}")

    # 9. 모델 및 토크나이저 저장 (허깅페이스 디렉토리 포맷 형식으로 수정)
    print("\n=== 모델 저장 ===")
    save_dir = "best_mobilebert_cs2_steam"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"모델 및 토크나이저가 '{save_dir}' 폴더에 정상 저장되었습니다.")


if __name__ == "__main__":
    main()