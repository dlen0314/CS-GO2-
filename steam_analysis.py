import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
from tqdm import tqdm


def main():
    # 1. 디바이스 및 저장된 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "best_mobilebert_cs2_steam"

    print("학습된 MobileBERT 모델을 불러오는 중...")
    tokenizer = MobileBertTokenizer.from_pretrained(save_dir)
    model = MobileBertForSequenceClassification.from_pretrained(save_dir)
    model.to(device)
    model.eval()

    # 2. 분석할 데이터 로드
    path = "cs2_steam_ready.csv"
    df = pd.read_csv(path, encoding="utf-8", lineterminator='\n')

    # 전체 데이터를 다 돌리면 시간이 걸리므로 무작위 500개 샘플 추출 분석
    df_sample = df.sample(n=min(500, len(df)), random_state=42).copy()

    # 3. 추론(Inference) 함수 정의
    def predict_sentiment(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
        return prediction

    print("스팀 리뷰 데이터 감성 분석 시작 (MobileBERT)...")
    tqdm.pandas(desc="Analyzing")
    df_sample['pred_sentiment'] = df_sample['text'].progress_apply(predict_sentiment)

    # 4. 결과 집계 및 그래프 시각화
    counts = df_sample['pred_sentiment'].value_counts()
    neg_count = counts.get(0, 0)
    pos_count = counts.get(1, 0)

    plt.figure(figsize=(7, 5))
    bars = plt.bar(['Negative (0)', 'Positive (1)'], [neg_count, pos_count], color=['#ff6b6b', '#4dadf7'])
    plt.title("CS2 Update Steam Review Sentiment Analysis (MobileBERT)", fontsize=14, pad=15)
    plt.ylabel("Number of Reviews", fontsize=12)

    # 막대 그래프 상단에 숫자 표기
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + (max(neg_count, pos_count) * 0.02),
                 f"{yval} reviews", ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 결과 이미지 저장 및 화면 출력
    output_img = "cs2_sentiment_result.png"
    plt.tight_layout()
    plt.savefig(output_img)
    print(f"\n=== 분석 완료 ===")
    print(f"부정적 반응: {neg_count}건 / 긍정적 반응: {pos_count}건")
    print(f"결과 그래프가 '{output_img}' 파일로 저장되었습니다.")
    plt.show()


if __name__ == "__main__":
    main()