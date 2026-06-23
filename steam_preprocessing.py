import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def clean_text(text):
    if pd.isna(text): return None
    # 영문, 숫자, 기본 구두점만 남기기 (특수문자 및 노이즈 제거)
    text = re.sub(r"[^a-zA-Z0-9\s!?.\']", "", text)
    text = text.strip().lower()
    if len(text.split()) < 4: return None  # 단어 3개 이하 도배글 제거
    return text


def main():
    print("1. 원본 데이터 로드 및 전처리 시작...")
    df = pd.read_csv("cs2_raw_comments.csv", encoding="utf-8")

    # [평가기준 대응] 스팀 데이터는 원본 자체가 추천(1)/비추천(0)이므로 중립 데이터가 없음
    df['cleaned_text'] = df['text'].apply(clean_text)
    df = df.dropna(subset=['cleaned_text']).drop_duplicates(subset=['cleaned_text'])

    # 전처리 완료된 데이터를 '분석 대상 데이터'로 저장
    df.to_csv("cs2_steam_ready_full.csv", index=False, encoding="utf-8")
    print(f"전체 분석 대상 데이터 확보 완료: {len(df)}건")

    # [평가기준 대응] 학습 데이터 3,000건 (긍/부정 각 1,500건) 균형 추출
    print("\n2. 학습 데이터 3,000건 샘플링 및 EDA 시각화 진행...")

    # 라벨에 맞춰 1500건씩 무작위 추출 (random_state로 결과 고정)
    pos_df = df[df['label'] == 1].sample(n=1500, random_state=42)
    neg_df = df[df['label'] == 0].sample(n=1500, random_state=42)

    # 두 데이터를 합치고 순서를 무작위로 섞음
    train_df = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    # 최종 학습용 데이터 저장 (6번 학습 코드에서는 이 파일을 불러오게 됨)
    train_df.to_csv("cs2_steam_ready.csv", index=False, encoding="utf-8")
    print("학습용 데이터 'cs2_steam_ready.csv' (3,000건) 저장 완료.")

    # --- EDA 시각화 ---
    plt.figure(figsize=(12, 5))

    # 1. 라벨 분포 그래프 (클래스 불균형 해소 확인용)
    plt.subplot(1, 2, 1)
    sns.countplot(data=train_df, x='label', palette=['#ff6b6b', '#4dadf7'])
    plt.title("Label Distribution (0: Negative, 1: Positive)", fontsize=13)
    plt.xlabel("Sentiment")
    plt.ylabel("Count")

    # 2. 문장 길이(단어 수) 분포 그래프 (max_length 설정 근거용)
    plt.subplot(1, 2, 2)
    train_df['word_count'] = train_df['cleaned_text'].apply(lambda x: len(x.split()))
    sns.histplot(data=train_df, x='word_count', bins=30, kde=True, color='purple')
    plt.title("Review Word Count Distribution", fontsize=13)
    plt.xlabel("Number of Words")
    plt.xlim(0, 150)  # 대부분의 리뷰 길이에 맞춰 X축 시각적 제한

    plt.tight_layout()
    plt.savefig("eda_result.png")
    print("EDA 시각화 완료. 결과가 'eda_result.png'로 저장되었어.")
    plt.show()


if __name__ == "__main__":
    main()