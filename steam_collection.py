import requests
import pandas as pd
import time


def get_steam_reviews(appid, total_wanted=1000):
    review_data = []
    cursor = '*'
    url = f"https://store.steampowered.com/appreviews/{appid}?json=1"

    while len(review_data) < total_wanted:
        params = {
            'filter': 'recent',
            'language': 'english',
            'review_type': 'all',
            'purchase_type': 'all',
            'cursor': cursor,
            'num_per_page': 100
        }
        response = requests.get(url, params=params).json()
        if response.get('success') != 1 or not response.get('reviews'):
            break
        for review in response['reviews']:
            review_data.append({
                'text': review['review'],
                'label': 1 if review['voted_up'] else 0
            })
        print(f"수집 중... {len(review_data)}개 확보")
        cursor = response['cursor']
        time.sleep(0.5)
    return pd.DataFrame(review_data)


# CS2 ID인 730으로 1000개 수집 후 저장
df = get_steam_reviews(appid=730, total_wanted=30000)
df.to_csv("cs2_raw_comments.csv", index=False, encoding="utf-8")
print("1단계 수집 완료.")