import os
import sys
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# パス設定
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = "/content/medical_project/src"

# ★ TinyBERT Fold0 の保存先
MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"
MODEL_DIR_NAME = MODEL_NAME.split("/")[-1].lower()
MODEL_DIR = f"/content/drive/MyDrive/medical_project/models/{MODEL_DIR_NAME}/fold0"

# ★ 推論したいデータ（test）
TEST_PATH = "/content/drive/MyDrive/medical_project_data/processed/test_clean.parquet"

# ★ 出力ファイル（submit.csv）
OUTPUT_PATH = "/content/drive/MyDrive/medical_project/submit/submit.csv"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

print("MODEL_DIR:", MODEL_DIR)
print("TEST_PATH:", TEST_PATH)
print("OUTPUT_PATH:", OUTPUT_PATH)

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# 自作モジュール
from models.datasetClass import TextDataset
from models.predict import predict


def main():
    device = "cpu"
    print("Using device:", device)

    # データ読み込み
    df = pd.read_parquet(TEST_PATH)

    # モデル読み込み
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)

    # Dataset（max_len=64）
    ds = TextDataset(df, tokenizer, max_len=64)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)

    # 推論（確率）
    print("Running prediction...")
    prob = predict(model, loader, device)

    # ★ 0/1 に変換（threshold=0.5）
    pred = (prob >= 0.5).astype(int)

    # ★ submit.csv の形式に整形
    df_out = pd.DataFrame({
        "index": df["id"],     # id を index として使う
        "prediction": pred     # 0/1
    })

    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved submit.csv to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()