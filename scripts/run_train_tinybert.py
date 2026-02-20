import os
import sys
import torch
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW

# CPU 最適化
torch.set_num_threads(4)

# src を import path に追加
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
DATA_PATH = "/content/drive/MyDrive/medical_project_data/processed/train_clean.parquet"
SAVE_DIR = "/content/drive/MyDrive/medical_project/models"

# ★ モデル名（フォルダ名に変換する）
MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"
MODEL_DIR_NAME = MODEL_NAME.split("/")[-1].lower()  # tinybert_general_4l_312d

# ★ MODEL ごとに保存フォルダを分ける
SAVE_DIR = f"/content/drive/MyDrive/medical_project/models/{MODEL_DIR_NAME}"
os.makedirs(SAVE_DIR, exist_ok=True)

print("SRC_PATH:", SRC_PATH)  # デバッグ用
print("DATA_PATH:", DATA_PATH)  # デバッグ用
print("SAVE_DIR:", SAVE_DIR)

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)


# 自作モジュール
from models.datasetClass import TextDataset
from models.model_train_loop import train_one_epoch
from models.predict import predict
from models.find_best import find_best_threshold

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # データ読み込み
    df = pd.read_parquet(DATA_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["judgement"])):
        print(f"\n===== Fold {fold} =====")

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_ds = TextDataset(train_df, tokenizer, max_len=128)
        val_ds = TextDataset(val_df, tokenizer, max_len=128)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=1
        ).to(device)

        optimizer = AdamW(model.parameters(), lr=2e-5)

        # 1 epoch でまずは動作確認
        train_one_epoch(model, train_loader, optimizer, device)

        # 推論
        val_prob = predict(model, val_loader, device)

        # 閾値最適化
        t, f = find_best_threshold(val_df["judgement"].values, val_prob)
        print(f"Fold {fold} best threshold={t:.3f}, Fβ={f:.4f}")

        # 保存
        fold_dir = f"{SAVE_DIR}/fold{fold}"
        os.makedirs(fold_dir, exist_ok=True)

        model.save_pretrained(fold_dir)
        tokenizer.save_pretrained(fold_dir)

        print(f"Saved model to: {fold_dir}")


if __name__ == "__main__":
    main()