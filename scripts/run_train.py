import os
import sys
import torch
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW

# src を import path に追加
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
DATA_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "data", "processed", "train_clean.parquet"))

print("SRC_PATH:", SRC_PATH)  # デバッグ用
print("DATA_PATH:", DATA_PATH)  # デバッグ用

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

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["judgement"])):
        print(f"\n===== Fold {fold} =====")

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_ds = TextDataset(train_df, tokenizer)
        val_ds = TextDataset(val_df, tokenizer)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

        model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
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

if __name__ == "__main__":
    main()