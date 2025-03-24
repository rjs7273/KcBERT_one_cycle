# T4 GPU 런타임

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AdamW
from tqdm import tqdm  # tqdm 임포트
import numpy as np
import os
import re

# ------------------------
# 1. 데이터셋 정의
# ------------------------
class SentimentDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

# ------------------------
# 2. 모델 정의
# ------------------------
class KcBERTSentiment(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("beomi/KcELECTRA-base")
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.bert.config.hidden_size, 3)  # [fear, neutral, greed]
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = outputs[:, 0]  # CLS 토큰
        logits = self.linear(self.dropout(pooled))
        probs = self.softmax(logits)
        return probs

# ------------------------
# 3. Threshold 스케줄
# ------------------------
def get_threshold(epoch):
    if epoch < 2:
        return 1.0
    elif epoch < 4:
        return 0.9
    elif epoch < 6:
        return 0.7
    elif epoch < 8:
        return 0.6
    else:
        return 0.5

# ------------------------
# 4. 학습 루프
# ------------------------
def train(model, dataloader, optimizer, device):
    model.train()
    loss_fn = nn.MSELoss()
    total_loss = 0
    # tqdm으로 진행률 바 추가
    for batch in tqdm(dataloader, desc="Training", ncols=100, leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# ------------------------
# 5. 가장 최근 에포크 로드
# ------------------------

model_dir = "/content/drive/MyDrive/KcBERT/model"
pattern = re.compile(r"kcbert_epoch(\d+)\.pt")

def find_latest_model(model_dir):
    latest_epoch = -1
    latest_path = None
    for fname in os.listdir(model_dir):
        match = pattern.match(fname)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_path = os.path.join(model_dir, fname)
    return latest_path, latest_epoch

# ------------------------
# 6. 전체 프로세스
# ------------------------
def main():
    # 데이터 로드
    labeled_df = pd.read_csv("/content/drive/MyDrive/KcBERT/data/삼성전자_sample_with_label_1_to_10.csv")
    full_df = pd.read_csv("/content/drive/MyDrive/KcBERT/data/삼성전자_preprocess.csv")

    labeled_df = labeled_df.dropna(subset=["content"])
    full_df = full_df.dropna(subset=["content"])

    tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 생성 및 로딩
    model = KcBERTSentiment()
    latest_model_path, latest_epoch = find_latest_model(model_dir)

    if latest_model_path:
        model.load_state_dict(torch.load(latest_model_path, map_location="cpu"))
        model.eval()
        print(f"✅ 가장 최근 모델 로드 완료: {latest_model_path} (epoch {latest_epoch})")
    else:
        print("🚀 저장된 모델이 없어 새로 시작합니다.")

    model = model.to(device)

    # 초기 라벨 데이터셋
    labeled_texts = labeled_df["content"].tolist()
    labeled_labels = labeled_df[["fear_score", "neutral_score", "greed_score"]].values.tolist()
    labeled_dataset = SentimentDataset(labeled_texts, labeled_labels, tokenizer)
    labeled_loader = DataLoader(labeled_dataset, batch_size=32, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    # 에포크 범위 수정: 최신 에포크부터 학습 시작
    for epoch in range(latest_epoch, 10):  # 최신 에포크부터 시작
        print(f"\n🌀 Epoch {epoch+1}")
        loss = train(model, labeled_loader, optimizer, device)
        print(f"Train loss: {loss:.4f}")

        # 모델 저장
        save_path = f"/content/drive/MyDrive/KcBERT/model/kcbert_epoch{epoch+1}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"📦 모델 저장 완료: {save_path}")

        # ---------- Pseudo-labeling ----------
        threshold = get_threshold(epoch)
        print(f"Using threshold: {threshold}")
        model.eval()

        full_texts = full_df["content"].tolist()
        full_dataset = SentimentDataset(full_texts, tokenizer=tokenizer)
        full_loader = DataLoader(full_dataset, batch_size=64)

        pseudo_texts, pseudo_labels = [], []

        with torch.no_grad():
            for batch in full_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids, attention_mask)  # (batch_size, 3)
                confidences, _ = torch.max(outputs, dim=1)
                mask = confidences > threshold
                for i in range(len(mask)):
                    if mask[i]:
                        pseudo_texts.append(tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True))
                        pseudo_labels.append(outputs[i].cpu().numpy())

        # 새롭게 확보된 pseudo-labeled 데이터셋 생성
        if pseudo_labels:
            print(f"  ✅ Accepted pseudo-labels: {len(pseudo_labels)}")
            labeled_texts += pseudo_texts
            labeled_labels += pseudo_labels
            labeled_dataset = SentimentDataset(labeled_texts, labeled_labels, tokenizer)
            labeled_loader = DataLoader(labeled_dataset, batch_size=256, shuffle=True)
        else:
            print("  ⚠️ No pseudo-labels accepted this round.")

if __name__ == "__main__":
    main()
