"""
sample_with_label_1_to_10.csv, sampling_10000.csv 이용해
KcBERT 학습시키는 코드
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from tqdm import tqdm
import pandas as pd
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
        self.linear = nn.Linear(self.bert.config.hidden_size, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = outputs[:, 0]
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
# 4. 학습 루프 (TPU 최적화)
# ------------------------
def train(model, dataloader, optimizer, device):
    model.train()
    loss_fn = nn.MSELoss()
    total_loss = 0

    parallel_loader = pl.ParallelLoader(dataloader, [device])
    tpu_loader = parallel_loader.per_device_loader(device)

    for batch in tqdm(tpu_loader, desc="Training", ncols=100, leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        xm.optimizer_step(optimizer)
        xm.mark_step()
        total_loss += loss.item()

    return total_loss / len(tpu_loader)

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
    device = xm.xla_device()
    tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")

    # 데이터 로드
    full_df = pd.read_csv("/content/drive/MyDrive/KcBERT/data/삼성전자_sampling_10000_data.csv")
    labeled_df = pd.read_csv("/content/drive/MyDrive/KcBERT/data/삼성전자_sample_with_label_1_to_10.csv")

    full_df = full_df.dropna(subset=["content"])
    labeled_df = labeled_df.dropna(subset=["content"])

    labeled_texts = labeled_df["content"].tolist()
    labeled_labels = labeled_df[["fear_score", "neutral_score", "greed_score"]].values.tolist()

    labeled_dataset = SentimentDataset(labeled_texts, labeled_labels, tokenizer)
    labeled_loader = DataLoader(labeled_dataset, batch_size=128, shuffle=True, num_workers=0)

    # 모델 생성 및 로딩
    model = KcBERTSentiment()
    latest_model_path, latest_epoch = find_latest_model(model_dir)

    if latest_model_path:
        model.load_state_dict(torch.load(latest_model_path, map_location="cpu"))
        model.eval()
        print(f"✅ 가장 최근 모델 로드 완료: {latest_model_path} (epoch {latest_epoch})")
    else:
        latest_epoch = 0
        print("🚀 저장된 모델이 없어 새로 시작합니다.")

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # 학습 반복
    for epoch in range(latest_epoch, 10):
        print(f"\n🌀 Epoch {epoch+1}")
        loss = train(model, labeled_loader, optimizer, device)
        print(f"Train loss: {loss:.4f}")

        save_path = f"{model_dir}/kcbert_epoch{epoch+1}.pt"
        xm.save(model.state_dict(), save_path)
        print(f"📦 모델 저장 완료: {save_path}")

        # ---------- Pseudo-labeling ----------
        threshold = get_threshold(epoch)
        print(f"Using threshold: {threshold}")
        model.eval()

        full_texts = full_df["content"].tolist()
        full_dataset = SentimentDataset(full_texts, tokenizer=tokenizer)
        full_loader = DataLoader(full_dataset, batch_size=64)

        pseudo_texts, pseudo_labels = [], []

        parallel_loader = pl.ParallelLoader(full_loader, [device])
        tpu_loader = parallel_loader.per_device_loader(device)

        with torch.no_grad():
            for batch in tqdm(tpu_loader, desc="Pseudo-labeling", ncols=100, leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids, attention_mask)
                confidences, _ = torch.max(outputs, dim=1)
                mask = confidences > threshold
                for i in range(len(mask)):
                    if mask[i]:
                        pseudo_texts.append(tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True))
                        pseudo_labels.append(outputs[i].cpu().numpy())

        if pseudo_labels:
            print(f"  ✅ Accepted pseudo-labels: {len(pseudo_labels)}")
            labeled_texts += pseudo_texts
            labeled_labels += pseudo_labels
            labeled_dataset = SentimentDataset(labeled_texts, labeled_labels, tokenizer)
            labeled_loader = DataLoader(labeled_dataset, batch_size=256, shuffle=True, num_workers=0)
        else:
            print("  ⚠️ No pseudo-labels accepted this round.")
