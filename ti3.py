import pandas as pd
import torch
from torch_geometric.data import DataLoader, HeteroData
from transformers import BertTokenizer, BertModel
from torch_geometric.nn import HGTConv
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'


# ---- 数据加载 ----
df = pd.read_csv('train.csv')  # 包含 context, label, category
contexts = df['content'].tolist()
labels = torch.tensor(df['label'].astype(int).tolist())
categories = df['category'].tolist()

# ---- BERT 嵌入 ----
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert = BertModel.from_pretrained('bert-base-chinese')
def get_bert_emb(texts, batch=16):
    embs = []
    for i in range(0, len(texts), batch):
        inputs = tokenizer(texts[i:i+batch], padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            embs.append(bert(**inputs).pooler_output)
    return torch.cat(embs)

text_emb = get_bert_emb(contexts)
cat_emb = get_bert_emb(categories)

# ---- 构造异构图 ----
data = HeteroData()
data['weibo'].x = text_emb
data['weibo'].y = labels
data['category'].x = cat_emb

# 边：微博 -> 其对应类别
unique_cats = list(set(categories))
cat2idx = {c:i for i,c in enumerate(unique_cats)}
ei = torch.tensor([[i for i,_ in enumerate(categories)],
                   [cat2idx[c] for c in categories]], dtype=torch.long)
data['weibo','belongs_to','category'].edge_index = ei

loader = DataLoader([data], batch_size=1, shuffle=True)

# ---- 模型 ----
class HGTModel(torch.nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.conv1 = HGTConv((-1, -1), 128, metadata=metadata)
        self.conv2 = HGTConv(128, 128, metadata=metadata)
        self.lin = torch.nn.Linear(128, 2)
    def forward(self, d):
        x = self.conv1(d.x_dict, d.edge_index_dict)
        x = self.conv2(x, d.edge_index_dict)
        return self.lin(x['weibo'])

model = HGTModel(metadata=data.metadata())
opt = torch.optim.Adam(model.parameters(), lr=2e-5)

# ---- 训练循环 & 记录指标 ----
epochs = 10
train_losses, train_acc = [], []

for epoch in range(1, epochs + 1):
    model.train()
    loss_sum, correct, total = 0, 0, 0
    for batch in tqdm(loader, desc=f"Epoch {epoch}", unit="batch"):
        opt.zero_grad()
        out = model(batch)
        loss = F.cross_entropy(out, batch['weibo'].y)
        loss.backward()
        opt.step()

        loss_sum += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == batch['weibo'].y).sum().item()
        total += batch['weibo'].y.size(0)

    avg_loss = loss_sum / len(loader)
    acc = correct / total
    train_losses.append(avg_loss)
    train_acc.append(acc)
    print(f" → Epoch {epoch}: Loss={avg_loss:.4f}, Acc={acc*100:.2f}%")

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(range(1, epochs+1), train_losses, marker='o')
plt.title('Training Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')

plt.subplot(1,2,2)
plt.plot(range(1, epochs+1), train_acc, marker='o', color='green')
plt.title('Training Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()
