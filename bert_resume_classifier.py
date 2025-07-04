import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Load data
df = pd.read_csv('Resume.csv')
df = df[['Category', 'Resume']].dropna().reset_index(drop=True)

# Label encode the categories
le = LabelEncoder()
df['label'] = le.fit_transform(df['Category'])

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Dataset class
class ResumeDataset(Dataset):
    def __init__(self, texts, labels, max_len=256):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = tokenizer(self.texts[idx], padding='max_length', truncation=True, max_length=self.max_len, return_tensors="pt")
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx])
        }

# Prepare data
X_train, X_val, y_train, y_val = train_test_split(df['Resume'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42)
train_dataset = ResumeDataset(X_train, y_train)
val_dataset = ResumeDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Model definition
class BERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        return self.linear(self.dropout(pooled_output))

# Train function
def train(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Evaluation function
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
    accuracy = correct / len(loader.dataset)
    return total_loss / len(loader), accuracy

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTClassifier(num_classes=len(le.classes_)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(3):
    print(f"Epoch {epoch + 1}")
    train_loss = train(model, train_loader, optimizer, loss_fn, device)
    val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# Save model
torch.save(model.state_dict(), 'bert_resume_classifier.pt')
print("✅ Model trained and saved as 'bert_resume_classifier.pt'")
