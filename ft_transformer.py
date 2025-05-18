# %%
import numpy as np
from tqdm import tqdm

import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchkeras.tabular import TabularPreprocessor, TabularDataset
from torchkeras.tabular.models import FTTransformerConfig, FTTransformerModel

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

np.random.seed(42)
train_df = pd.read_csv("data/train_df.csv")

train_df.head()

# ----- 數值型與類別型特徵 -----
numerical_features = ['age']
categorical_features = ['gender', 'primary_diagnosis', 'discharge_to', 'num_procedures', 'days_in_hospital', 'comorbidity_score']
target = 'readmitted'

# %%
df = train_df.copy()

# %% [markdown]
# # 前處理

# %%
# 數值標準化
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# 將類別特徵轉為 category 類型
for col in categorical_features:
    df[col] = df[col].astype('category')

df[target] = df[target].astype('category')

# 轉成 numpy
X_num = df[numerical_features].values.astype(np.float32)
X_cat = df[categorical_features].values
y = df[target].values.astype(np.int64)

# 分割訓練測試資料
X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = train_test_split(
    X_num, X_cat, y, test_size=0.2, random_state=42
)

# %%
# 1. 將數據轉為 DataFrame 格式，因為 TabularPreprocessor 通常需要 DataFrame
numerical_columns = [f'num_{i}' for i in range(len(numerical_features))]
categorical_columns = [f'cat_{i}' for i in range(len(categorical_features))]

train_df_train_csv = pd.DataFrame({
    **{f'num_{i}': X_num_train[:, i] for i in range(X_num_train.shape[1])},
    **{f'cat_{i}': X_cat_train[:, i] for i in range(X_cat_train.shape[1])},
    'target': y_train
})

test_df_train_csv = pd.DataFrame({
    **{f'num_{i}': X_num_test[:, i] for i in range(X_num_test.shape[1])},
    **{f'cat_{i}': X_cat_test[:, i] for i in range(X_cat_test.shape[1])},
    'target': y_test
})

# %%
train_df_train_csv.head()

# %%
train_target_df = train_df_train_csv['target']
train_df_train_csv = train_df_train_csv.drop(columns=['target'])
test_target_df = test_df_train_csv['target']
test_df_train_csv = test_df_train_csv.drop(columns=['target'])

# 使用 TabularPreprocessor 進行預處理
preprocessor = TabularPreprocessor(
    cat_features=categorical_columns,  # 類別特徵
    numeric_features=numerical_columns,    # 數值特徵
    normalization='standard',           # 數值特徵標準化（可選：'minmax' 或 None）
    onehot_max_cat_num=1,
)

preprocessor.fit(train_df_train_csv)

# %%
train_df_train_csv

# %%
train_df_train_csv_processed = preprocessor.transform(train_df_train_csv)
test_df_train_csv_processed = preprocessor.transform(test_df_train_csv)

# %%
train_df_train_csv_processed = train_df_train_csv_processed.loc[:, ~train_df_train_csv_processed.columns.duplicated(keep='last')]
test_df_train_csv_processed = test_df_train_csv_processed.loc[:, ~test_df_train_csv_processed.columns.duplicated(keep='last')]

numerical_features_processed = preprocessor.get_numeric_features()
embedding_features_processed = preprocessor.get_embedding_features()


# %%
train_df_train_csv_processed['target'] = train_target_df
test_df_train_csv_processed['target'] = test_target_df

# === 3. 建立 TabularDataset ===
ds_train = TabularDataset(
    data=train_df_train_csv_processed,
    task='classification',
    target=['target'],
    continuous_cols=numerical_features_processed,
    categorical_cols=embedding_features_processed
)

ds_test = TabularDataset(
    data=test_df_train_csv_processed,
    task='classification',
    target=['target'],
    continuous_cols=numerical_features_processed,
    categorical_cols=embedding_features_processed
)

dl_train = DataLoader(ds_train, batch_size=128, shuffle=False, num_workers=0, pin_memory=False)

dl_test = DataLoader(ds_test, batch_size=128, shuffle=False, num_workers=0, pin_memory=False)

# %%
print("categorical features (cols):", embedding_features_processed)
print("categorical features (array):", train_df_train_csv_processed[embedding_features_processed].shape[1])

print("numerical features (cols):", numerical_features_processed)
print("numerical features (array):", train_df_train_csv_processed[numerical_features_processed].shape[1])

# %% [markdown]
# # 建立模型

# %%
model_config = FTTransformerConfig(
    # ModelConfig 參數
    task="classification",  # 二元分類
    num_attn_blocks=3,
)

config = model_config.merge_dataset_config(ds_train)

# %%
# 初始化模型
model = FTTransformerModel(config)

# %% [markdown]
# # 訓練模型

# %%
# 設置優化器和損失函數
device = 'cuda' if torch.cuda.is_available() else 'cpu'
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# %%
# 訓練模型
def train_model(model: FTTransformerModel, train_loader: DataLoader, criterion: torch.nn.CrossEntropyLoss, optimizer: torch.optim.Adam, num_epochs:int=20, device:str=device):
    model.to(device)
    model.train()

    progress_bar = tqdm(range(num_epochs), leave=False)
    for epoch in progress_bar:
        total_loss = 0
        for batch in train_loader:
            numerical = batch['continuous'].to(device)
            categorical = batch['categorical'].to(device)
            target = batch['target'].to(device).squeeze()
            optimizer.zero_grad()
            outputs = model({'continuous': numerical, 'categorical': categorical})
            logits = outputs['logits']
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs} | Avg. Loss: {total_loss/len(train_loader):.4f}")
    
    # 儲存模型（建議用 .pt 或 .pth）
    torch.save(model.state_dict(), 'ft_transformer_model.pth')


# 評估模型
def evaluate_model(model, test_loader, device=device):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in test_loader:
            numerical = batch['continuous'].to(device)
            categorical = batch['categorical'].to(device)
            target = batch['target'].to(device).squeeze()
            outputs = model({'continuous': numerical, 'categorical': categorical})
            _, preds = torch.max(outputs['logits'], dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(target.cpu().numpy())
    accuracy = accuracy_score(true_labels, predictions)
    print(f'Test Accuracy: {accuracy:.4f}')
    print('\nClassification Report:')
    print(classification_report(true_labels, predictions))

# 執行訓練和評估
train_model(model, dl_train, criterion, optimizer, num_epochs=100)
evaluate_model(model, dl_test)

# %% [markdown]
# # 測試集

# %%
test_df_test_csv = pd.read_csv("data/test_df.csv")

# 將類別特徵轉為 category 類型
for col in categorical_features:
    test_df_test_csv[col] = test_df_test_csv[col].astype('category')

# %%
X_num_test_csv = test_df_test_csv[numerical_features].values.astype(np.float32)
X_cat_test_csv = test_df_test_csv[categorical_features].values

# %%
test_df_test_csv = pd.DataFrame({
    **{f'num_{i}': X_num_test_csv[:, i] for i in range(X_num_test_csv.shape[1])},
    **{f'cat_{i}': X_cat_test_csv[:, i] for i in range(X_cat_test_csv.shape[1])},
})

# %%
len(test_df_test_csv)

# %%
test_df_test_csv.head()

# %%
test_df_test_csv_processed = preprocessor.transform(test_df_test_csv)
test_df_test_csv_processed = test_df_test_csv_processed.loc[:, ~test_df_test_csv_processed.columns.duplicated(keep='last')]

ds_test_test_csv = TabularDataset(
    data=test_df_test_csv_processed,
    task='classification',
    continuous_cols=numerical_features_processed,
    categorical_cols=embedding_features_processed
)

dl_test_test_csv = DataLoader(ds_test_test_csv, batch_size=128, shuffle=False, num_workers=0, pin_memory=False)

# %%
print("categorical features (cols):", embedding_features_processed)
print("categorical features (array):", test_df_test_csv_processed[embedding_features_processed].shape[1])

print("numerical features (cols):", numerical_features_processed)
print("numerical features (array):", test_df_test_csv_processed[numerical_features_processed].shape[1])

# %%
# 12. 設置優化器和損失函數
device = 'cuda' if torch.cuda.is_available() else 'cpu'
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# 初始化模型
model = FTTransformerModel(config=config)
model.load_state_dict(torch.load("ft_transformer_model.pth"))
model.to(device)

predictions = []
true_labels = []
# evaluation mode
model.eval()
with torch.no_grad():
    for batch in dl_test_test_csv:
        numerical = batch['continuous'].to(device)
        categorical = batch['categorical'].to(device)
        outputs = model({'continuous': numerical, 'categorical': categorical})
        _, preds = torch.max(outputs['logits'], dim=1)
        predictions.extend(preds.cpu().numpy())

submission_df = pd.DataFrame({
    "Patient_ID": range(1, len(predictions)+1),
    "readmitted": predictions
})

submission_df.to_csv(path_or_buf='data/submission_df.csv', index=False)



