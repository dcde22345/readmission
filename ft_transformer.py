
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

from imblearn.over_sampling import SMOTE

np.random.seed(42)

# ----- 數值型與類別型特徵 -----
NUMERICAL_FEATURES = ['age']
CATEGORICAL_FEATURES = ['gender', 'primary_diagnosis', 'discharge_to', 'num_procedures', 'days_in_hospital', 'comorbidity_score']
TARGET = 'readmitted'
NUMERICAL_COLUMNS = [f'num_{i}' for i in range(len(NUMERICAL_FEATURES))]
CATEGORICAL_COLUMNS = [f'cat_{i}' for i in range(len(CATEGORICAL_FEATURES))]

class CustomerFTTransformer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.train_df = pd.read_csv("data/train_df.csv")
        self.test_df_test_csv = pd.read_csv("data/test_df.csv")

        # 使用 TabularPreprocessor 進行預處理
        self.preprocessor = TabularPreprocessor(
            cat_features=CATEGORICAL_COLUMNS,  # 類別特徵
            numeric_features=NUMERICAL_COLUMNS,    # 數值特徵
            normalization='standard',           # 數值特徵標準化（可選：'minmax' 或 None）
            onehot_max_cat_num=1,
        )

        self.model_config = FTTransformerConfig(
            # ModelConfig 參數
            task="classification",  # 二元分類
            num_attn_blocks=3,
        )

        self.model = None
        self.config = None
        self.train_df_train_csv_processed = None
        self.test_df_train_csv_processed = None
        self.train_target_df = None
        self.test_target_df = None
        self.train_df_train_csv = None
        self.test_df_train_csv = None
        self.numerical_features_processed = None
        self.embedding_features_processed = None
        self.ds_train = None
        self.ds_test = None
        self.dl_train = None
        self.dl_test = None

    def format_dataframe(self):
        # 將numerical做standarize
        self.train_df[NUMERICAL_FEATURES] = self.scaler.fit_transform(self.train_df[NUMERICAL_FEATURES])
        # 將類別特徵轉為 category 類型
        for col in CATEGORICAL_FEATURES:
            self.train_df[col] = self.train_df[col].astype('category')

        self.train_df[TARGET] = self.train_df[TARGET].astype('category')


    def _split_train_test(self):
        # 轉成 numpy
        X_num = self.train_df[NUMERICAL_FEATURES].values.astype(np.float32)
        X_cat = self.train_df[CATEGORICAL_FEATURES].values
        y = self.train_df[TARGET].values.astype(np.int64)

        # 分割訓練測試資料
        return train_test_split(
            X_num, X_cat, y, test_size=0.2, random_state=42
        )


    def preprocess(self):
        X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = self._split_train_test()

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

        self.train_target_df = train_df_train_csv['target']
        self.test_target_df = test_df_train_csv['target']

        self.train_df_train_csv = train_df_train_csv.drop(columns=['target'])
        self.test_df_train_csv = test_df_train_csv.drop(columns=['target'])

        self.preprocessor.fit(train_df_train_csv)
        train_df_train_csv_processed = self.preprocessor.transform(train_df_train_csv)
        test_df_train_csv_processed = self.preprocessor.transform(test_df_train_csv)

        self.train_df_train_csv_processed = train_df_train_csv_processed.loc[:, ~train_df_train_csv_processed.columns.duplicated(keep='last')]
        self.test_df_train_csv_processed = test_df_train_csv_processed.loc[:, ~test_df_train_csv_processed.columns.duplicated(keep='last')]

    def set_feautres_processed(self):
        self.numerical_features_processed = self.preprocessor.get_numeric_features()
        self.embedding_features_processed = self.preprocessor.get_embedding_features()
        self.train_df_train_csv_processed['target'] = self.train_target_df
        self.test_df_train_csv_processed['target'] = self.test_target_df

    def set_tablar_dataset(self):
        # === 建立 TabularDataset ===
        self.ds_train = TabularDataset(
            data=self.train_df_train_csv_processed,
            task='classification',
            target=['target'],
            continuous_cols=self.numerical_features_processed,
            categorical_cols=self.embedding_features_processed
        )

        self.ds_test = TabularDataset(
            data=self.test_df_train_csv_processed,
            task='classification',
            target=['target'],
            continuous_cols=self.numerical_features_processed,
            categorical_cols=self.embedding_features_processed
        )

        self.dl_train = DataLoader(self.ds_train, batch_size=128, shuffle=False, num_workers=0, pin_memory=False)
        self.dl_test = DataLoader(self.ds_test, batch_size=128, shuffle=False, num_workers=0, pin_memory=False)

        print("categorical features (cols):", self.embedding_features_processed)
        print("categorical features (array):", self.train_df_train_csv_processed[self.embedding_features_processed].shape[1])
        print("numerical features (cols):", self.numerical_features_processed)
        print("numerical features (array):", self.train_df_train_csv_processed[self.numerical_features_processed].shape[1])

    def set_model_config(self):
        # 建立模型
        self.config = self.model_config.merge_dataset_config(self.ds_train)
        # 初始化模型
        self.model = FTTransformerModel(self.config)

    def train_model(self, num_epochs: int):
        # 訓練模型
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        self.model.to(device)
        self.model.train()

        progress_bar = tqdm(range(num_epochs), leave=False)
        for epoch in progress_bar:
            total_loss = 0
            for batch in self.dl_train:
                numerical = batch['continuous'].to(device)
                categorical = batch['categorical'].to(device)
                target = batch['target'].to(device).squeeze()
                optimizer.zero_grad()
                outputs = self.model({'continuous': numerical, 'categorical': categorical})
                logits = outputs['logits']
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs} | Avg. Loss: {total_loss/len(self.dl_train):.4f}")
        
        # 儲存模型（建議用 .pt 或 .pth）
        torch.save(self.model.state_dict(), 'ft_transformer_model.pth')


    # 評估模型
    def evaluate_model(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = FTTransformerModel(config=self.config)
        self.model.load_state_dict(torch.load("ft_transformer_model.pth"))
        self.model.to(device=device)        
        self.model.eval()
        
        predictions = []
        true_labels = []
        with torch.no_grad():
            for batch in self.dl_test:
                numerical = batch['continuous'].to(device)
                categorical = batch['categorical'].to(device)
                target = batch['target'].to(device).squeeze()
                outputs = self.model({'continuous': numerical, 'categorical': categorical})
                _, preds = torch.max(outputs['logits'], dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(target.cpu().numpy())
        accuracy = accuracy_score(true_labels, predictions)
        print(f'Test Accuracy: {accuracy:.4f}')
        print('\nClassification Report:')
        print(classification_report(true_labels, predictions))

    def test_csv_model_eval(self):
        # 將類別特徵轉為 category 類型
        for col in CATEGORICAL_FEATURES:
            self.test_df_test_csv[col] = self.test_df_test_csv[col].astype('category')

        X_num_test_csv = self.test_df_test_csv[NUMERICAL_FEATURES].values.astype(np.float32)
        X_cat_test_csv = self.test_df_test_csv[CATEGORICAL_FEATURES].values

        self.test_df_test_csv = pd.DataFrame({
            **{f'num_{i}': X_num_test_csv[:, i] for i in range(X_num_test_csv.shape[1])},
            **{f'cat_{i}': X_cat_test_csv[:, i] for i in range(X_cat_test_csv.shape[1])},
        })

        test_df_test_csv_processed = self.preprocessor.transform(self.test_df_test_csv)
        test_df_test_csv_processed = test_df_test_csv_processed.loc[:, ~test_df_test_csv_processed.columns.duplicated(keep='last')]

        ds_test_test_csv = TabularDataset(
            data=test_df_test_csv_processed,
            task='classification',
            continuous_cols=self.numerical_features_processed,
            categorical_cols=self.embedding_features_processed
        )

        dl_test_test_csv = DataLoader(ds_test_test_csv, batch_size=128, shuffle=False, num_workers=0, pin_memory=False)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 初始化模型
        self.model = FTTransformerModel(config=self.config)
        self.model.load_state_dict(torch.load("ft_transformer_model.pth"))
        self.model.to(device)

        predictions = []

        # evaluation mode
        self.model.eval()
        with torch.no_grad():
            for batch in dl_test_test_csv:
                numerical = batch['continuous'].to(device)
                categorical = batch['categorical'].to(device)
                outputs = self.model({'continuous': numerical, 'categorical': categorical})
                _, preds = torch.max(outputs['logits'], dim=1)
                predictions.extend(preds.cpu().numpy())

        submission_df = pd.DataFrame({
            "Patient_ID": range(1, len(predictions)+1),
            "readmitted": predictions
        })

        submission_df.to_csv(path_or_buf='data/submission_df.csv', index=False)



