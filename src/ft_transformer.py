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

from src.imbalance_data_processing import apply_smote, apply_under_sampling

from sklearn.preprocessing import LabelEncoder


np.random.seed(42)

# ----- 數值型與類別型特徵 -----
NUMERICAL_FEATURES = ['age', 'num_procedures', 'days_in_hospital', 'comorbidity_score']
CATEGORICAL_FEATURES = ['gender', 'primary_diagnosis', 'discharge_to']
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
        # 將類別特徵編碼為數值（SMOTENC 需要整數形式的類別特徵）
        self.label_encoders = {}
        
        for col in CATEGORICAL_FEATURES:
            le = LabelEncoder()
            self.train_df[col] = le.fit_transform(self.train_df[col])
            self.label_encoders[col] = le
            
            # 轉換為 category 類型
            self.train_df[col] = self.train_df[col].astype('category')
            
            print(f"Encoded {col}: {self.train_df[col].cat.categories.tolist()}")

        self.train_df[TARGET] = self.train_df[TARGET].astype('category')
        
        print("=== Feature Encoding Complete ===")
        print(f"Numerical features: {NUMERICAL_FEATURES}")
        print(f"Categorical features (encoded): {CATEGORICAL_FEATURES}")
        
        # 檢查編碼後的數據樣本
        print("\nSample of encoded data:")
        print(self.train_df[CATEGORICAL_FEATURES].head())

    def _split_train_test(self):
        # 轉成 numpy
        X_num = self.train_df[NUMERICAL_FEATURES].values.astype(np.float32)
        X_cat = self.train_df[CATEGORICAL_FEATURES].values
        y = self.train_df[TARGET].values.astype(np.int64)

        # 分割訓練測試資料
        return train_test_split(
            X_num, X_cat, y, test_size=0.2, random_state=42
        )


    def preprocess(self, use_smote=False, use_under_sampling=False, smote_method='smotenc', under_sampling_method='tomek'):
        X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = self._split_train_test()
        
        # 保存原始的 y_train 用於 class weight 計算
        self.original_y_train = y_train.copy()
        
        # 標準化數值特徵（這對 SMOTE 很重要）
        X_num_train = self.scaler.fit_transform(X_num_train)
        X_num_test = self.scaler.transform(X_num_test)

        # 檢查採樣方法衝突
        if use_smote and use_under_sampling:
            print(f"Applying Under Sampling using: {under_sampling_method}, then applying SMOTE using: {smote_method}")

        # Apply Under Sampling if requested
        if use_under_sampling:
            print(f"🔄 Applying under sampling with method: {under_sampling_method}")
            X_num_train_orig_size = len(X_num_train)
            X_num_train, X_cat_train, y_train = apply_under_sampling(X_num_train, X_cat_train, y_train, method=under_sampling_method)
            print(f"After under sampling - Training set size: {len(y_train)} (was {X_num_train_orig_size})")

        # Apply SMOTE if requested (and not using under sampling)
        if use_smote:
            print(f"🔄 Applying SMOTE with method: {smote_method}")
            X_num_train_orig_size = len(X_num_train)
            X_num_train, X_cat_train, y_train = apply_smote(X_num_train, X_cat_train, y_train, method=smote_method)
            print(f"After SMOTE - Training set size: {len(y_train)} (was {X_num_train_orig_size})")
            print(f"After SMOTE - Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
            print(f"After SMOTE - X_num_train shape: {X_num_train.shape}")
            print(f"After SMOTE - X_cat_train shape: {X_cat_train.shape}")

        train_df_train_csv = pd.DataFrame({
            **{f'num_{i}': X_num_train[:, i] for i in range(X_num_train.shape[1])},
            **{f'cat_{i}': X_cat_train[:, i] for i in range(X_cat_train.shape[1])},
            'target': y_train
        })

        print(f"train_df_train_csv shape after DataFrame creation: {train_df_train_csv.shape}")

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

        print(f"Final train_df_train_csv_processed shape: {self.train_df_train_csv_processed.shape}")
        print(f"Final test_df_train_csv_processed shape: {self.test_df_train_csv_processed.shape}")
        print(f"=== END PREPROCESS DEBUG ===")

    # 檢查類別分佈
    def check_class_distribution(self, df:pd.DataFrame, target_col:str):
        """
        Check and print the class distribution of the target variable.
        
        Args:
            df (pd.DataFrame): Input dataframe to check
            target_col (str): Name of target column to analyze
            
        Returns:
            tuple: (value_counts, imbalance_ratio)
        """
        print("=== Class Distribution Analysis ===")
        
        # Get class distribution
        class_counts = df[target_col].value_counts().sort_index()
        print(f"Class distribution:")
        for class_val, count in class_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  Class {class_val}: {count} samples ({percentage:.2f}%)")
        
        # Calculate imbalance ratio
        min_class = class_counts.min()
        max_class = class_counts.max()
        imbalance_ratio = max_class / min_class
        print(f"Imbalance ratio: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 2.0:
            print("⚠️  Dataset is imbalanced.")
        else:
            print("✅ Dataset is relatively balanced.")
        
        return class_counts, imbalance_ratio

    # 設定特徵處理
    def set_feautres_processed(self):
        self.numerical_features_processed = self.preprocessor.get_numeric_features()
        self.embedding_features_processed = self.preprocessor.get_embedding_features()
        
        # 將資料與目標變數合併後進行 shuffle
        self.train_df_train_csv_processed['target'] = self.train_target_df
        self.test_df_train_csv_processed['target'] = self.test_target_df
        
        # 對訓練資料進行 shuffle
        self.train_df_train_csv_processed = self.train_df_train_csv_processed.sample(
            frac=1, random_state=42
        ).reset_index(drop=True)
        
        # 對測試資料進行 shuffle 
        self.test_df_train_csv_processed = self.test_df_train_csv_processed.sample(
            frac=1, random_state=42
        ).reset_index(drop=True)

    # 設定 TabularDataset
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

    # 設定模型設定
    def set_model_config(self):
        # 建立模型
        self.config = self.model_config.merge_dataset_config(self.ds_train)
        # 初始化模型
        self.model = FTTransformerModel(self.config)

    # 訓練模型
    def train_model(self, num_epochs: int, model_name: str, use_class_weight=False, plot_train_metrics=True):
        """
        訓練 FT-Transformer 模型
        
        Args:
            num_epochs (int): 訓練輪數
            model_name (str): 模型保存名稱
            use_class_weight (bool): 是否使用 class weight 來處理類別不平衡
            plot_train_metrics (bool): 是否在訓練後繪製訓練集的ROC和混淆矩陣
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        # 計算並設定 class weights
        if use_class_weight:
            class_weight_tensor = self._compute_class_weights(device)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weight_tensor)
            print(f"Using class weights: {class_weight_tensor.cpu().numpy()}")
        else:
            criterion = torch.nn.CrossEntropyLoss()
            print("Using standard CrossEntropyLoss (no class weights)")

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
        torch.save(self.model.state_dict(), f'./models/{model_name}')
        print(f"Model saved as ./models/{model_name}")
        
        # 繪製訓練集評估指標
        if plot_train_metrics:
            print("\n=== Training Set Evaluation ===")
            self.evaluate_train_set()

    def evaluate_train_set(self):
        """
        評估訓練集並繪製ROC曲線和混淆矩陣
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model.eval()
        predictions = []
        probabilities = []
        true_labels = []
        
        with torch.no_grad():
            for batch in self.dl_train:
                numerical = batch['continuous'].to(device)
                categorical = batch['categorical'].to(device)
                target = batch['target'].to(device).squeeze()
                
                outputs = self.model({'continuous': numerical, 'categorical': categorical})
                logits = outputs['logits']
                
                # 獲取預測概率
                probs = torch.softmax(logits, dim=1)
                
                # 獲取預測類別
                _, preds = torch.max(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
                true_labels.extend(target.cpu().numpy())
        
        # 轉換為numpy arrays
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        true_labels = np.array(true_labels)
        
        # 計算準確率
        accuracy = accuracy_score(true_labels, predictions)
        print(f'Training Set Accuracy: {accuracy:.4f}')
        
        # 顯示分類報告
        print('\nTraining Set Classification Report:')
        print(classification_report(true_labels, predictions))
        
        # 計算混淆矩陣
        cm = confusion_matrix(true_labels, predictions)
        print('\nTraining Set Confusion Matrix:')
        print(cm)
        
        # 繪製混淆矩陣和ROC曲線
        self._plot_train_metrics(true_labels, predictions, probabilities, cm)
        
        return accuracy, cm

    def _plot_train_metrics(self, true_labels, predictions, probabilities, cm):
        """
        繪製訓練集的ROC曲線和混淆矩陣
        
        Args:
            true_labels: 真實標籤
            predictions: 預測標籤
            probabilities: 預測概率
            cm: 混淆矩陣
        """
        try:
            import seaborn as sns
            from sklearn.metrics import roc_curve, auc
            import matplotlib
            
            # 設定非互動式後端（適用於服務器環境）
            matplotlib.use('Agg')
            
            # 設定matplotlib中文字體
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 創建子圖
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # === 1. 繪製混淆矩陣 ===
            if len(np.unique(true_labels)) == 2:
                class_labels = ['Not Readmitted', 'Readmitted']
            else:
                class_labels = [f'Class {label}' for label in np.unique(true_labels)]
            
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                       xticklabels=class_labels, 
                       yticklabels=class_labels,
                       ax=axes[0])
            axes[0].set_title("Training Set - Confusion Matrix", fontsize=14, fontweight='bold')
            axes[0].set_xlabel("Predicted", fontsize=12)
            axes[0].set_ylabel("Actual", fontsize=12)
            
            # === 2. 繪製ROC曲線 ===
            if len(np.unique(true_labels)) == 2:  # 二元分類
                # 使用正類別（class 1）的概率
                y_prob_positive = probabilities[:, 1]
                
                # 計算ROC曲線
                fpr, tpr, thresholds = roc_curve(true_labels, y_prob_positive)
                roc_auc = auc(fpr, tpr)
                
                # 繪製ROC曲線
                axes[1].plot(fpr, tpr, color='darkorange', lw=3, 
                           label=f'ROC curve (AUC = {roc_auc:.4f})')
                axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                           label='Random Classifier', alpha=0.7)
                axes[1].set_xlim([0.0, 1.0])
                axes[1].set_ylim([0.0, 1.05])
                axes[1].set_xlabel('False Positive Rate', fontsize=12)
                axes[1].set_ylabel('True Positive Rate', fontsize=12)
                axes[1].set_title('Training Set - ROC Curve', fontsize=14, fontweight='bold')
                axes[1].legend(loc="lower right")
                axes[1].grid(True, alpha=0.3)
                
                print(f"Training Set ROC AUC: {roc_auc:.4f}")
            else:
                axes[1].text(0.5, 0.5, 'ROC curve only available\nfor binary classification', 
                           ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
                axes[1].set_title('Training Set - ROC Curve (N/A)', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # 保存圖片
            plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight', facecolor='white')
            print("✅ 訓練集評估圖表已保存為 'training_metrics.png'")
            
            # 嘗試顯示圖片（如果在支援的環境中）
            try:
                plt.show()
            except:
                print("📊 圖表已生成並保存，但無法在當前環境中顯示")
            
            # 關閉圖形以釋放記憶體
            plt.close(fig)
            
        except ImportError as e:
            print(f"⚠️ Warning: 缺少必要套件: {e}")
            print("請安裝必要套件: pip install seaborn scikit-learn matplotlib")
        except Exception as e:
            print(f"無法創建訓練集評估圖表: {e}")
            import traceback
            traceback.print_exc()

    def _compute_class_weights(self, device):
        """
        計算 class weights 用於平衡訓練
        
        Args:
            device: 運算設備 ('cuda' 或 'cpu')
            
        Returns:
            torch.Tensor: class weights tensor
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        # 使用當前（經過採樣後）的 y_train 來計算 class weights
        y_for_weights = self.train_target_df.values
        print("Using current (post-sampling) data distribution for class weight computation")
        
        # 獲取唯一類別
        unique_classes = np.unique(y_for_weights)
        
        # 計算 class weights
        class_weights = compute_class_weight(
            class_weight='balanced', 
            classes=unique_classes, 
            y=y_for_weights
        )
        
        # 轉換為 tensor
        class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        
        # 輸出 class weight 資訊
        print(f"Current class distribution: {dict(zip(*np.unique(y_for_weights, return_counts=True)))}")
        for i, (class_label, weight) in enumerate(zip(unique_classes, class_weights)):
            print(f"Class {class_label}: weight = {weight:.4f}")
        
        # 如果有原始資料分布，也顯示對比
        if hasattr(self, 'original_y_train'):
            original_distribution = dict(zip(*np.unique(self.original_y_train, return_counts=True)))
            print(f"Original class distribution (for reference): {original_distribution}")
        
        return class_weight_tensor

    def get_class_weights_info(self):
        """
        獲取 class weights 資訊，用於調試
        
        Returns:
            dict: 包含 class weights 和相關資訊的字典
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        # 使用當前（經過採樣後）的資料分布
        if hasattr(self, 'train_target_df') and self.train_target_df is not None:
            y_for_weights = self.train_target_df.values
            print("Using current (post-sampling) data distribution for class weight computation")
        else:
            print("⚠️ Warning: train_target_df not found. Run preprocess() first.")
            return None
        
        unique_classes = np.unique(y_for_weights)
        class_weights = compute_class_weight(
            class_weight='balanced', 
            classes=unique_classes, 
            y=y_for_weights
        )
        
        unique, counts = np.unique(y_for_weights, return_counts=True)
        current_class_distribution = dict(zip(unique, counts))
        
        result = {
            'current_class_distribution': current_class_distribution,
            'class_weights': dict(zip(unique_classes, class_weights)),
            'imbalance_ratio': max(counts) / min(counts) if min(counts) > 0 else float('inf')
        }
        
        # 如果有原始資料分布，也添加到結果中
        if hasattr(self, 'original_y_train'):
            original_unique, original_counts = np.unique(self.original_y_train, return_counts=True)
            result['original_class_distribution'] = dict(zip(original_unique, original_counts))
            result['original_imbalance_ratio'] = max(original_counts) / min(original_counts)
        
        return result

    # 評估模型
    def evaluate_model(self, model_name: str):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = FTTransformerModel(config=self.config)
        self.model.load_state_dict(torch.load(f"./models/{model_name}"))
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
        
        # 計算準確率
        accuracy = accuracy_score(true_labels, predictions)
        print(f'Test Accuracy: {accuracy:.4f}')
        
        # 顯示分類報告
        print('\nClassification Report:')
        print(classification_report(true_labels, predictions))
        
        # 顯示混淆矩陣
        print('\nConfusion Matrix:')
        cm = confusion_matrix(true_labels, predictions)
        print(cm)
        
        # 創建更詳細的混淆矩陣顯示
        print('\nDetailed Confusion Matrix:')
        unique_labels = sorted(list(set(true_labels + predictions)))
        
        # 打印表頭
        print('Predicted:', end='')
        for label in unique_labels:
            print(f'{label:>8}', end='')
        print()
        
        # 打印每一行
        for i, true_label in enumerate(unique_labels):
            print(f'Actual {true_label}:', end='')
            for j, pred_label in enumerate(unique_labels):
                print(f'{cm[i,j]:>8}', end='')
            print()
        
        # 可選：創建視覺化的混淆矩陣
        self._plot_confusion_matrix(cm, unique_labels)
        
        return accuracy, cm
    
    def _plot_confusion_matrix(self, cm, labels):
        """
        繪製混淆矩陣的視覺化圖表
        """
        try:
            import seaborn as sns
            
            # 設定標籤映射
            if len(labels) == 2:
                class_labels = ['Not Readmitted', 'Readmitted']
            else:
                class_labels = [f'Class {label}' for label in labels]
            
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                       xticklabels=class_labels, 
                       yticklabels=class_labels)
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.show()
            
            # 同時保存圖片
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            print("混淆矩陣圖表已保存為 'confusion_matrix.png'")
            
        except ImportError:
            print("Seaborn 未安裝，無法顯示視覺化混淆矩陣")
            print("請安裝 seaborn: pip install seaborn")
        except Exception as e:
            print(f"無法創建視覺化混淆矩陣: {e}")

    # 測試 test_df.csv 模型評估
    def test_csv_model_eval(self):
        # 使用訓練時的 LabelEncoder 來編碼類別特徵
        if hasattr(self, 'label_encoders'):
            for col in CATEGORICAL_FEATURES:
                # 使用訓練時保存的 LabelEncoder
                self.test_df_test_csv[col] = self.label_encoders[col].transform(self.test_df_test_csv[col])
                self.test_df_test_csv[col] = self.test_df_test_csv[col].astype('category')
        else:
            print("⚠️ Warning: No label encoders found. Run format_dataframe() first.")
            # 將類別特徵轉為 category 類型（原有邏輯）
            for col in CATEGORICAL_FEATURES:
                self.test_df_test_csv[col] = self.test_df_test_csv[col].astype('category')

        # 使用訓練時的 scaler 來標準化測試數據
        X_num_test_csv = self.test_df_test_csv[NUMERICAL_FEATURES].values.astype(np.float32)
        X_num_test_csv = self.scaler.transform(X_num_test_csv)  # 使用已經 fit 的 scaler
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

    def plot_train_roc_and_confusion_matrix(self, model_name=None):
        """
        單獨繪製訓練集的ROC曲線和混淆矩陣（無需重新訓練）
        
        Args:
            model_name (str, optional): 如果提供，會載入指定的模型文件
        """
        if model_name:
            # 載入指定的模型
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = FTTransformerModel(config=self.config)
            self.model.load_state_dict(torch.load(f"./models/{model_name}"))
            self.model.to(device)
            print(f"Model loaded from ./models/{model_name}")
        
        if self.model is None:
            print("⚠️ Error: No model available. Please train a model first or provide model_name.")
            return None
        
        print("🔍 Evaluating training set...")
        return self.evaluate_train_set()