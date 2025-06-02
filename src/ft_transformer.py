import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from tqdm import tqdm
import traceback

from torch.utils.data import Dataset, DataLoader
from torchkeras.tabular import TabularPreprocessor, TabularDataset
from torchkeras.tabular.models import FTTransformerConfig, FTTransformerModel

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.imbalance_data_processing import apply_smote, apply_under_sampling
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight

np.random.seed(42)

# ----- 數值型與類別型特徵 -----
NUMERICAL_FEATURES = ['age', 'num_procedures', 'days_in_hospital', 'comorbidity_score']
CATEGORICAL_FEATURES = ['gender', 'primary_diagnosis', 'discharge_to']
TARGET = 'readmitted'
NUMERICAL_COLUMNS = [f'num_{i}' for i in range(len(NUMERICAL_FEATURES))]
CATEGORICAL_COLUMNS = [f'cat_{i}' for i in range(len(CATEGORICAL_FEATURES))]

class CustomerFTTransformer:
    def __init__(self, num_attn_blocks=1, dropout=0.1):
        self.num_attn_blocks = num_attn_blocks
        self.dropout = dropout

        # 讀取訓練數據
        self.train_df = pd.read_csv("data/train_df.csv")

        # 對訓練數據進行shuffle，確保數據隨機性
        self.train_df = self.train_df.sample(frac=1, random_state=42).reset_index(drop=True)
        self.test_df_test_csv = pd.read_csv("data/test_df.csv")

        self.scaler = StandardScaler()
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
            num_attn_blocks=self.num_attn_blocks,  # 使用實例變量
            ff_dropout=self.dropout,
            attn_dropout=self.dropout,
            embedding_dropout=self.dropout,
            add_norm_dropout=self.dropout,
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

        self.dl_train = DataLoader(self.ds_train, batch_size=256, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
        self.dl_test = DataLoader(self.ds_test, batch_size=256, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

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
    def train_model(self, num_epochs: int, model_name: str, use_class_weight=False, plot_train_metrics=True,
                   enable_cv=False, k_folds=3, save_best_model=True, early_stopping=True, patience=10, min_delta=1e-4):
        """
        訓練 FT-Transformer 模型（支持 Cross Validation）
        
        Args:
            num_epochs (int): 訓練輪數
            model_name (str): 模型保存名稱
            use_class_weight (bool): 是否使用 class weight 來處理類別不平衡
            plot_train_metrics (bool): 是否在訓練後繪製訓練集的ROC和混淆矩陣
            enable_cv (bool): 是否啟用 Cross Validation
            k_folds (int): K-fold CV 的 fold 數量（當 enable_cv=True 時有效）
            save_best_model (bool): 是否保存最佳模型（CV模式下）
            early_stopping (bool): 是否啟用 early stopping
            patience (int): early stopping 的耐心值（多少個 epoch 沒有改善就停止）
            min_delta (float): 最小改善閾值
            
        Returns:
            dict: 訓練結果（包含CV結果如果啟用的話）
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 顯示GPU信息
        if torch.cuda.is_available():
            print(f"🖥️ GPU 可用: {torch.cuda.get_device_name(0)}")
            print(f"🔢 可用GPU數量: {torch.cuda.device_count()}")
            print(f"💾 GPU記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            if torch.cuda.device_count() > 1:
                print(f"🚀 將使用 {torch.cuda.device_count()} 個GPU進行訓練")
        else:
            print("⚠️ 未檢測到CUDA設備，將使用CPU訓練")
        
        if not enable_cv:
            # 原有的常規訓練模式
            self._train_single_model(num_epochs, model_name, use_class_weight, 
                                          plot_train_metrics, device, early_stopping, patience, min_delta)
        else:
            # Cross Validation 模式
            self._train_with_cross_validation(num_epochs, model_name, use_class_weight,
                                                   k_folds, save_best_model, device, early_stopping, patience, min_delta)
    
    # 單一模型訓練
    def _train_single_model(self, num_epochs, model_name, use_class_weight, plot_train_metrics, device, early_stopping, patience, min_delta):
        """
        原有的單一模型訓練方法
        """
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
        
        # 多GPU支持
        if torch.cuda.device_count() > 1:
            print(f"🔗 Using {torch.cuda.device_count()} GPUs for training")
            self.model = torch.nn.DataParallel(self.model)
        
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
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), f'./models/{model_name}')
        print(f"Model saved as ./models/{model_name}")
        
        # 繪製訓練集評估指標
        if plot_train_metrics:
            print("\n=== Training Set Evaluation ===")
            self.evaluate_train_set(model_name=model_name)
            
        return {"model_name": model_name, "training_completed": True}
    
    # 交叉驗證訓練
    def _train_with_cross_validation(self, num_epochs, model_name, use_class_weight, k_folds, save_best_model, device, early_stopping, patience, min_delta):
        """
        執行 K-fold Cross Validation 訓練
        """
        
        print(f"\n🔄 開始 {k_folds}-Fold Cross Validation 訓練...")
        
        # 準備完整的訓練數據
        X_train_full = self.train_df_train_csv_processed.values
        y_train_full = self.train_target_df.values
        
        # 初始化 KFold
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        # 儲存每個 fold 的結果
        cv_results = {
            'fold_accuracies': [],
            'fold_losses': [],
            'fold_models': [],
            'fold_metrics': [],
            'mean_accuracy': 0.0,
            'std_accuracy': 0.0,
            'mean_loss': 0.0,
            'std_loss': 0.0
        }
        
        best_accuracy = 0.0
        best_model_state = None
        best_fold = -1
        
        # 執行 K-fold CV
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_full)):
            print(f"\n📋 Fold {fold + 1}/{k_folds}")
            print("-" * 50)
            
            # 分割當前 fold 的數據
            X_train_fold = X_train_full[train_idx]
            X_val_fold = X_train_full[val_idx]
            y_train_fold = y_train_full[train_idx]
            y_val_fold = y_train_full[val_idx]
            
            # 創建當前 fold 的 DataLoader
            train_dl_fold, val_dl_fold = self._create_fold_dataloaders(
                X_train_fold, X_val_fold, y_train_fold, y_val_fold
            )
            
            # 重新初始化模型（每個 fold 使用新的模型）
            model_fold = FTTransformerModel(config=self.config)
            model_fold.to(device)

            # 訓練當前 fold 的模型
            fold_accuracy, fold_loss, model_state = self._train_fold(
                model_fold, train_dl_fold, val_dl_fold, num_epochs, 
                use_class_weight, device, fold + 1, early_stopping, patience, min_delta
            )
            
            # 保存結果
            cv_results['fold_accuracies'].append(fold_accuracy)
            cv_results['fold_losses'].append(fold_loss)
            cv_results['fold_models'].append(model_state)
            
            # 詳細評估當前 fold
            fold_metrics = self._evaluate_fold(model_fold, val_dl_fold, device)
            cv_results['fold_metrics'].append(fold_metrics)
            
            # 檢查是否是最佳模型
            if fold_accuracy > best_accuracy:
                best_accuracy = fold_accuracy
                best_model_state = model_state
                best_fold = fold + 1
            
            print(f"Fold {fold + 1} - Validation Accuracy: {fold_accuracy:.4f}, Loss: {fold_loss:.4f}")
        
        # 計算 CV 統計結果
        cv_results['mean_accuracy'] = np.mean(cv_results['fold_accuracies'])
        cv_results['std_accuracy'] = np.std(cv_results['fold_accuracies'])
        cv_results['mean_loss'] = np.mean(cv_results['fold_losses'])
        cv_results['std_loss'] = np.std(cv_results['fold_losses'])
        cv_results['best_fold'] = best_fold
        cv_results['best_accuracy'] = best_accuracy
        
        # 保存最佳模型
        if save_best_model and best_model_state is not None:
            # 載入最佳模型到當前的instance
            self.model.load_state_dict(best_model_state)
            
            # 保存最佳模型
            torch.save(best_model_state, f'./models/{model_name}_best_cv.pth')
            print(f"💾 最佳模型已保存: ./models/{model_name}_best_cv.pth (來自 Fold {best_fold})")
            
        # 輸出CV總結
        print(f"\n{'='*60}")
        print(f"🎯 Cross Validation 總結")
        print(f"{'='*60}")
        print(f"平均準確率: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
        print(f"平均損失: {cv_results['mean_loss']:.4f} ± {cv_results['std_loss']:.4f}")
        print(f"最佳模型來自: Fold {best_fold} (準確率: {best_accuracy:.4f})")
        print(f"{'='*60}")
        
        cv_results['cv_completed'] = True


        # 繪製每個fold的混淆矩陣
        print("\n📊 Plotting confusion matrices for each fold...")
        for fold_idx, fold_metric in enumerate(cv_results['fold_metrics']):
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(fold_metric['true_labels'], fold_metric['predictions'])
            
            # 使用seaborn繪製混淆矩陣熱圖
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - Fold {fold_idx + 1}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # 保存圖片
            plt.savefig(f'./images/{model_name}_fold{fold_idx + 1}_cm.png')
            plt.close()
            
            print(f"Confusion matrix for fold {fold_idx + 1} saved as: ./images/{model_name}_fold{fold_idx + 1}_cm.png")
        
        return cv_results
    
    # 為當前 fold 創建 DataLoader
    def _create_fold_dataloaders(self, X_train_fold, X_val_fold, y_train_fold, y_val_fold, batch_size=128):
        """
        為當前 fold 創建 DataLoader
        """
        # 轉換為 DataFrame 格式
        train_fold_df = pd.DataFrame(X_train_fold, columns=self.train_df_train_csv_processed.columns)
        val_fold_df = pd.DataFrame(X_val_fold, columns=self.train_df_train_csv_processed.columns)
        
        # 添加 target 列
        train_fold_df['target'] = y_train_fold
        val_fold_df['target'] = y_val_fold
        
        # 創建 TabularDataset
        train_ds_fold = TabularDataset(
            data=train_fold_df,
            task='classification',
            target=['target'],
            continuous_cols=self.numerical_features_processed,
            categorical_cols=self.embedding_features_processed
        )
        
        val_ds_fold = TabularDataset(
            data=val_fold_df,
            task='classification',
            target=['target'],
            continuous_cols=self.numerical_features_processed,
            categorical_cols=self.embedding_features_processed
        )
        
        # 創建 DataLoader（GPU 優化設置）
        train_dl_fold = DataLoader(
            train_ds_fold, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True, 
            persistent_workers=True,
            drop_last=True  # 避免最後一個batch太小
        )
        val_dl_fold = DataLoader(
            val_ds_fold, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True, 
            persistent_workers=True
        )
        
        return train_dl_fold, val_dl_fold
    
    # 訓練單個 fold 的模型（支持 Early Stopping）
    def _train_fold(self, model, train_dl, val_dl, num_epochs, use_class_weight, device, fold_num, early_stopping, patience, min_delta):
        """
        訓練單個 fold 的模型（支持 Early Stopping）
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # 設定損失函數
        if use_class_weight:
            class_weight_tensor = self._compute_class_weights(device)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weight_tensor)
        else:
            criterion = torch.nn.CrossEntropyLoss()
        
        model.to(device)
        model.train()
        
        # Early Stopping 相關變數
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        early_stopped = False
        
        # 訓練循環
        progress_bar = tqdm(range(num_epochs), leave=False, desc=f"Fold {fold_num}")
        for epoch in progress_bar:
            # === 訓練階段 ===
            model.train()
            total_train_loss = 0
            for batch in train_dl:
                numerical = batch['continuous'].to(device)
                categorical = batch['categorical'].to(device)
                target = batch['target'].to(device).squeeze()
                
                optimizer.zero_grad()
                outputs = model({'continuous': numerical, 'categorical': categorical})
                logits = outputs['logits']
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_dl)
            
            # === 驗證階段 ===
            model.eval()
            total_val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_dl:
                    numerical = batch['continuous'].to(device)
                    categorical = batch['categorical'].to(device)
                    target = batch['target'].to(device).squeeze()
                    
                    outputs = model({'continuous': numerical, 'categorical': categorical})
                    logits = outputs['logits']
                    loss = criterion(logits, target)
                    
                    total_val_loss += loss.item()
                    _, predicted = torch.max(logits, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            avg_val_loss = total_val_loss / len(val_dl)
            val_accuracy = correct / total
            
            # === Early Stopping 檢查 ===
            if early_stopping:
                # 檢查是否有改善
                if avg_val_loss < best_val_loss - min_delta:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    model_to_save = model.module if hasattr(model, 'module') else model
                    best_model_state = model_to_save.state_dict().copy()
                else:
                    patience_counter += 1
                
                # 檢查是否需要 early stop
                if patience_counter >= patience:
                    early_stopped = True
                    print(f"\n⏹️ Early stopping triggered at epoch {epoch + 1}")
                    print(f"Best validation loss: {best_val_loss:.4f}")
                    break
            else:
                # 不使用 early stopping 時，總是保存最新的模型
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    model_to_save = model.module if hasattr(model, 'module') else model
                    best_model_state = model_to_save.state_dict().copy()
            
            # 更新進度條
            progress_bar.set_description(
                f"Fold {fold_num} - Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.4f} | Patience: {patience_counter}/{patience}"
            )
        
        # 恢復最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # 最終驗證
        final_val_accuracy, final_val_loss = self._validate_fold(model, val_dl, criterion, device)
        
        # 輸出訓練結果
        if early_stopped:
            print(f"✅ Fold {fold_num} completed with early stopping after {epoch + 1} epochs")
        else:
            print(f"✅ Fold {fold_num} completed full training ({num_epochs} epochs)")
        
        print(f"Final validation - Accuracy: {final_val_accuracy:.4f}, Loss: {final_val_loss:.4f}")
        
        # 返回時確保返回正確的模型狀態
        if best_model_state is not None:
            return final_val_accuracy, final_val_loss, best_model_state
        else:
            model_to_save = model.module if hasattr(model, 'module') else model
            return final_val_accuracy, final_val_loss, model_to_save.state_dict().copy()
    
    # 在驗證集上評估模型
    def _validate_fold(self, model, val_dl, criterion, device):
        """
        在驗證集上評估模型
        """
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_dl:
                numerical = batch['continuous'].to(device)
                categorical = batch['categorical'].to(device)
                target = batch['target'].to(device).squeeze()
                
                outputs = model({'continuous': numerical, 'categorical': categorical})
                logits = outputs['logits']
                loss = criterion(logits, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(val_dl)
        
        return accuracy, avg_loss
    
    # 詳細評估單個 fold 的性能
    def _evaluate_fold(self, model, val_dl, device):
        """
        詳細評估單個 fold 的性能
        """
        model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in val_dl:
                numerical = batch['continuous'].to(device)
                categorical = batch['categorical'].to(device)
                target = batch['target'].to(device).squeeze()
                
                outputs = model({'continuous': numerical, 'categorical': categorical})
                logits = outputs['logits']
                _, preds = torch.max(logits, 1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(target.cpu().numpy())
        
        # 計算詳細指標
        
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': predictions,
            'true_labels': true_labels
        }

    # 評估訓練集並繪製ROC曲線和混淆矩陣
    def evaluate_train_set(self, model_name:str):
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
        
        # 計算混淆矩陣
        cm = confusion_matrix(true_labels, predictions)
        print('\nTraining Set Confusion Matrix:')
        print(cm)
        
        # 繪製混淆矩陣和ROC曲線
        self._plot_train_metrics(true_labels, predictions, probabilities, cm, model_name=model_name)
        
        return accuracy, cm

    # 繪製訓練集的ROC曲線和混淆矩陣
    def _plot_train_metrics(self, true_labels, predictions, probabilities, cm, model_name="model"):
        """
        繪製訓練集的ROC曲線和混淆矩陣
        
        Args:
            true_labels: 真實標籤
            predictions: 預測標籤
            probabilities: 預測概率
            cm: 混淆矩陣
            model_name: 模型名稱，用於保存文件名
        """
        try:
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
            
            # 保存圖片，使用model_name命名
            filename = f'{model_name}_training_metrics.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ 訓練集評估圖表已保存為 '{filename}'")
            
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
            traceback.print_exc()

    # 計算 class weights 用於平衡訓練
    def _compute_class_weights(self, device):
        """
        計算 class weights 用於平衡訓練
        
        Args:
            device: 運算設備 ('cuda' 或 'cpu')
            
        Returns:
            torch.Tensor: class weights tensor
        """
        
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

    # 獲取 class weights 資訊，用於調試
    def get_class_weights_info(self):
        """
        獲取 class weights 資訊，用於調試
        
        Returns:
            dict: 包含 class weights 和相關資訊的字典
        """        
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
        
        # 嘗試載入模型檔案
        try:
            model_state = torch.load(f"./models/{model_name}")
            print(f"✅ 模型檔案載入成功: ./models/{model_name}")
        except FileNotFoundError:
            print(f"❌ 模型檔案不存在: ./models/{model_name}")
            return None, None
        
        # 檢查模型架構是否匹配
        try:
            # 先嘗試用當前配置載入
            self.model = FTTransformerModel(config=self.config)
            self.model.load_state_dict(model_state)
            print("✅ 模型架構匹配，直接載入成功")
        except RuntimeError as e:
            if "Unexpected key(s) in state_dict" in str(e):
                print("⚠️ 模型架構不匹配，嘗試自動調整...")
                
                # 分析保存的模型有多少個attention blocks
                attention_block_keys = [key for key in model_state.keys() if "mha_block_" in key]
                if attention_block_keys:
                    # 提取最大的block編號
                    max_block_num = max([int(key.split("mha_block_")[1].split(".")[0]) for key in attention_block_keys])
                    inferred_num_blocks = max_block_num + 1  # block編號從0開始
                    
                    print(f"🔍 檢測到保存的模型有 {inferred_num_blocks} 個attention blocks")
                    print(f"📝 當前配置的attention blocks: {self.num_attn_blocks}")
                    
                    # 更新實例變量
                    self.num_attn_blocks = inferred_num_blocks
                    
                    # 重新創建配置
                    adjusted_config = FTTransformerConfig(
                        task="classification",
                        num_attn_blocks=self.num_attn_blocks,
                    )
                    adjusted_config = adjusted_config.merge_dataset_config(self.ds_train)
                    
                    # 用調整後的配置創建模型
                    self.model = FTTransformerModel(config=adjusted_config)
                    self.model.load_state_dict(model_state)
                    print(f"✅ 模型架構已調整為 {self.num_attn_blocks} 個attention blocks並載入成功")
                    
                    # 更新模型配置
                    self.config = adjusted_config
                else:
                    print("❌ 無法自動判斷模型架構，請檢查模型檔案")
                    raise e
            else:
                raise e
        
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

    # 單獨繪製訓練集的ROC曲線和混淆矩陣（無需重新訓練）
    def plot_train_roc_and_confusion_matrix(self, model_name=None):
        """
        單獨繪製訓練集的ROC曲線和混淆矩陣（無需重新訓練）
        
        Args:
            model_name (str, optional): 如果提供，會載入指定的模型文件
        """
        if model_name:
            # 載入指定的模型
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # 嘗試載入模型檔案
            try:
                model_state = torch.load(f"./models/{model_name}")
                print(f"✅ 模型檔案載入成功: ./models/{model_name}")
            except FileNotFoundError:
                print(f"❌ 模型檔案不存在: ./models/{model_name}")
                return None
            
            # 檢查模型架構是否匹配
            try:
                # 先嘗試用當前配置載入
                self.model = FTTransformerModel(config=self.config)
                self.model.load_state_dict(model_state)
                print("✅ 模型架構匹配，直接載入成功")
            except RuntimeError as e:
                if "Unexpected key(s) in state_dict" in str(e):
                    print("⚠️ 模型架構不匹配，嘗試自動調整...")
                    
                    # 分析保存的模型有多少個attention blocks
                    attention_block_keys = [key for key in model_state.keys() if "mha_block_" in key]
                    if attention_block_keys:
                        # 提取最大的block編號
                        max_block_num = max([int(key.split("mha_block_")[1].split(".")[0]) for key in attention_block_keys])
                        inferred_num_blocks = max_block_num + 1  # block編號從0開始
                        
                        print(f"🔍 檢測到保存的模型有 {inferred_num_blocks} 個attention blocks")
                        print(f"📝 當前配置的attention blocks: {self.num_attn_blocks}")
                        
                        # 更新實例變量
                        self.num_attn_blocks = inferred_num_blocks
                        
                        # 重新創建配置
                        adjusted_config = FTTransformerConfig(
                            task="classification",
                            num_attn_blocks=self.num_attn_blocks,
                        )
                        adjusted_config = adjusted_config.merge_dataset_config(self.ds_train)
                        
                        # 用調整後的配置創建模型
                        self.model = FTTransformerModel(config=adjusted_config)
                        self.model.load_state_dict(model_state)
                        print(f"✅ 模型架構已調整為 {self.num_attn_blocks} 個attention blocks並載入成功")
                        
                        # 更新模型配置
                        self.config = adjusted_config
                    else:
                        print("❌ 無法自動判斷模型架構，請檢查模型檔案")
                        raise e
                else:
                    raise e
            
            self.model.to(device)
            print(f"Model loaded from ./models/{model_name}")
        
        if self.model is None:
            print("⚠️ Error: No model available. Please train a model first or provide model_name.")
            return None
        
        print("🔍 Evaluating training set...")
        return self.evaluate_train_set(model_name=model_name)

    def get_model_info(self):
        """
        顯示當前模型配置資訊
        """
        print(f"\n{'='*50}")
        print(f"🤖 模型配置資訊")
        print(f"{'='*50}")
        print(f"Attention Blocks: {self.num_attn_blocks}")
        print(f"Task: {self.model_config.task}")
        
        if hasattr(self, 'config') and self.config is not None:
            print(f"配置已合併: ✅")
            if hasattr(self.config, 'd_out'):
                print(f"輸出維度: {self.config.d_out}")
        else:
            print(f"配置已合併: ❌")
            
        if hasattr(self, 'model') and self.model is not None:
            print(f"模型已初始化: ✅")
            # 計算模型參數數量
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"總參數數量: {total_params:,}")
            print(f"可訓練參數: {trainable_params:,}")
        else:
            print(f"模型已初始化: ❌")
        
        print(f"{'='*50}")
        
        return {
            'num_attn_blocks': self.num_attn_blocks,
            'task': self.model_config.task,
            'config_merged': hasattr(self, 'config') and self.config is not None,
            'model_initialized': hasattr(self, 'model') and self.model is not None
        }