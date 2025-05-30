import numpy as np
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours

# 使用 SMOTE 處理類別不平衡
def apply_smote(X_num_train, X_cat_train, y_train, method='smote', random_state=42):
    """
    Apply SMOTE (Synthetic Minority Oversampling Technique) to handle class imbalance.
    
    Args:
        X_num_train (np.ndarray): Numerical training features (should be standardized)
        X_cat_train (np.ndarray): Categorical training features  
        y_train (np.ndarray): Training labels
        method (str): SMOTE method - 'smote', 'smotenc', or 'borderline'
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: Resampled (X_num_train, X_cat_train, y_train)
    """
    print(f"Applying {method.upper()} to handle class imbalance...")
    
    # Print original class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"Original class distribution: {dict(zip(unique, counts))}")
    
    # 檢查是否需要 SMOTE
    min_class_count = min(counts)
    max_class_count = max(counts)
    imbalance_ratio = max_class_count / min_class_count
    
    if imbalance_ratio < 1.5:
        print(f"Dataset is relatively balanced (ratio: {imbalance_ratio:.2f}). Skipping {method.upper()}.")
        return X_num_train, X_cat_train, y_train
    
    try:
        if method.lower() == 'smotenc':
            return apply_smotenc(X_num_train, X_cat_train, y_train, random_state)
        else:
            # 原本的 SMOTE 實現
            return apply_standard_smote(X_num_train, X_cat_train, y_train, method, random_state)
        
    except Exception as e:
        print(f"{method.upper()} failed: {e}")
        print("Continuing with original data...")
        return X_num_train, X_cat_train, y_train

def apply_smotenc(X_num_train, X_cat_train, y_train, random_state=42):
    """
    Apply SMOTENC specifically for mixed data types (numerical + categorical)
    
    Args:
        X_num_train (np.ndarray): Numerical training features
        X_cat_train (np.ndarray): Categorical training features  
        y_train (np.ndarray): Training labels
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: Resampled (X_num_train, X_cat_train, y_train)
    """
    print("Using SMOTENC for mixed data types...")
    
    # 檢查數據維度
    print(f"Numerical features shape: {X_num_train.shape}")
    print(f"Categorical features shape: {X_cat_train.shape}")
    
    # 合併數值和類別特徵
    if X_cat_train.shape[1] > 0:
        X_combined = np.concatenate([X_num_train, X_cat_train], axis=1)
        
        # 創建類別特徵索引（在合併後的特徵矩陣中的位置）
        categorical_indices = list(range(X_num_train.shape[1], X_combined.shape[1]))
        print(f"Categorical feature indices: {categorical_indices}")
        print(f"Combined features shape: {X_combined.shape}")
        
        # 檢查類別特徵是否為整數
        for i, cat_idx in enumerate(categorical_indices):
            cat_col = X_combined[:, cat_idx]
            if not np.all(cat_col == cat_col.astype(int)):
                print(f"⚠️ Warning: Categorical feature {i} contains non-integer values!")
                # 四捨五入並轉為整數
                X_combined[:, cat_idx] = np.round(cat_col).astype(int)
        
        # 檢查最小類別數量用於設定 k_neighbors
        unique, counts = np.unique(y_train, return_counts=True)
        min_class_count = min(counts)
        k_neighbors = min(5, max(1, min_class_count - 1))
        
        # 應用 SMOTENC
        smote_nc = SMOTENC(
            categorical_features=categorical_indices, 
            random_state=random_state,
            k_neighbors=k_neighbors
        )
        
        X_resampled, y_resampled = smote_nc.fit_resample(X_combined, y_train)
        
        # 分離數值和類別特徵
        X_num_resampled = X_resampled[:, :X_num_train.shape[1]]
        X_cat_resampled = X_resampled[:, X_num_train.shape[1]:]
        
        # 確保類別特徵為正確的數據類型
        X_cat_resampled = X_cat_resampled.astype(X_cat_train.dtype)
        
    else:
        # 如果沒有類別特徵，使用標準 SMOTE
        print("No categorical features found. Using standard SMOTE...")
        return apply_standard_smote(X_num_train, X_cat_train, y_train, 'smote', random_state)
    
    # Print results
    unique_new, counts_new = np.unique(y_resampled, return_counts=True)
    print(f"After SMOTENC class distribution: {dict(zip(unique_new, counts_new))}")
    print(f"Training set size: {len(y_train)} -> {len(y_resampled)}")
    
    return X_num_resampled, X_cat_resampled, y_resampled

def apply_standard_smote(X_num_train, X_cat_train, y_train, method='smote', random_state=42):
    """
    Apply standard SMOTE methods (original implementation)
    """
    unique, counts = np.unique(y_train, return_counts=True)
    min_class_count = min(counts)
    
    if X_num_train.shape[1] > 0:  # 如果有數值特徵
        # 只對數值特徵使用 SMOTE
        k_neighbors = min(5, max(1, min_class_count - 1))
        
        if method.lower() == 'borderline':
            try:
                from imblearn.over_sampling import BorderlineSMOTE
                smote = BorderlineSMOTE(random_state=random_state, k_neighbors=k_neighbors)
            except ImportError:
                print("BorderlineSMOTE not available, using standard SMOTE")
                smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        else:
            smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
            
        X_num_resampled, y_resampled = smote.fit_resample(X_num_train, y_train)
        
        # 對於類別特徵，使用最近鄰插值
        if X_cat_train.shape[1] > 0:
            from sklearn.neighbors import NearestNeighbors
            
            # 找到新樣本在原始樣本中的最近鄰
            nbrs = NearestNeighbors(n_neighbors=1).fit(X_num_train)
            _, indices = nbrs.kneighbors(X_num_resampled)
            
            # 使用最近鄰的類別特徵值
            X_cat_resampled = X_cat_train[indices.flatten()]
        else:
            X_cat_resampled = np.array([]).reshape(len(y_resampled), 0)
    else:
        # 如果沒有數值特徵，只對類別特徵使用 SMOTE
        print("Warning: No numerical features found. SMOTE may not work well with only categorical features.")
        
        # 將類別特徵轉換為數值進行 SMOTE
        X_combined = X_cat_train.astype(float)
        smote = SMOTE(random_state=random_state, k_neighbors=min(5, min_class_count-1))
        X_combined_resampled, y_resampled = smote.fit_resample(X_combined, y_train)
        
        # 將結果四捨五入並轉回整數（對類別特徵）
        X_cat_resampled = np.round(X_combined_resampled).astype(X_cat_train.dtype)
        X_num_resampled = np.array([]).reshape(len(y_resampled), 0)
    
    # Print new class distribution
    unique_new, counts_new = np.unique(y_resampled, return_counts=True)
    print(f"After {method.upper()} class distribution: {dict(zip(unique_new, counts_new))}")
    print(f"Training set size: {len(y_train)} -> {len(y_resampled)}")
    
    return X_num_resampled, X_cat_resampled, y_resampled

def apply_under_sampling(X_num_train, X_cat_train, y_train, method='random', random_state=42):
    """
    Apply Under Sampling to handle class imbalance by reducing majority class samples.
    
    Args:
        X_num_train (np.ndarray): Numerical training features (should be standardized)
        X_cat_train (np.ndarray): Categorical training features  
        y_train (np.ndarray): Training labels
        method (str): Under sampling method - 'random', 'tomek', 'enn'
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: Resampled (X_num_train, X_cat_train, y_train)
    """
    print(f"=== Applying {method} under sampling ===")
    
    # Print original class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"Original class distribution: {dict(zip(unique, counts))}")
    
    # 檢查是否需要 Under Sampling
    min_class_count = min(counts)
    max_class_count = max(counts)
    imbalance_ratio = max_class_count / min_class_count
    
    print(f"Imbalance ratio: {imbalance_ratio:.2f}")
    
    if imbalance_ratio < 1.5:
        print(f"Dataset is relatively balanced (ratio: {imbalance_ratio:.2f}). Skipping under sampling.")
        return X_num_train, X_cat_train, y_train
    
    try:
        # 合併數值和類別特徵
        if X_cat_train.shape[1] > 0:
            X_combined = np.concatenate([X_num_train, X_cat_train], axis=1)
        else:
            X_combined = X_num_train
        
        # 根據方法進行採樣
        print(f"Applying {method} under sampling...")
        if method == 'random':
            X_resampled, y_resampled = random_under_sampling(X_combined, y_train, random_state)
        elif method == 'tomek':
            X_resampled, y_resampled = tomek_links_under_sampling(X_combined, y_train)
        elif method == 'enn':
            X_resampled, y_resampled = enn_under_sampling(X_combined, y_train)
        else:
            print(f"Unknown method: {method}. Using random under sampling.")
            X_resampled, y_resampled = random_under_sampling(X_combined, y_train, random_state)
        
        # 分離數值和類別特徵
        if X_cat_train.shape[1] > 0:
            num_features_count = X_num_train.shape[1]
            X_num_resampled = X_resampled[:, :num_features_count]
            X_cat_resampled = X_resampled[:, num_features_count:]
        else:
            X_num_resampled = X_resampled
            X_cat_resampled = np.array([]).reshape(len(y_resampled), 0)
        
        # Print final results
        unique_new, counts_new = np.unique(y_resampled, return_counts=True)
        print(f"=== Under sampling results ===")
        print(f"Final class distribution: {dict(zip(unique_new, counts_new))}")
        print(f"Training set size: {len(y_train)} -> {len(y_resampled)}")
        print(f"Samples removed: {len(y_train) - len(y_resampled)}")
        
        # 檢查是否真的有變化
        if len(y_resampled) == len(y_train):
            print("⚠️ WARNING: Under sampling did not change the dataset size!")
            print("   Possible reasons:")
            print(f"   1. {method} method found no samples to remove")
            print("   2. Error in sampling implementation")
            print("   3. Data characteristics not suitable for this method")
        else:
            print(f"✅ Under sampling successful using {method} method")
        
        return X_num_resampled, X_cat_resampled, y_resampled
        
    except Exception as e:
        print(f"Under sampling failed: {e}")
        print("Continuing with original data...")
        return X_num_train, X_cat_train, y_train

def random_under_sampling(X, y, random_state=42):
    """
    Random under sampling: 隨機移除多數類樣本
    """
    unique, counts = np.unique(y, return_counts=True)
    min_count = min(counts)
    
    # 設定隨機種子
    np.random.seed(random_state)
    
    indices_to_keep = []
    for class_label in unique:
        class_indices = np.where(y == class_label)[0]
        if len(class_indices) > min_count:
            # 隨機選擇 min_count 個樣本
            selected_indices = np.random.choice(class_indices, min_count, replace=False)
            indices_to_keep.extend(selected_indices)
        else:
            # 保留所有樣本（少數類）
            indices_to_keep.extend(class_indices)
    
    indices_to_keep = np.array(indices_to_keep)
    return X[indices_to_keep], y[indices_to_keep]

def tomek_links_under_sampling(X, y):
    """
    Tomek Links under sampling: 移除 Tomek Links 中的多數類樣本
    """
    print("Attempting Tomek Links under sampling...")
    
    try:
        tomek = TomekLinks()
        X_resampled, y_resampled = tomek.fit_resample(X, y)
        
        print(f"After Tomek Links: {len(y_resampled)} samples")
        print(f"Removed {len(y) - len(y_resampled)} samples")
        print(f"New class distribution: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
        
        if len(y_resampled) == len(y):
            print("⚠️ Warning: Tomek Links found no samples to remove.")
            print("   This could mean:")
            print("   1. Dataset has no Tomek Links (good data quality)")
            print("   2. Classes are well separated")
            print("   3. Dataset is too small to find meaningful links")
            print("   Recommendation: Try 'random' or 'enn' method instead")
        
        return X_resampled, y_resampled
        
    except ImportError as e:
        print(f"imblearn import failed: {e}")
        print("Please install imblearn: pip install imbalanced-learn")
        print("Falling back to random under sampling...")
        return random_under_sampling(X, y)
    except Exception as e:
        print(f"Tomek Links failed with error: {e}")
        print("Falling back to random under sampling...")
        return random_under_sampling(X, y)

def enn_under_sampling(X, y):
    """
    Edited Nearest Neighbours under sampling: 移除被鄰居錯誤分類的樣本
    """
    try:
        enn = EditedNearestNeighbours()
        X_resampled, y_resampled = enn.fit_resample(X, y)
        return X_resampled, y_resampled
        
    except ImportError:
        print("imblearn not available for ENN. Using random under sampling.")
        return random_under_sampling(X, y)
    