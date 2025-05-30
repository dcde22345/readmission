# ModelEvaluator - 模型評估器

## 概述

`ModelEvaluator` 是一個專門設計用於評估 FT-Transformer 模型的 Python 類別，能夠：

1. ✅ 使用 `ft_transformer.py` 裡面的 `evaluate_model` 程式
2. ✅ 使用 `models` 資料夾裡面的所有模型
3. ✅ 找出 recall 最高的模型
4. ✅ 提供詳細的評估報告和結果保存功能

## 主要功能

### 🔍 自動模型發現
- 自動掃描 `models` 資料夾中的所有模型檔案
- 支援 `.pth` 格式和無副檔名的模型檔案
- 按檔案名稱排序進行有序評估

### 📊 全面評估指標
- **準確率 (Accuracy)**
- **Macro Recall**: 各類別 recall 的平均值
- **Weighted Recall**: 按樣本數量加權的 recall
- **各類別 Recall**: 每個類別的獨立 recall 值
- **混淆矩陣**: 完整的分類結果矩陣

### 🏆 最佳模型識別
- 自動比較所有模型的 Macro Recall
- 識別並報告 recall 最高的模型
- 提供詳細的最佳模型性能指標

### 💾 結果儲存
- 將評估結果儲存為 CSV 格式
- 包含所有關鍵指標的摘要表格
- 支援自訂輸出檔案名稱

## 安裝需求

```bash
pip install torch pandas numpy scikit-learn pathlib
```

## 檔案結構

```
project/
├── models/                          # 模型檔案資料夾
│   ├── ft_transformer_model.pth
│   ├── ft_transformer_model_smote.pth
│   └── ...
├── src/
│   └── ft_transformer.py           # 原始 FT-Transformer 實現
├── data/
│   ├── train_df.csv
│   └── test_df.csv
├── model_evaluator_final.py        # ModelEvaluator 主要類別
├── demo.py                          # 使用示例
└── README_ModelEvaluator.md         # 說明文件
```

## 快速開始

### 基本使用

```python
from model_evaluator_final import ModelEvaluator

# 1. 創建評估器
evaluator = ModelEvaluator(models_dir="models")

# 2. 評估所有模型
results = evaluator.evaluate_all_models()

# 3. 獲取最佳模型
best_model, best_score = evaluator.get_best_model()
print(f"最佳模型: {best_model}")
print(f"最佳 Recall: {best_score:.4f}")

# 4. 顯示詳細結果
evaluator.print_summary()

# 5. 保存結果
evaluator.save_results('evaluation_results.csv')
```

### 測試模式（限制模型數量）

```python
# 只評估前3個模型進行快速測試
results = evaluator.evaluate_all_models(max_models=3)
```

## 類別方法說明

### ModelEvaluator(models_dir="models")
**初始化評估器**
- `models_dir`: 模型檔案所在資料夾路徑

### evaluate_all_models(max_models=None)
**評估所有模型**
- `max_models`: 限制評估的最大模型數量（用於測試）
- 返回: 包含所有評估結果的字典

### evaluate_single_model(model_name)
**評估單一模型**
- `model_name`: 模型檔案名稱
- 返回: 該模型的詳細評估結果

### get_best_model()
**獲取最佳模型**
- 返回: (模型名稱, Macro Recall 分數)

### get_results_summary()
**獲取結果摘要**
- 返回: 包含所有模型評估結果的 pandas DataFrame

### print_summary()
**印出詳細摘要**
- 在控制台顯示格式化的評估結果表格

### save_results(output_file='model_evaluation_results.csv')
**保存結果到檔案**
- `output_file`: 輸出 CSV 檔案名稱

## 輸出結果格式

### 控制台輸出範例
```
📊 模型評估結果摘要
================================================================================
                              Model  Accuracy  Macro_Recall  Weighted_Recall  Class_0_Recall  Class_1_Recall Error
ft_transformer_model_smote.pth       0.7800        0.6250           0.7600          0.9000          0.3500      
ft_transformer_model.pth             0.7700        0.5300           0.7500          0.9000          0.1600      
...

🏆 最佳模型（按Macro Recall排序）:
   模型名稱: ft_transformer_model_smote.pth
   Macro Recall: 0.6250
   各類別Recall:
     class_0: 0.9000
     class_1: 0.3500
```

### CSV 輸出格式
| Model | Accuracy | Macro_Recall | Weighted_Recall | Class_0_Recall | Class_1_Recall | Error |
|-------|----------|--------------|-----------------|----------------|----------------|-------|
| ft_transformer_model_smote.pth | 0.7800 | 0.6250 | 0.7600 | 0.9000 | 0.3500 | |
| ft_transformer_model.pth | 0.7700 | 0.5300 | 0.7500 | 0.9000 | 0.1600 | |

## 進階功能

### 錯誤處理
- 自動捕獲並記錄模型載入錯誤
- 繼續評估其他可用模型
- 在結果中標記失敗的模型

### 記憶體優化
- 模型載入後自動清理 GPU 記憶體
- 支援 CPU 和 CUDA 環境
- 批次處理避免記憶體溢出

### 可擴展性
- 易於擴展支援其他評估指標
- 可客製化評估邏輯
- 支援不同的模型格式

## 注意事項

1. **資料一致性**: 確保所有模型都是使用相同的資料預處理方式訓練的
2. **檔案路徑**: 確認 `models` 資料夾存在且包含有效的模型檔案  
3. **記憶體需求**: 大型模型評估可能需要較多記憶體
4. **執行時間**: 評估所有模型可能需要一些時間，建議先使用 `max_models` 參數測試

## 故障排除

### 常見問題

**Q: 找不到模型檔案**
```
FileNotFoundError: 模型資料夾不存在: models
```
A: 確認 `models` 資料夾存在且包含 `.pth` 檔案

**Q: 模型載入失敗**
```
評估模型 xxx.pth 時發生錯誤: ...
```
A: 檢查模型檔案是否完整，確認模型架構一致性

**Q: CUDA 記憶體不足**
A: 減少 `max_models` 參數或使用 CPU 模式

### 技術支援
如有其他問題，請檢查：
1. PyTorch 版本兼容性
2. 依賴套件完整性
3. 資料檔案可用性（train_df.csv, test_df.csv）

---

## 授權
此專案遵循 MIT 授權條款。 