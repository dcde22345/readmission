import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report
from typing import Dict, List, Tuple, Optional

# 將src資料夾加入路徑，以便導入ft_transformer模組
sys.path.append('src')
from ft_transformer import CustomerFTTransformer


class ModelEvaluator:
    """
    一個用於評估所有訓練好的模型並找出recall最高的模型類別
    
    功能:
    1. 使用ft_transformer.py裡面的evaluate_model程式
    2. 使用models資料夾裡面的所有model
    3. 找出recall最高的model
    4. 提供詳細的評估報告
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        初始化ModelEvaluator
        
        Args:
            models_dir (str): 模型檔案資料夾路徑
        """
        self.models_dir = Path(models_dir)
        
        # 初始化FT-Transformer（使用預設設定以避免複雜化）
        self.ft_transformer = CustomerFTTransformer()
        
        # 存儲評估結果
        self.evaluation_results: Dict[str, Dict] = {}
        self.best_recall_model: Optional[str] = None
        self.best_recall_score: float = 0.0
        
        # 預處理資料
        self._prepare_data()
    
    def _prepare_data(self):
        """準備資料進行評估"""
        print("🔄 正在準備資料...")
        
        try:
            # 格式化資料框
            self.ft_transformer.format_dataframe()
            
            # 預處理資料（使用預設設定）
            self.ft_transformer.preprocess(
                use_smote=False,
                use_under_sampling=False
            )
            
            # 設定特徵
            self.ft_transformer.set_feautres_processed()
            
            # 建立資料集
            self.ft_transformer.set_tablar_dataset()
            
            # 設定模型配置
            self.ft_transformer.set_model_config()
            
            print("✅ 資料準備完成")
            
        except Exception as e:
            print(f"❌ 資料準備失敗: {e}")
            raise
    
    def get_model_files(self) -> List[str]:
        """
        獲取models資料夾中的所有模型檔案
        
        Returns:
            List[str]: 模型檔案名稱列表
        """
        if not self.models_dir.exists():
            raise FileNotFoundError(f"模型資料夾不存在: {self.models_dir}")
        
        # 尋找所有.pth檔案和無副檔名的模型檔案
        model_files = []
        for file_path in self.models_dir.iterdir():
            if file_path.is_file():
                if (file_path.suffix == '.pth' or 
                    ('model' in file_path.name.lower() and file_path.suffix == '')):
                    model_files.append(file_path.name)
        
        if not model_files:
            raise ValueError(f"在 {self.models_dir} 中沒有找到模型檔案")
        
        return sorted(model_files)
    
    def evaluate_single_model(self, model_name: str) -> Dict:
        """
        評估單一模型並計算recall值
        
        Args:
            model_name (str): 模型檔案名稱
            
        Returns:
            Dict: 包含評估結果的字典
        """
        print(f"\n🔍 正在評估模型: {model_name}")
        
        try:
            # 使用ft_transformer的evaluate_model方法
            accuracy, confusion_matrix = self.ft_transformer.evaluate_model(model_name)
            
            # 獲取預測結果以計算詳細指標
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            predictions = []
            true_labels = []
            
            self.ft_transformer.model.eval()
            with torch.no_grad():
                for batch in self.ft_transformer.dl_test:
                    numerical = batch['continuous'].to(device)
                    categorical = batch['categorical'].to(device)
                    target = batch['target'].to(device).squeeze()
                    outputs = self.ft_transformer.model({'continuous': numerical, 'categorical': categorical})
                    _, preds = torch.max(outputs['logits'], dim=1)
                    predictions.extend(preds.cpu().numpy())
                    true_labels.extend(target.cpu().numpy())
            
            # 計算分類報告
            class_report = classification_report(true_labels, predictions, output_dict=True)
            
            # 提取recall值
            macro_recall = class_report['macro avg']['recall']
            weighted_recall = class_report['weighted avg']['recall']
            
            # 計算各類別的recall
            individual_recalls = {}
            for key, metrics in class_report.items():
                if key not in ['accuracy', 'macro avg', 'weighted avg']:
                    individual_recalls[f'class_{key}'] = metrics.get('recall', 0.0)
            
            result = {
                'model_name': model_name,
                'accuracy': accuracy,
                'macro_recall': macro_recall,
                'weighted_recall': weighted_recall,
                'individual_recalls': individual_recalls,
                'confusion_matrix': confusion_matrix.tolist(),
                'classification_report': class_report
            }
            
            print(f"✅ 模型 {model_name} 評估完成")
            print(f"   準確率: {accuracy:.4f}")
            print(f"   Macro Recall: {macro_recall:.4f}")
            print(f"   Weighted Recall: {weighted_recall:.4f}")
            
            return result
            
        except Exception as e:
            error_msg = f"評估模型 {model_name} 時發生錯誤: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'model_name': model_name,
                'error': error_msg,
                'accuracy': 0.0,
                'macro_recall': 0.0,
                'weighted_recall': 0.0
            }
    
    def evaluate_all_models(self, max_models: Optional[int] = None) -> Dict[str, Dict]:
        """
        評估所有模型
        
        Args:
            max_models (Optional[int]): 最大評估模型數量（用於測試）
            
        Returns:
            Dict[str, Dict]: 包含所有模型評估結果的字典
        """
        print(f"\n🚀 開始評估所有模型...")
        
        model_files = self.get_model_files()
        if max_models:
            model_files = model_files[:max_models]
            print(f"限制評估 {max_models} 個模型進行測試")
        
        print(f"找到 {len(model_files)} 個模型檔案:")
        for i, model_file in enumerate(model_files, 1):
            print(f"  {i}. {model_file}")
        
        # 重置結果
        self.evaluation_results = {}
        self.best_recall_model = None
        self.best_recall_score = 0.0
        
        # 評估每個模型
        for i, model_file in enumerate(model_files, 1):
            print(f"\n📍 進度: {i}/{len(model_files)}")
            result = self.evaluate_single_model(model_file)
            self.evaluation_results[model_file] = result
            
            # 更新最佳recall模型
            current_recall = result.get('macro_recall', 0.0)
            if current_recall > self.best_recall_score:
                self.best_recall_score = current_recall
                self.best_recall_model = model_file
                print(f"🎯 新的最佳模型: {model_file} (Macro Recall: {current_recall:.4f})")
        
        print(f"\n🎉 評估完成！")
        print(f"🏆 最佳模型: {self.best_recall_model}")
        print(f"🎯 最佳 Macro Recall: {self.best_recall_score:.4f}")
        
        return self.evaluation_results
    
    def get_best_model(self) -> Tuple[Optional[str], float]:
        """
        獲取recall最高的模型
        
        Returns:
            Tuple[Optional[str], float]: (最佳模型名稱, 最佳recall分數)
        """
        return self.best_recall_model, self.best_recall_score
    
    def get_results_summary(self) -> pd.DataFrame:
        """
        獲取所有模型的評估結果摘要
        
        Returns:
            pd.DataFrame: 包含所有模型評估結果的DataFrame
        """
        if not self.evaluation_results:
            return pd.DataFrame()
        
        summary_data = []
        for model_name, result in self.evaluation_results.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': result.get('accuracy', 0.0),
                'Macro_Recall': result.get('macro_recall', 0.0),
                'Weighted_Recall': result.get('weighted_recall', 0.0),
                'Class_0_Recall': result.get('individual_recalls', {}).get('class_0', 0.0),
                'Class_1_Recall': result.get('individual_recalls', {}).get('class_1', 0.0),
                'Error': result.get('error', '')
            })
        
        df = pd.DataFrame(summary_data)
        # 按照Macro Recall降序排列
        df = df.sort_values('Macro_Recall', ascending=False).reset_index(drop=True)
        
        return df
    
    def save_results(self, output_file: str = 'model_evaluation_results.csv'):
        """
        將評估結果保存到CSV檔案
        
        Args:
            output_file (str): 輸出檔案名稱
        """
        summary_df = self.get_results_summary()
        if not summary_df.empty:
            summary_df.to_csv(output_file, index=False)
            print(f"📊 評估結果已保存到: {output_file}")
        else:
            print("⚠️ 沒有評估結果可保存")
    
    def print_summary(self):
        """印出評估結果摘要"""
        if not self.evaluation_results:
            print("⚠️ 尚未進行模型評估")
            return
        
        print("\n" + "="*80)
        print("📊 模型評估結果摘要")
        print("="*80)
        
        summary_df = self.get_results_summary()
        print(summary_df.to_string(index=False, float_format='{:.4f}'.format))
        
        print(f"\n🏆 最佳模型（按Macro Recall排序）:")
        print(f"   模型名稱: {self.best_recall_model}")
        print(f"   Macro Recall: {self.best_recall_score:.4f}")
        
        # 顯示各類別詳細recall
        if self.best_recall_model and self.best_recall_model in self.evaluation_results:
            best_result = self.evaluation_results[self.best_recall_model]
            if 'individual_recalls' in best_result:
                print(f"   各類別Recall:")
                for class_name, recall in best_result['individual_recalls'].items():
                    print(f"     {class_name}: {recall:.4f}")


def main():
    """主程式"""
    print("🚀 啟動模型評估器...")
    
    try:
        # 創建評估器
        evaluator = ModelEvaluator(models_dir="models")
        
        # 評估所有模型
        results = evaluator.evaluate_all_models()
        
        # 顯示結果摘要
        evaluator.print_summary()
        
        # 保存結果
        evaluator.save_results('model_evaluation_results.csv')
        
        print("\n✅ 評估完成！")
        
    except Exception as e:
        print(f"❌ 程式執行失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 