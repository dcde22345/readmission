import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # 使用非互動式後端

# 將src資料夾加入路徑，以便導入ft_transformer模組
sys.path.append('src')
from ft_transformer import CustomerFTTransformer


class ModelEvaluator:
    """
    一個用於評估所有訓練好的模型並找出recall最高的模型類別
    
    This class handles:
    1. 使用ft_transformer.py裡面的evaluate_model程式
    2. 使用models資料夾裡面的所有model
    3. 找出recall最高的model
    4. 為每個模型生成視覺化圖片並存到images資料夾
    """
    
    def __init__(self, models_dir: str = "models", images_dir: str = "images", 
                 use_smote: bool = False, use_under_sampling: bool = False, 
                 smote_method: str = 'smotenc', under_sampling_method: str = 'tomek'):
        """
        初始化ModelEvaluator
        
        Args:
            models_dir (str): 模型檔案資料夾路徑
            images_dir (str): 圖片輸出資料夾路徑
            use_smote (bool): 是否使用SMOTE進行資料平衡
            use_under_sampling (bool): 是否使用under sampling
            smote_method (str): SMOTE方法
            under_sampling_method (str): Under sampling方法
        """
        self.models_dir = Path(models_dir)
        self.images_dir = Path(images_dir)
        self.use_smote = use_smote
        self.use_under_sampling = use_under_sampling
        self.smote_method = smote_method
        self.under_sampling_method = under_sampling_method
        
        # 創建images資料夾（如果不存在）
        self.images_dir.mkdir(exist_ok=True)
        print(f"📁 圖片將保存到: {self.images_dir}")
        
        # 初始化FT-Transformer
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
        
        # 格式化資料框
        self.ft_transformer.format_dataframe()
        
        # 預處理資料
        self.ft_transformer.preprocess(
            use_smote=self.use_smote,
            use_under_sampling=self.use_under_sampling,
            smote_method=self.smote_method,
            under_sampling_method=self.under_sampling_method
        )
        
        # 設定特徵（使用正確的方法名稱）
        self.ft_transformer.set_feautres_processed()
        
        # 建立資料集（使用正確的方法名稱）
        self.ft_transformer.set_tablar_dataset()
        
        # 設定模型配置
        self.ft_transformer.set_model_config()
        
        print("✅ 資料準備完成")
    
    def _plot_confusion_matrix(self, cm, labels, model_name: str):
        """
        為特定模型繪製混淆矩陣的視覺化圖表並保存
        
        Args:
            cm: 混淆矩陣
            labels: 標籤列表
            model_name: 模型名稱
        """
        try:
            # 設定標籤映射
            if len(labels) == 2:
                class_labels = ['Not Readmitted', 'Readmitted']
            else:
                class_labels = [f'Class {label}' for label in labels]
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                       xticklabels=class_labels, 
                       yticklabels=class_labels)
            plt.title(f"Confusion Matrix - {model_name}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            
            # 清理模型名稱用於檔案名稱
            safe_model_name = model_name.replace('.pth', '').replace('/', '_').replace('\\', '_')
            confusion_matrix_file = self.images_dir / f"confusion_matrix_{safe_model_name}.png"
            
            plt.savefig(confusion_matrix_file, dpi=300, bbox_inches='tight')
            print(f"  🖼️ 混淆矩陣圖表已保存: {confusion_matrix_file}")
            plt.close()
            
        except Exception as e:
            print(f"  ❌ 無法創建模型 {model_name} 的混淆矩陣圖表: {e}")
    
    def _generate_model_performance_chart(self, model_name: str, result: Dict):
        """
        為特定模型生成性能指標圖表
        
        Args:
            model_name: 模型名稱
            result: 評估結果字典
        """
        try:
            if 'individual_recalls' not in result:
                return
                
            # 準備數據
            metrics = {
                'Accuracy': result.get('accuracy', 0.0),
                'Macro Recall': result.get('macro_recall', 0.0),
                'Weighted Recall': result.get('weighted_recall', 0.0)
            }
            
            # 添加各類別recall
            for class_name, recall_val in result['individual_recalls'].items():
                metrics[f'{class_name} Recall'] = recall_val
            
            # 創建圖表
            fig, ax = plt.subplots(figsize=(10, 6))
            
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            bars = ax.bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'lightpink'])
            
            # 添加數值標籤
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{value:.3f}', ha='center', va='bottom')
            
            ax.set_ylim(0, 1.1)
            ax.set_ylabel('Score')
            ax.set_title(f'Performance Metrics - {model_name}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # 保存圖片
            safe_model_name = model_name.replace('.pth', '').replace('/', '_').replace('\\', '_')
            performance_file = self.images_dir / f"performance_{safe_model_name}.png"
            
            plt.savefig(performance_file, dpi=300, bbox_inches='tight')
            print(f"  🖼️ 性能圖表已保存: {performance_file}")
            plt.close()
            
        except Exception as e:
            print(f"  ❌ 無法創建模型 {model_name} 的性能圖表: {e}")
    
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
                # 包含.pth檔案和看起來像模型的檔案
                if (file_path.suffix == '.pth' or 
                    ('model' in file_path.name.lower() and file_path.suffix == '')):
                    model_files.append(file_path.name)
        
        if not model_files:
            raise ValueError(f"在 {self.models_dir} 中沒有找到模型檔案")
        
        print(f"找到 {len(model_files)} 個模型檔案:")
        for model_file in model_files:
            print(f"  - {model_file}")
        
        return sorted(model_files)
    
    def evaluate_single_model(self, model_name: str) -> Dict:
        """
        評估單一模型
        
        Args:
            model_name (str): 模型檔案名稱
            
        Returns:
            Dict: 包含評估結果的字典
        """
        print(f"\n🔍 正在評估模型: {model_name}")
        
        try:
            # 使用ft_transformer的evaluate_model方法
            accuracy, confusion_matrix = self.ft_transformer.evaluate_model(model_name)
            
            # 獲取詳細的分類報告
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # 重新獲取預測結果以計算詳細指標
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
            
            # 計算分類報告字典格式
            class_report = classification_report(true_labels, predictions, output_dict=True)
            
            # 提取recall值 (針對每個類別和總體)
            recalls = {}
            for key, metrics in class_report.items():
                if key not in ['accuracy', 'macro avg', 'weighted avg']:
                    recalls[f'class_{key}'] = metrics.get('recall', 0.0)
            
            # 使用macro average recall作為主要指標
            macro_recall = class_report['macro avg']['recall']
            weighted_recall = class_report['weighted avg']['recall']
            
            result = {
                'model_name': model_name,
                'accuracy': accuracy,
                'macro_recall': macro_recall,
                'weighted_recall': weighted_recall,
                'individual_recalls': recalls,
                'confusion_matrix': confusion_matrix.tolist(),
                'full_classification_report': class_report
            }
            
            print(f"✅ 模型 {model_name} 評估完成")
            print(f"   準確率: {accuracy:.4f}")
            print(f"   Macro Recall: {macro_recall:.4f}")
            print(f"   Weighted Recall: {weighted_recall:.4f}")
            
            # 為模型生成混淆矩陣圖片
            self._plot_confusion_matrix(confusion_matrix, ['Not Readmitted', 'Readmitted'], model_name)
            
            # 為模型生成性能圖表
            self._generate_model_performance_chart(model_name, result)
            
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
    
    def generate_model_comparison_chart(self):
        """
        生成所有模型的比較圖表
        """
        if not self.evaluation_results:
            print("⚠️ 尚未進行模型評估，無法生成比較圖表")
            return
        
        try:
            # 準備數據
            models = []
            accuracies = []
            macro_recalls = []
            weighted_recalls = []
            
            for model_name, result in self.evaluation_results.items():
                if 'error' not in result:  # 只包含成功評估的模型
                    # 清理模型名稱用於顯示
                    display_name = model_name.replace('.pth', '')
                    models.append(display_name)
                    accuracies.append(result.get('accuracy', 0.0))
                    macro_recalls.append(result.get('macro_recall', 0.0))
                    weighted_recalls.append(result.get('weighted_recall', 0.0))
            
            if not models:
                print("⚠️ 沒有成功評估的模型，無法生成比較圖表")
                return
            
            # 創建子圖
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
            
            # 1. 準確率比較（柱狀圖）
            axes[0, 0].bar(models, accuracies, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('Accuracy Comparison')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 添加數值標籤
            for i, v in enumerate(accuracies):
                axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            # 2. Macro Recall比較（柱狀圖）
            bars = axes[0, 1].bar(models, macro_recalls, color='lightgreen', alpha=0.7)
            axes[0, 1].set_title('Macro Recall Comparison')
            axes[0, 1].set_ylabel('Macro Recall')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 突出顯示最佳模型
            if self.best_recall_model:
                best_idx = None
                for i, model in enumerate(models):
                    if self.best_recall_model.replace('.pth', '') == model:
                        best_idx = i
                        break
                if best_idx is not None:
                    bars[best_idx].set_color('gold')
                    bars[best_idx].set_alpha(1.0)
            
            # 添加數值標籤
            for i, v in enumerate(macro_recalls):
                axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            # 3. Weighted Recall比較（柱狀圖）
            axes[1, 0].bar(models, weighted_recalls, color='lightcoral', alpha=0.7)
            axes[1, 0].set_title('Weighted Recall Comparison')
            axes[1, 0].set_ylabel('Weighted Recall')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 添加數值標籤
            for i, v in enumerate(weighted_recalls):
                axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            # 4. 雷達圖比較（所有指標）
            if len(models) <= 5:  # 只在模型數量不多時顯示雷達圖
                ax = axes[1, 1]
                
                # 準備雷達圖數據
                metrics = ['Accuracy', 'Macro Recall', 'Weighted Recall']
                angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
                angles += angles[:1]  # 閉合圖形
                
                ax = plt.subplot(2, 2, 4, projection='polar')
                
                for i, model in enumerate(models):
                    values = [accuracies[i], macro_recalls[i], weighted_recalls[i]]
                    values += values[:1]  # 閉合圖形
                    
                    color = plt.cm.tab10(i)
                    ax.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
                    ax.fill(angles, values, alpha=0.1, color=color)
                
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(metrics)
                ax.set_ylim(0, 1)
                ax.set_title('Performance Radar Chart')
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            else:
                # 如果模型太多，顯示簡單的線圖
                ax = axes[1, 1]
                x_pos = range(len(models))
                ax.plot(x_pos, accuracies, 'o-', label='Accuracy', linewidth=2)
                ax.plot(x_pos, macro_recalls, 's-', label='Macro Recall', linewidth=2)
                ax.plot(x_pos, weighted_recalls, '^-', label='Weighted Recall', linewidth=2)
                ax.set_title('Performance Trends')
                ax.set_ylabel('Score')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(models, rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存比較圖表
            comparison_file = self.images_dir / "model_comparison_chart.png"
            plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
            print(f"🖼️ 模型比較圖表已保存: {comparison_file}")
            plt.close()
            
        except Exception as e:
            print(f"❌ 無法創建模型比較圖表: {e}")
            import traceback
            traceback.print_exc()
    
    def evaluate_all_models(self, recall_metric: str = 'macro_recall') -> Dict[str, Dict]:
        """
        評估所有模型
        
        Args:
            recall_metric (str): 使用的recall指標 ('macro_recall' 或 'weighted_recall')
            
        Returns:
            Dict[str, Dict]: 包含所有模型評估結果的字典
        """
        print(f"\n🚀 開始評估所有模型（使用 {recall_metric} 作為比較指標）...")
        
        model_files = self.get_model_files()
        
        # 重置結果
        self.evaluation_results = {}
        self.best_recall_model = None
        self.best_recall_score = 0.0
        
        # 評估每個模型
        for model_file in model_files:
            result = self.evaluate_single_model(model_file)
            self.evaluation_results[model_file] = result
            
            # 更新最佳recall模型
            current_recall = result.get(recall_metric, 0.0)
            if current_recall > self.best_recall_score:
                self.best_recall_score = current_recall
                self.best_recall_model = model_file
        
        print(f"\n🎯 評估完成！")
        print(f"最佳模型: {self.best_recall_model}")
        print(f"最佳 {recall_metric}: {self.best_recall_score:.4f}")
        
        # 生成模型比較圖表
        self.generate_model_comparison_chart()
        
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
            print("⚠️ 尚未進行模型評估，請先執行 evaluate_all_models()")
            return pd.DataFrame()
        
        summary_data = []
        for model_name, result in self.evaluation_results.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': result.get('accuracy', 0.0),
                'Macro Recall': result.get('macro_recall', 0.0),
                'Weighted Recall': result.get('weighted_recall', 0.0),
                'Error': result.get('error', '')
            })
        
        df = pd.DataFrame(summary_data)
        # 按照Macro Recall降序排列
        df = df.sort_values('Macro Recall', ascending=False).reset_index(drop=True)
        
        return df
    
    def save_results(self, output_file: str = 'model_evaluation_results.csv'):
        """
        將評估結果保存到CSV檔案
        
        Args:
            output_file (str): 輸出檔案名稱
        """
        summary_df = self.get_results_summary()
        summary_df.to_csv(output_file, index=False)
        print(f"📊 評估結果已保存到: {output_file}")
    
    def print_detailed_results(self):
        """印出詳細的評估結果"""
        if not self.evaluation_results:
            print("⚠️ 尚未進行模型評估")
            return
        
        print("\n" + "="*80)
        print("📊 詳細評估結果")
        print("="*80)
        
        # 按照macro recall排序
        sorted_results = sorted(
            self.evaluation_results.items(),
            key=lambda x: x[1].get('macro_recall', 0.0),
            reverse=True
        )
        
        for i, (model_name, result) in enumerate(sorted_results, 1):
            print(f"\n{i}. 模型: {model_name}")
            print(f"   準確率: {result.get('accuracy', 0.0):.4f}")
            print(f"   Macro Recall: {result.get('macro_recall', 0.0):.4f}")
            print(f"   Weighted Recall: {result.get('weighted_recall', 0.0):.4f}")
            
            if 'individual_recalls' in result:
                print("   各類別Recall:")
                for class_name, recall_val in result['individual_recalls'].items():
                    print(f"     {class_name}: {recall_val:.4f}")
            
            if result.get('error'):
                print(f"   ❌ 錯誤: {result['error']}")
        
        print(f"\n🏆 最佳模型: {self.best_recall_model}")
        print(f"🎯 最佳Macro Recall: {self.best_recall_score:.4f}")


# 使用範例
if __name__ == "__main__":
    # 創建評估器實例
    evaluator = ModelEvaluator(
        models_dir="models",
        images_dir="images",
        use_smote=False,
        use_under_sampling=False
    )
    
    # 評估所有模型
    results = evaluator.evaluate_all_models(recall_metric='macro_recall')
    
    # 獲取最佳模型
    best_model, best_score = evaluator.get_best_model()
    print(f"\n🏆 最佳模型: {best_model}")
    print(f"🎯 最佳Recall分數: {best_score:.4f}")
    
    # 顯示結果摘要
    print("\n📊 結果摘要:")
    summary_df = evaluator.get_results_summary()
    print(summary_df.to_string(index=False))
    
    # 保存結果
    evaluator.save_results('model_evaluation_results.csv')
    
    # 顯示詳細結果
    evaluator.print_detailed_results()
