from model_evaluator_final import ModelEvaluator

def demo():
    """
    ModelEvaluator 使用示例
    """
    print("="*60)
    print("        ModelEvaluator 使用示例")
    print("="*60)
    
    # 1. 創建評估器實例
    print("\n1. 創建評估器...")
    evaluator = ModelEvaluator(models_dir="models")
    
    # 2. 評估前三個模型（用於快速測試）
    print("\n2. 評估前3個模型進行測試...")
    results = evaluator.evaluate_all_models(max_models=3)
    
    # 3. 顯示結果摘要
    print("\n3. 顯示結果摘要...")
    evaluator.print_summary()
    
    # 4. 獲取最佳模型
    best_model, best_score = evaluator.get_best_model()
    print(f"\n4. 最佳模型資訊:")
    print(f"   模型名稱: {best_model}")
    print(f"   Macro Recall: {best_score:.4f}")
    
    # 5. 保存結果
    print("\n5. 保存評估結果...")
    evaluator.save_results('demo_results.csv')
    
    print("\n✅ 示例完成！")
    
    return evaluator

if __name__ == "__main__":
    evaluator = demo() 