from data_collection import DataCollector
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
from model_deployment import ModelDeployer
from data_preparation import DataPreparer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import json

# 完整的机器学习流程示例
class MLProjectPipeline:
    def __init__(self):
        self.data_collector = DataCollector()
        self.data_preparer = None
        self.model_trainer = ModelTrainer()
        self.model_evaluator = ModelEvaluator()
        self.model_deployer = ModelDeployer()
    
    def run_complete_pipeline(self, target_column):
        """运行完整的机器学习流水线"""
        print("=" * 60)
        print("机器学习项目完整流程")
        print("=" * 60)
        
        # 1. 数据收集
        print("\n第1步：数据收集")
        print("-" * 30)
        user_data = self.data_collector.collect_user_data(1000)
        behavior_data = self.data_collector.collect_behavior_data(5000)
        
        # 合并数据（简化示例）
        merged_data = pd.merge(user_data, behavior_data, on='user_id', how='inner')
        # print(f"合并后的数据集: {merged_data.head(25)}")
        
        # 创建目标变量（示例：是否购买）
        merged_data[target_column] = (merged_data['behavior_type'] == '购买').astype(int)
        # print(f"目标变量 '{target_column}' 已创建，数据集示例：\n{merged_data[[target_column]].head(25)}")
        
        # 2. 数据准备
        print("\n第2步：数据准备")
        print("-" * 30)
        
        # 选择特征列
        feature_columns = ['age', 'gender', 'city', 'duration']
        if all(col in merged_data.columns for col in feature_columns):
            data_for_ml = merged_data[feature_columns + [target_column]].copy()
            # print(f"用于机器学习的数据集示例：\n{data_for_ml.head(25)}")
            
            # 处理类别变量
            # data_for_ml['gender'] = data_for_ml['gender'].map({'男': 0, '女': 1}).astype(int)
            # data_for_ml['city'] = data_for_ml['city'].map({'北京': 0, '上海': 1, '广州': 2, '深圳': 3}).astype(int)
            # print(f"类别变量已编码，数据集示例：\n{data_for_ml.head(25)}")
            
            # 数据准备
            self.data_preparer = DataPreparer(data_for_ml)
            splits, encoders, scaler = self.data_preparer.prepare_pipeline(target_column)

            # print(f"训练集示例：\n{splits['X_train'].head(25)}")
            # print(f"验证集示例：\n{splits['X_val'].head(25)}")
            # print(f"测试集示例：\n{splits['X_test'].head(25)}")
            # print(f"训练集标签示例：\n{splits['y_train'].head(25)}")
            # print(f"验证集标签示例：\n{splits['y_val'].head(25)}")
            # print(f"测试集标签示例：\n{splits['y_test'].head(25)}")
            
            # 3. 模型训练
            print("\n第3步：模型训练")
            print("-" * 30)
            
            # 注册模型
            self.model_trainer.register_model(
                '逻辑回归', LogisticRegression(random_state=42), 'classification'
            )
            self.model_trainer.register_model(
                '随机森林', RandomForestClassifier(n_estimators=100, random_state=42), 'classification'
            )
            
            # 训练模型
            trained_models = self.model_trainer.train_all_models(
                splits['X_train'], splits['y_train']
            )
            
            # 4. 模型评估
            print("\n第4步：模型评估")
            print("-" * 30)
            
            results = self.model_trainer.evaluate_models(
                splits['X_test'], splits['y_test']
            )
            # print(f"\n模型评估结果汇总：\n{json.dumps(results, indent=2, ensure_ascii=False)}")
            
            best_name, best_model = self.model_trainer.get_best_model(results)

            # print(f"\n最佳模型：{best_name}，模型对象：{best_model}")
            
            # 5. 模型部署
            print("\n第5步：模型部署")
            print("-" * 30)
            
            # 保存模型
            model_path = self.model_deployer.save_model(best_model, "production_model")

            # 加载模型
            self.model_deployer.load_model("production_model", model_path)
            
            # 创建预测服务
            prediction_service = self.model_deployer.create_prediction_service(
                "production_model"
            )
            
            # 测试预测服务
            test_input = splits['X_test'].head(10)
            prediction_result = prediction_service(test_input)
            
            print("\n预测服务测试结果：")
            print(json.dumps(prediction_result, indent=2, ensure_ascii=False))
            
            print("\n" + "=" * 60)
            print("机器学习项目流程完成！")
            print("=" * 60)
            
            return {
                'data': data_for_ml,
                'splits': splits,
                'best_model': best_model,
                'best_model_name': best_name,
                'evaluation_results': results,
                'prediction_service': prediction_service
            }
        
        else:
            print("数据列不完整，无法继续流程")
            return None

# 运行完整流程
pipeline = MLProjectPipeline()
project_results = pipeline.run_complete_pipeline('purchased')
print(project_results['best_model_name'])
print(project_results['evaluation_results'])
print(project_results['data'].head(25))