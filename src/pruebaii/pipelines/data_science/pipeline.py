from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    split_data, 
    train_supervised_models, evaluate_supervised,
    train_unsupervised_models, evaluate_unsupervised,
    train_rl_models, evaluate_rl_models
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # Split Data
            node(
                func=split_data,
                inputs="preprocessed_data",
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            
            # 1. Supervised
            node(
                func=train_supervised_models,
                inputs=["X_train", "y_train"],
                outputs=["model_supervised_rf", "model_supervised_xgb", "model_supervised_svm", "model_supervised_lr", "model_supervised_knn", "model_supervised_tpot"],
                name="train_supervised_models_node",
            ),
            node(
                func=evaluate_supervised,
                inputs=["model_supervised_rf", "model_supervised_xgb", "model_supervised_svm", "model_supervised_lr", "model_supervised_knn", "model_supervised_tpot", "X_test", "y_test"],
                outputs="supervised_metrics",
                name="evaluate_supervised_node",
            ),
            
            # 2. Unsupervised
            node(
                func=train_unsupervised_models,
                inputs=["X_train"],
                outputs=["model_unsupervised_kmeans", "model_unsupervised_dbscan", "model_unsupervised_agg", "model_unsupervised_gmm", "model_unsupervised_iso", "preprocessor_unsup"],
                name="train_unsupervised_models_node",
            ),
            node(
                func=evaluate_unsupervised,
                inputs=["model_unsupervised_kmeans", "model_unsupervised_dbscan", "model_unsupervised_agg", "model_unsupervised_gmm", "model_unsupervised_iso", "preprocessor_unsup", "X_test"],
                outputs="unsupervised_metrics",
                name="evaluate_unsupervised_node",
            ),
            
            # 3. Reinforcement Learning
            node(
                func=train_rl_models,
                inputs=["X_train", "y_train"],
                outputs=["model_rl_ppo", "model_rl_a2c", "model_rl_dqn", "model_rl_sac", "model_rl_ddpg", "preprocessor_rl"],
                name="train_rl_models_node",
            ),
            node(
                func=evaluate_rl_models,
                inputs=["model_rl_ppo", "model_rl_a2c", "model_rl_dqn", "model_rl_sac", "model_rl_ddpg", "preprocessor_rl", "X_test", "y_test"],
                outputs="rl_metrics",
                name="evaluate_rl_models_node",
            ),
        ]
    )
