import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score

# Supervisados
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from tpot import TPOTClassifier

# No Supervisados
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest

# RL
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C, DQN, SAC, DDPG

def split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    drop_cols = ['id_estudiante', 'nombre', 'rut', 'email', 'estado_matricula', 'target']
    X = data.drop(columns=[col for col in drop_cols if col in data.columns])
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

def get_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_features = ['año_ingreso', 'total_asignaturas', 'promedio_general', 'asistencia_general', 'evaluaciones_totales']
    categorical_features = ['carrera', 'sede']
    numeric_features = [col for col in numeric_features if col in X_train.columns]
    categorical_features = [col for col in categorical_features if col in X_train.columns]
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor

# --- 1. SUPERVISADOS (5 Modelos) ---
def train_supervised_models(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Pipeline, Pipeline, Pipeline, Pipeline, Pipeline, Pipeline]:
    preprocessor = get_preprocessor(X_train)
    rf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])
    xgb = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))])
    svm = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', SVC(random_state=42, probability=True))])
    lr = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(random_state=42, max_iter=1000))])
    knn = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', KNeighborsClassifier())])
    
    tpot_clf = TPOTClassifier(generations=1, population_size=5, random_state=42, verbose=0)
    tpot = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', tpot_clf)])
    
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    tpot.fit(X_train, y_train)
    
    tpot_final = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', tpot_clf.fitted_pipeline_)])
    
    return rf, xgb, svm, lr, knn, tpot_final

def evaluate_supervised(rf: Pipeline, xgb: Pipeline, svm: Pipeline, lr: Pipeline, knn: Pipeline, tpot: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    models = {'RandomForest': rf, 'XGBoost': xgb, 'SVM': svm, 'LogisticRegression': lr, 'KNN': knn, 'TPOT': tpot}
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'F1': f1_score(y_test, y_pred, average='macro', zero_division=0)
        })
    return pd.DataFrame(results)

# --- 2. NO SUPERVISADOS (5 Modelos) ---
def train_unsupervised_models(X_train: pd.DataFrame) -> Tuple[Any, Any, Any, Any, Any, ColumnTransformer]:
    preprocessor = get_preprocessor(X_train)
    X_processed = preprocessor.fit_transform(X_train)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    
    kmeans = KMeans(n_clusters=3, random_state=42).fit(X_processed)
    dbscan = DBSCAN(eps=0.5, min_samples=5).fit(X_processed)
    agg = AgglomerativeClustering(n_clusters=3).fit(X_processed)
    gmm = GaussianMixture(n_components=3, random_state=42).fit(X_processed)
    iso = IsolationForest(random_state=42, contamination=0.1).fit(X_processed)
    
    return kmeans, dbscan, agg, gmm, iso, preprocessor

def evaluate_unsupervised(kmeans: Any, dbscan: Any, agg: Any, gmm: Any, iso: Any, preprocessor: ColumnTransformer, X_test: pd.DataFrame) -> pd.DataFrame:
    models = {'KMeans': kmeans, 'DBSCAN': dbscan, 'Agglomerative': agg, 'GMM': gmm, 'IsolationForest': iso}
    X_processed = preprocessor.transform(X_test)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    
    results = []
    for name, model in models.items():
        if hasattr(model, 'predict'):
            preds = model.predict(X_processed)
        else:
            preds = model.fit_predict(X_processed)
        
        n_clusters = len(np.unique(preds))
        if 1 < n_clusters < len(preds):
            sil = silhouette_score(X_processed, preds)
        else:
            sil = -1
            
        results.append({
            'Model': name,
            'Silhouette_Score': sil,
            'N_Clusters_Found': n_clusters
        })
    return pd.DataFrame(results)

# --- 3. REINFORCEMENT LEARNING (5 Modelos) ---
class ClassificationEnv(gym.Env):
    def __init__(self, X, y):
        super(ClassificationEnv, self).__init__()
        self.X = np.array(X if not hasattr(X, "toarray") else X.toarray(), dtype=np.float32)
        self.y = np.array(y, dtype=np.int64)
        self.current_step = 0
        self.action_space = spaces.Discrete(max(2, len(np.unique(self.y))))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.X.shape[1],), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self.X[self.current_step], {}

    def step(self, action):
        reward = 1.0 if action == self.y[self.current_step] else -1.0
        self.current_step += 1
        terminated = bool(self.current_step >= len(self.X))
        obs = self.X[self.current_step] if not terminated else np.zeros(self.X.shape[1], dtype=np.float32)
        return obs, reward, terminated, False, {}

class RegressionEnv(gym.Env):
    def __init__(self, X, y):
        super(RegressionEnv, self).__init__()
        self.X = np.array(X if not hasattr(X, "toarray") else X.toarray(), dtype=np.float32)
        self.y = np.array(y, dtype=np.float32)
        self.current_step = 0
        self.action_space = spaces.Box(low=-100.0, high=100.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.X.shape[1],), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self.X[self.current_step], {}

    def step(self, action):
        target = self.y[self.current_step]
        pred = action[0]
        reward = -float((target - pred)**2)
        self.current_step += 1
        terminated = bool(self.current_step >= len(self.X))
        obs = self.X[self.current_step] if not terminated else np.zeros(self.X.shape[1], dtype=np.float32)
        return obs, reward, terminated, False, {}

def train_rl_models(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Any, Any, Any, Any, Any, ColumnTransformer]:
    preprocessor = get_preprocessor(X_train)
    X_processed = preprocessor.fit_transform(X_train)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    
    env_disc = ClassificationEnv(X_processed, y_train)
    env_cont = RegressionEnv(X_processed, y_train)
    
    ppo = PPO("MlpPolicy", env_disc, n_steps=128, n_epochs=2, batch_size=64, verbose=0)
    a2c = A2C("MlpPolicy", env_disc, n_steps=5, verbose=0)
    dqn = DQN("MlpPolicy", env_disc, learning_starts=10, target_update_interval=50, verbose=0)
    sac = SAC("MlpPolicy", env_cont, learning_starts=10, batch_size=64, verbose=0)
    ddpg = DDPG("MlpPolicy", env_cont, learning_starts=10, batch_size=64, verbose=0)
    
    for model in [ppo, a2c, dqn, sac, ddpg]:
        model.learn(total_timesteps=500)
        
    return ppo, a2c, dqn, sac, ddpg, preprocessor

def evaluate_rl_models(ppo: Any, a2c: Any, dqn: Any, sac: Any, ddpg: Any, preprocessor: ColumnTransformer, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    models = {'PPO': ppo, 'A2C': a2c, 'DQN': dqn, 'SAC': sac, 'DDPG': ddpg}
    X_processed = preprocessor.transform(X_test)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    
    env_disc = ClassificationEnv(X_processed, y_test)
    env_cont = RegressionEnv(X_processed, y_test)
    
    results = []
    for name, model in models.items():
        env = env_disc if name in ['PPO', 'A2C', 'DQN'] else env_cont
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            
        results.append({
            'Model': name,
            'Total_Test_Reward': total_reward,
            'Avg_Reward_per_step': total_reward / len(y_test)
        })
        
    return pd.DataFrame(results)
