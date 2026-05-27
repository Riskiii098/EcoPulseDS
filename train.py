import sys
import io
# Mencegah crash encoding emoji dari library mlflow di Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import pandas as pd
import numpy as np # pyright: ignore [missing-import]
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor  # pyright: ignore [missing-import]
import mlflow  # pyright: ignore [missing-import]
import mlflow.sklearn  # pyright: ignore [missing-import]

warnings.filterwarnings('ignore')

# =======================================================
# 1. KONFIGURASI MLFLOW
# =======================================================
# Set DagsHub Credentials
os.environ["MLFLOW_TRACKING_USERNAME"] = "Riskiii098"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "0daeefe4719977be945bce17d986e8466a661396"

# Setup Remote MLflow Tracking di DagsHub
dagshub_uri = "https://dagshub.com/Riskiii098/EcoPulseDS.mlflow"
mlflow.set_tracking_uri(dagshub_uri)
mlflow.set_experiment("EcoPulse_Poverty_Prediction")

def evaluate_metrics(y_true, y_pred):
    """Menghitung evaluasi performa regresi"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def main():
    print("="*50)
    print("[INFO] MEMULAI PIPELINE MACHINE LEARNING")
    print("="*50)

    # =======================================================
    # 2. PEMUATAN DATA & TRAIN-TEST SPLIT
    # =======================================================
    data_path = 'data/processed/poverty_macro_clean.csv'
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data tidak ditemukan di {data_path}. Pastikan notebook preprocessing sudah dijalankan.")
        
    df = pd.read_csv(data_path)
    print(f"[1] Memuat dataset '{data_path}' dengan {df.shape[0]} baris.")

    # Menangani NaN jika masih ada sisa (sebagai tindakan pencegahan)
    df.dropna(subset=['Laju_PDRB', 'TPT', 'GK_Urban', 'GK_Rural'], inplace=True)
    
    # Memisahkan Fitur (X) dan Target (y)
    # Target bersifat multi-output: memprediksi kemiskinan kota dan desa sekaligus
    features = ['Tahun', 'Laju_PDRB', 'TPT']
    targets = ['GK_Urban', 'GK_Rural']
    
    X = df[features]
    y = df[targets]
    
    # Train-Test Split (80% Latih, 20% Uji)
    # Karena jumlah data provinsi terbatas per tahun, pengacakan stratified berbasis tahun direkomendasikan
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=X['Tahun'])
    
    print(f"[2] Train-Test Split selesai: {X_train.shape[0]} Latih | {X_test.shape[0]} Uji.")

    # =======================================================
    # 3. TRAINING BASELINE MODEL (RIDGE REGRESSION)
    # =======================================================
    print("\n[3] Melatih Baseline Model (Ridge Regression)...")
    
    # Memulai session run MLflow untuk Baseline
    with mlflow.start_run(run_name="Baseline_PolyRidgeCV"):
        # Pipeline scikit-learn: Poly -> Scaling -> Modeling
        # PolynomialFeatures (degree=3) membantu menangkap pola non-linear 
        lr_pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=3, include_bias=False)),
            ('scaler', StandardScaler()),
            ('model', MultiOutputRegressor(RidgeCV(alphas=np.logspace(-2, 3, 50))))
        ])
        
        lr_pipeline.fit(X_train, y_train)
        
        # Evaluasi
        y_train_pred_lr = lr_pipeline.predict(X_train)
        mae_train_lr, rmse_train_lr, r2_train_lr = evaluate_metrics(y_train, y_train_pred_lr)
        
        y_test_pred_lr = lr_pipeline.predict(X_test)
        mae_test_lr, rmse_test_lr, r2_test_lr = evaluate_metrics(y_test, y_test_pred_lr)
        
        # Ekstrak alpha terbaik 
        best_alphas = [est.alpha_ for est in lr_pipeline.named_steps['model'].estimators_]
        
        # Logging ke MLflow
        mlflow.log_param("model_type", "MultiOutput-PolyRidgeCV")
        mlflow.log_param("scaler", "StandardScaler")
        mlflow.log_param("poly_degree", 3)
        mlflow.log_param("best_alphas", str(best_alphas))
        
        mlflow.log_metrics({
            "train_mae": mae_train_lr, "train_rmse": rmse_train_lr, "train_r2": r2_train_lr,
            "test_mae": mae_test_lr, "test_rmse": rmse_test_lr, "test_r2": r2_test_lr
        })
        
        # Simpan artefak model
        mlflow.sklearn.log_model(lr_pipeline, "baseline_model")
        
        print(f"    -> [PolyRidge] Train R2: {r2_train_lr:.4f} | Test R2: {r2_test_lr:.4f}")
        print(f"    -> [PolyRidge] Train RMSE: {rmse_train_lr:.2f} | Test RMSE: {rmse_test_lr:.2f}")

    # =======================================================
    # 4. TRAINING ADVANCED MODEL (XGBOOST REGRESSOR)
    # =======================================================
    print("\n[4] Melatih Advanced Model (Tuned XGBoost)...")
    
    with mlflow.start_run(run_name="Advanced_TunedXGBoost"):
        # XGBoost dibungkus MultiOutputRegressor
        xgb_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', MultiOutputRegressor(XGBRegressor(random_state=42, objective='reg:squarederror')))
        ])
        
        # Parameter grid untuk GridSearchCV
        param_grid = {
            'model__estimator__n_estimators': [50, 100],
            'model__estimator__max_depth': [3, 5],
            'model__estimator__learning_rate': [0.05, 0.1],
            'model__estimator__subsample': [0.7, 0.9],
            'model__estimator__colsample_bytree': [0.8, 1.0]
        }
        
        print("    -> Sedang mencari parameter terbaik (GridSearchCV), harap tunggu sebentar...")
        grid_search = GridSearchCV(xgb_pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)
        
        best_pipeline = grid_search.best_estimator_
        
        # Evaluasi
        y_train_pred_xgb = best_pipeline.predict(X_train)
        mae_train_xgb, rmse_train_xgb, r2_train_xgb = evaluate_metrics(y_train, y_train_pred_xgb)
        
        y_test_pred_xgb = best_pipeline.predict(X_test)
        mae_test_xgb, rmse_test_xgb, r2_test_xgb = evaluate_metrics(y_test, y_test_pred_xgb)
        
        # Logging parameter
        mlflow.log_param("model_type", "MultiOutput-TunedXGBoost")
        mlflow.log_param("scaler", "StandardScaler")
        # Simpan parameter terbaik ke MLflow
        for key, value in grid_search.best_params_.items():
            mlflow.log_param(key.replace('model__estimator__', ''), value)
        
        # Logging metrics
        mlflow.log_metrics({
            "train_mae": mae_train_xgb, "train_rmse": rmse_train_xgb, "train_r2": r2_train_xgb,
            "test_mae": mae_test_xgb, "test_rmse": rmse_test_xgb, "test_r2": r2_test_xgb
        })
        
        # Simpan artefak model XGBoost
        mlflow.sklearn.log_model(best_pipeline, "advanced_model")
        
        print(f"    -> [Tuned XGBoost] Train R2: {r2_train_xgb:.4f} | Test R2: {r2_test_xgb:.4f}")
        print(f"    -> [Tuned XGBoost] Train RMSE: {rmse_train_xgb:.2f} | Test RMSE: {rmse_test_xgb:.2f}")

    print("="*50)
    print("[INFO] TRAINING SELESAI")
    print("Semua artefak model, parameter, dan metrik telah dicatat di MLflow.")
    print("Untuk melihat dashboard UI MLflow, jalankan perintah:")
    print("mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")
    print("="*50)

if __name__ == "__main__":
    main()
