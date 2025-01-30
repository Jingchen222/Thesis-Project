# src/random_forest_model.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

def extract_features(features_ds, target_da, sequence_length, start_index, end_index):
    X_list, y_list = [], []
    for i in range(start_index, end_index - sequence_length):
        window = features_ds.isel(valid_time=slice(i, i + sequence_length)).values
        mean_features = np.mean(window, axis=0).flatten()
        std_features = np.std(window, axis=0).flatten()
        features = np.concatenate([mean_features, std_features])
        X_list.append(features)
        
        target_value = target_da.isel(valid_time=i + sequence_length)\
                                .mean(dim=['latitude', 'longitude'])\
                                .values
        y_list.append(float(target_value.item()))
    return np.array(X_list), np.array(y_list)

def train_and_evaluate_rf(features_combined, targets_da, sequence_length, train_start, train_end):
    # 提取特征
    X_full, y_full = extract_features(features_combined, targets_da, sequence_length, train_start, train_end)
    
    # 划分训练和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
    
    # 训练模型
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # 评估模型
    y_pred = rf_model.predict(X_test)
    r2_score = rf_model.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    print("随机森林测试集R²得分:", r2_score)
    print("随机森林测试集均方误差 (MSE):", mse)
    
    # 可选：保存模型
    model_save_path = "/content/drive/MyDrive/weather_forecast_project/rf_model.pkl"
    joblib.dump(rf_model, model_save_path)
    print(f"随机森林模型已保存到: {model_save_path}")
