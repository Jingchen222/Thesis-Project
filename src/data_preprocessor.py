# src/data_preprocessor.py

import numpy as np
import xarray as xr
import pandas as pd

class DataProcessor:
    def __init__(self, lead_time=28):
        """
        初始化数据预处理器。

        Args:
            lead_time (int): 预报提前期（天），默认为28天。
        """
        self.lead_time = lead_time
        # 定义需要的变量，已排除 'sst' 和压力层变量
        self.basic_features = ['t2m', 'msl', 'u10', 'v10', 'tcwv', 'blh', 'sp', 'tp']
        self.derived_features = ['wind_speed', 'temp_advection']
    
    def load_dataset(self, file_path):
        """
        加载ERA5数据集。

        Args:
            file_path (str): 数据集文件路径。

        Returns:
            xarray.Dataset: 加载后的数据集。
        """
        print(f"加载数据集: {file_path}")
        ds = xr.open_dataset(file_path)
        return ds
    
    def load_and_check(self, ds):
        """
        检查数据集中的基本变量，并处理缺失值。

        Args:
            ds (xarray.Dataset): 加载后的数据集。

        Returns:
            xarray.Dataset: 处理后的数据集。
        """
        # 检查基本特征是否存在
        missing_vars = [var for var in self.basic_features if var not in ds.data_vars]
        if missing_vars:
            raise ValueError(f"缺失以下基本特征变量: {missing_vars}")
        
        print("所有必要的基本变量均存在。")
        
        # 处理缺失值（填充缺失值）
        print("处理缺失值...")
        for var in self.basic_features:
            missing_count = ds[var].isnull().sum().item()
            if missing_count > 0:
                # 使用 valid_time 维度计算均值
                mean_val = ds[var].mean(dim='valid_time', skipna=True)
                ds[var] = ds[var].fillna(mean_val)
                print(f"变量 '{var}' 存在 {missing_count} 个缺失值，已用均值填充。")
            else:
                print(f"变量 '{var}' 无缺失值。")
        
        return ds
    
    def create_features(self, ds):
        """
        创建预测特征，包括基础特征和衍生特征。

        Args:
            ds (xarray.Dataset): 处理后的数据集。

        Returns:
            xarray.Dataset: 包含所有特征的数据集。
        """
        print("创建基础特征...")
        features = ds[self.basic_features].copy()
        
        print("计算衍生特征: wind_speed")
        features['wind_speed'] = np.sqrt(ds['u10']**2 + ds['v10']**2)
        
        # 暂不计算 temp_advection，需实现温度梯度计算
        # features['temp_advection'] = -ds['u10'] * self.dT_dx(ds) - ds['v10'] * self.dT_dy(ds)
        
        print("添加时间特征...")
        # 使用 valid_time 维度提取时间信息
        features = features.assign_coords(
            month=ds['valid_time.month'],
            season=ds['valid_time'].dt.season,
            year=ds['valid_time.year']
        )
        
        return features
    
    def create_targets(self, ds):
        """
        创建预测目标：2米温度距平。

        Args:
            ds (xarray.Dataset): 处理后的数据集。

        Returns:
            xarray.DataArray: 2米温度距平。
        """
        print("创建2米温度距平作为预测目标...")
        t2m = ds['t2m']
        # 使用 valid_time 作为时间维度进行分组
        climatology = t2m.groupby('valid_time.month').mean('valid_time')
        anomalies = t2m.groupby('valid_time.month') - climatology
        return anomalies
    
    def dT_dx(self, ds):
        # 温度梯度（x方向）的示例计算
        dx = ds['longitude'].diff('longitude').mean().values
        dT_dx = ds['t2m'].diff('longitude') / dx
        return dT_dx
    
    def dT_dy(self, ds):
        # 温度梯度（y方向）的示例计算
        dy = ds['latitude'].diff('latitude').mean().values
        dT_dy = ds['t2m'].diff('latitude') / dy
        return dT_dy
