# src/model_architecture.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LSTM, Dense, TimeDistributed, Flatten, InputLayer

class ForecastModel:
    def __init__(self):
        pass

    def build(self, input_shape):
        """
        构建并编译预测模型。

        Args:
            input_shape (tuple): 输入数据的形状，例如 (时间步长, 高度, 宽度, 通道数)

        Returns:
            tensorflow.keras.Model: 编译后的模型
        """
        model = Sequential()
        
        # 输入层
        model.add(InputLayer(input_shape=input_shape))
        
        # 1. 使用TimeDistributed封装CNN以提取每个时间步的空间特征
        model.add(TimeDistributed(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')))
        model.add(TimeDistributed(Flatten()))
        
        # 2. 使用LSTM处理时序特征
        model.add(LSTM(64, return_sequences=False))
        
        # 3. 最后一层Dense用于输出预测
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mse')
        return model
