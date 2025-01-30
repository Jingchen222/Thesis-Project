# src/trainer.py

class ModelTrainer:
    def __init__(self, model, train_X, train_y, val_X, val_y):
        """
        初始化模型训练器。

        Args:
            model: 已构建的Keras模型。
            train_X, train_y: 训练集特征和目标。
            val_X, val_y: 验证集特征和目标。
        """
        self.model = model
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y
    
    def train(self, epochs=10, batch_size=32):
        """
        训练模型。

        Args:
            epochs (int): 训练轮数。
            batch_size (int): 批量大小。

        Returns:
            history: 训练过程记录
        """
        history = self.model.fit(
            self.train_X, self.train_y,
            validation_data=(self.val_X, self.val_y),
            epochs=epochs,
            batch_size=batch_size
        )
        return history
    
    def evaluate(self):
        """
        在验证集上评估模型性能。

        Returns:
            loss: 验证集损失值
        """
        loss = self.model.evaluate(self.val_X, self.val_y)
        print(f"验证集损失: {loss}")
        return loss
