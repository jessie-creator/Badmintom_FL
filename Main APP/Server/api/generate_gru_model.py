import torch
import torch.nn as nn
import os

class CustomFunc:
    @staticmethod
    def custom_output(x):
        # 定義你自己的 custom_output 函式
        return x

class CustomGRUModel(nn.Module):
    def __init__(self, input_size, units, output_dim, num_layers, dropout_rate):
        super(CustomGRUModel, self).__init__()
        self.gru_layers = nn.ModuleList(
                [nn.GRU(input_size if i == 0 else units, units, batch_first=True) for i in range(num_layers)]
                )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.dense = nn.Linear(units, output_dim)

    def forward(self, x):
        for gru_layer in self.gru_layers:
            x, _ = gru_layer(x)
            x = self.dropout(x) # Apply dropout after each GRU layer
        # Use the output of the last GRU layer (x) for prediction
        x = self.dense(x[:, -1, :]) # Selecting the output from the last time step for prediction
        return CustomFunc.custom_output(x)

# 建立五個模型並儲存
num_models = 5
input_size = 10  # 替換為你想要的輸入維度
units = 20  # 替換為你想要的單位數
output_dim = 5  # 替換為你想要的輸出維度
num_layers = 2  # 替換為你想要的層數
dropout_rate = 0.2  # 替換為你想要的 dropout rate

model_directory = "model"
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

for i in range(num_models):
    model = CustomGRUModel(input_size, units, output_dim, num_layers, dropout_rate)
    model_path = os.path.join(model_directory, f"model{i + 1}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"模型 {model_path} 已儲存")
