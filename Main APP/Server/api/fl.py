import torch
import os

# 指定模型檔案所在的路徑
model_directory = "model_from_clients/high"  # 替換為你的模型檔案路徑
# 檢查目錄是否存在
if not os.path.exists(model_directory):
    print(f"目錄 {model_directory} 不存在")
    exit()
# 找出指定路徑下的所有模型檔案
model_files = [f for f in os.listdir(model_directory) if f.endswith('.pth')]
# 如果沒有找到模型檔案，則提示並退出
if not model_files:
    print(f"在 {model_directory} 中沒有找到任何模型檔案")
    exit()
# 初始化用於存儲所有模型權重的字典
all_weights = None
# 對每個模型檔案進行處理
for model_file in model_files:
    # 載入模型權重
    model_weights = torch.load(os.path.join(model_directory, model_file))

    # 如果是第一個模型，則將所有權重初始化為第一個模型的權重
    if all_weights is None:
        all_weights = model_weights.copy()
    else:
        # 對所有模型的權重進行累加
        for key in all_weights.keys():
            all_weights[key] += model_weights[key]
# 對所有模型的權重進行平均化
num_models = len(model_files)
avg_weights = {key: value / num_models for key, value in all_weights.items()}
# 儲存平均後的權重到新模型檔案
torch.save(avg_weights, 'uploaded_high_model.pth')
print("平均權重已儲存到 uploaded_high_model.pth")

# 指定模型檔案所在的路徑
model_directory = "model_from_clients/short"  # 替換為你的模型檔案路徑
# 檢查目錄是否存在
if not os.path.exists(model_directory):
    print(f"目錄 {model_directory} 不存在")
    exit()
# 找出指定路徑下的所有模型檔案
model_files = [f for f in os.listdir(model_directory) if f.endswith('.pth')]
# 如果沒有找到模型檔案，則提示並退出
if not model_files:
    print(f"在 {model_directory} 中沒有找到任何模型檔案")
    exit()
# 初始化用於存儲所有模型權重的字典
all_weights = None
# 對每個模型檔案進行處理
for model_file in model_files:
    # 載入模型權重
    model_weights = torch.load(os.path.join(model_directory, model_file))

    # 如果是第一個模型，則將所有權重初始化為第一個模型的權重
    if all_weights is None:
        all_weights = model_weights.copy()
    else:
        # 對所有模型的權重進行累加
        for key in all_weights.keys():
            all_weights[key] += model_weights[key]
# 對所有模型的權重進行平均化
num_models = len(model_files)
avg_weights = {key: value / num_models for key, value in all_weights.items()}
# 儲存平均後的權重到新模型檔案
torch.save(avg_weights, 'uploaded_short_model.pth')
print("平均權重已儲存到 uploaded_short_model.pth")
