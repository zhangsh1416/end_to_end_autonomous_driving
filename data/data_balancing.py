import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 步骤1: 读取数据
column_names = ['Image_path', 'throttle', 'brake', 'steering', 'velocity']
data = pd.read_csv('VehicleData.txt', sep=' ',names=column_names, header=None)
print(data.shape)
# 步骤2: 分离出steering angle不为0的数据和为0的数据
non_zero_steering = data[data['steering'] != 0]
zero_steering = data[data['steering'] == 0]

# 步骤3: 从steering angle为0的数据中随机选择同等数量的项
zero_steering_sample = zero_steering.sample(n=len(non_zero_steering))

# 步骤4: 合并数据
final_data = pd.concat([non_zero_steering, zero_steering_sample])

# 步骤5: 分为训练集和验证集
train_set, test_set = train_test_split(final_data, test_size=0.2)  # 以80%训练集，20%验证集的比例分割

# 可选：保存结果到文件
train_set.to_csv('train_set.csv', index=False)
test_set.to_csv('test_set.csv', index=False)

print("训练集和验证集已生成并保存。")
