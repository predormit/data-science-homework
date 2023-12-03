from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 导入数据集
iris = load_iris()

# 获取特征和标签
X = iris.data  # 特征
y = iris.target  # 标签

# 随机划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 查看切分后的数据集大小
print("训练集大小:", X_train.shape[0])
print("测试集大小:", X_test.shape[0])
