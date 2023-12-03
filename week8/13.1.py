import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 导入数据集
iris = load_iris()

# 获取特征和标签
X = iris.data  # 特征
y = iris.target  # 标签
target_names = iris.target_names  # 类别名称

# 可视化数据特征
plt.figure(figsize=(12, 9))

for i in range(len(iris.feature_names)):
    plt.subplot(2, 2, i+1)
    plt.scatter(X[:, i], y, c=y, cmap=plt.cm.Set1)
    plt.xlabel(iris.feature_names[i])
    plt.ylabel('target')
    plt.title(f'{iris.feature_names[i]} vs target')

plt.tight_layout()
plt.show()
