from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 导入数据集
iris = load_iris()

# 获取特征和标签
X = iris.data  # 特征
y = iris.target  # 标签

# 随机划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用KNN分类器进行训练和预测
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# 在训练集上测试准确度
train_accuracy = knn.score(X_train, y_train)
print(f"训练集准确度：{train_accuracy:.4f}")

# 在测试集上测试准确度
test_accuracy = knn.score(X_test, y_test)
print(f"测试集准确度：{test_accuracy:.4f}")
