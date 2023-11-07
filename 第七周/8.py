import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target

classes = np.unique(y)
centers = []

for c in classes:
    X_c = X[y == c]
    center_c = np.mean(X_c,axis=0)
    centers.append((center_c))

for i,center in enumerate(centers):
    print(f"类别 {i}: {center}")

distances = []

for i in range(len(X)):
    distances_c = [np.linalg.norm(X[i] - center) for center in centers]  # 计算欧式距离
    distances.append(distances_c)

print("数据点到中心点的欧式距离:")
for i, distance in enumerate(distances):
    print(f"点 {i}: {distance}")