from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

# 训练集文本和对应的标签
train_texts = ["good movie", "not a good movie", "did not like", "i like it", "good one"]
train_labels = [1, 0, 0, 1, 1]

# 测试集文本和对应的标签
test_texts = ["good movie", "not good"]
test_labels = [1, 0]

# 使用CountVectorizer将文本转换为词频向量
vectorizer = CountVectorizer(binary=True)
train_vectors = vectorizer.fit_transform(train_texts)
test_vectors = vectorizer.transform(test_texts)

# 使用BernoulliNB训练模型
classifier = BernoulliNB()
classifier.fit(train_vectors, train_labels)

# 在训练集上进行预测，并计算分类准确度
train_predictions = classifier.predict(train_vectors)
train_accuracy = accuracy_score(train_labels, train_predictions)
print("Train accuracy:", train_accuracy)

# 在测试集上进行预测，并计算分类准确度
test_predictions = classifier.predict(test_vectors)
test_accuracy = accuracy_score(test_labels, test_predictions)
print("Test accuracy:", test_accuracy)