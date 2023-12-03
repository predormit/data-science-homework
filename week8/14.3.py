from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

# 加载数据集
data = fetch_20newsgroups(subset='all')

# 定义tf-idf向量化器
#vectorizer = TfidfVectorizer()

# 向量化数据集
#X = vectorizer.fit_transform(data.data)

# 获取第一个文本的结果向量
#first_document_vector = X[0].toarray()

# 输出结果向量
#print(first_document_vector)