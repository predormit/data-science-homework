from sklearn.feature_extraction.text import CountVectorizer

# 准备样本文本
documents = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?'
]

# 创建CountVectorizer对象
vectorizer = CountVectorizer()

# 文本向量化
X = vectorizer.fit_transform(documents)

# 输出结果向量
result_vector = X.toarray()
print(result_vector)
