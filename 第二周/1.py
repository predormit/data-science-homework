def max_product(n):
    # 创建一个列表，用于保存乘积最大的正整数列表
    dp = [0] * (n + 1)
    dp[0] = 1

    # 计算乘积最大的正整数列表
    for i in range(1, n + 1):
        for j in range(i):
            # 计算乘积
            product = dp[j] * (i-j)
            # 更新乘积最大值
            if product > dp[i]:
                dp[i] = product

    # 构建乘积最大的正整数列表
    result = []
    i = n
    while i > 0:
        for j in range(i):
            if dp[j] * (i-j) == dp[i]:
                result.append(i-j)
                i = j
                break

    # 返回乘积最大的正整数列表
    return result


n = 2001
result = max_product(n)
print("乘积最大的正整数列表：", result)
