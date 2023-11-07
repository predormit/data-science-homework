import numpy as np

def add_noise(func, num_samples, mu=0, sigma=1):
# 生成样本点
    x = np.linspace(0, 1, num_samples)
    y = func(x)


    # 生成标准正态分布噪声
    noise = np.random.normal(mu, sigma, num_samples)
    print(noise)
    # 加噪声生成新样本点
    new_y = y + noise

    return new_y

def f(x):
    return 9 * x + 8


num_samples = 30
samples = add_noise(f, num_samples)


print(samples)