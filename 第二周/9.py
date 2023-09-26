import random
import math

def monte_carlo(f, a, b, num_samples):
    total = 0.0
    for _ in range(num_samples):
        x = random.uniform(a, b)
        total += f(x)
    return (b - a) * total / num_samples

def function(x):
    return x**2 + 4 * x * math.sin(x)

a = 2
b = 3
num_samples = 100000

integral = monte_carlo(function, a, b, num_samples)
print("定积分的近似值：", integral)