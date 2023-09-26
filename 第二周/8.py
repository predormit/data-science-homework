import random
import math
def square_root_4(times):
    sum = 0
    for i in range(times):
        x = random.uniform(0,1)
        y = random.uniform(0,1)
        d = pow(x,2) + pow(y,2)
        if d <= 1:
            sum += 1
    return 4 * (sum / times)
import math

def pi2():
    pi = 0.0
    denominator = 1
    sign = 1

    for i in range(1000000):
        pi += sign * (1 / denominator)
        denominator += 2
        sign *= -1

    return 4 * pi
def pi3():
    n = 6
    a = 1
    for i in range(14):
        n = 2 * n
        a = math.sqrt(2 - 2 * math.sqrt(1 - (a / 2) ** 2))
    return n * a / 2
x = square_root_4(10000000)
y = pi2()
z = pi3()
print(f"π ≈ {x:.10f}")
print(f"π ≈ {y:.10f}")
print(f"π ≈ {z:.10f}")