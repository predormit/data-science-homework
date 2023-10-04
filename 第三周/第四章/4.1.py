import math
def prime(x):
    q = True
    for i in range(2,int(math.sqrt(x))+1):
        if x % i == 0:
            print("不是质数")
            q = False
            break
    if q:
        print("质数")
prime(12)
prime(11)