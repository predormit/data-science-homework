def gcd(a,b):
    x = a % b
    while x != 0:
        a = b
        b = x
        x = a % b
    return b
a = int(input("请输入数字："))
b = int(input("请输入数字："))
if a < b:
    x = a
    a = b
    b = x
print(gcd(a,b))