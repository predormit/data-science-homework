def cube(s):
    sum = 1
    for i in range(1,s+1):
        sum = sum*i
    return sum
s = int(input("请输入数字:"))
print(cube(s))