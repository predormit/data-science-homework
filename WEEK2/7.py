def Cube_root_3(c):
    g = c
    i = 0
    while(abs(g*g*g-c)>0.00000000001):
        g = g - (g*g*g-c)/(3*g*g)
        i = i + 1
        print("%d:%.13f"%(i,g))
c = int(input("请输入数字: "))
Cube_root_3(c)