def Square_root_1():
    i = 0
    c = 2
    g = 0
    for j in range(0,c+1):
        if(j * j > c and g == 0):
            g = j - 1
    while(abs(g * g - c) > 0.0001):
        g += 0.00001
        i = i + 1
        print(i,g)
        if(g * g == c):
            break
    print(g)
Square_root_1()