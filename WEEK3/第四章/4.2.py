import time
start = time.time()

for i in range(0,10000):
    print(i)

end = time.time()
print("时间=%s"%(end-start))