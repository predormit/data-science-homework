s = input("请输入字符串:")
length = len(s)
j = 0
for i in range(1,length):
    if(s[i]==s[i-1]):
        j = 1
        break
if(j):
    print("yes")
else:
    print("no")

