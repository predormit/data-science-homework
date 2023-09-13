x = int(input())
y = int(input())
z = int(input())
q = x + y + z
print(min(x,y,z))
q -= min(x,y,z)
q -= max(x,y,z)
print(q)
print(max(x,y,z))