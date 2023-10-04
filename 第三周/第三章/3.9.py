A = [1,2,3,4,5,6,7,8,9]
B = A[:]
C = A[:]
D = A[:]

for i in range(1,len(A)):
    C[i] = C[i-1] * A[i]
for i in range(len(A) - 2,-1,-1):
    D[i] = D[i+1] * A[i]
for i in range(len(B)):
    if i == 0:
        B[i] = D[i + 1]
    elif i == len(B) - 1:
        B[i] = C[i - 1]
    else:
        B[i] = C[i-1] * D[i+1]
print(B)