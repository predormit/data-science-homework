
def Statistic(file):
    f = open(file)
    dictionary = {}
    for line in f.readlines():
        #print(type(line))
        if len(line) > 10:
            mark = [',','.',':','\'s',';','?','(',')']
            for m in mark:
                line = line.replace(m,'')
            #print(line)
            lineattr = line.strip().split(" ")
            for char in lineattr:
                if char not in dictionary:
                    dictionary[char] = 1
                else:
                    dictionary[char]+=1
    a = sorted(dictionary.items(),key=lambda x:x[1],reverse = True)
    #print(a)
    return a

def printWords(file,n):
    a = Statistic(file)
    for i in range(n):
        print(a[i])

file = "Shakespeare.txt"
n = 10
printWords(file,n)
