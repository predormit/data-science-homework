def insertion_sort(lst):
    for i in range( len(lst)):
        preIndex=i-1
        current=lst[i]
        while preIndex>=0 and lst[preIndex]>current:
            lst[preIndex+1]=lst[preIndex]
            preIndex-=1
        lst[preIndex+1]=current
    return lst
list1 = [6,5,3,1,8,7,2,4]
list1 = insertion_sort(list1)
print(f"排序结果：{list1}")