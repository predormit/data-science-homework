import random
import time

def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    l, r = 0, 0
    while l < len(left) and r < len(right):
        if left[l] <= right[r]:
            result.append(left[l])
            l += 1
        else:
            result.append(right[r])
            r += 1
    result.extend(left[l:])
    result.extend(right[r:])
    return result

# 生成多组长度递增的随机数列
num_lists = []
for i in range(1, 6):
    arr = [random.randint(1, 100) for _ in range(i * 1000)]
    num_lists.append(arr)

# 对每个数列进行排序并计算运行时间
for i, arr in enumerate(num_lists):
    print(f"Sort {i+1}:")

    # 选择排序
    start_time = time.time()
    selection_sort(arr)
    end_time = time.time()
    print(f"Selection Sort time: {end_time - start_time} seconds")

    # 归并排序
    start_time = time.time()
    sorted_arr = merge_sort(arr)
    end_time = time.time()
    print(f"Merge Sort time: {end_time - start_time} seconds")

    #print()