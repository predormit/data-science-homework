nums = input("请输入多个整数，用空格隔开：").split()
for i in range(len(nums)):
    nums[i] = int(nums[i])
nums.sort()
print( nums)