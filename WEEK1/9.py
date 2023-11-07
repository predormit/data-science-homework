nums = input("请输入多个整数，用空格隔开：").split()
for i in range(len(nums)):
    nums[i] = int(nums[i])
nums.sort()
for i in range(len(nums)-1,-1,-1):
    print(nums[i])
i = len(nums) - 1
while(i >= 0):
    print(nums[i])
    i = i - 1