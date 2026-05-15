# array = [1, 2, 3, 2, 1]
# nums = [3,2,3]
nums = [1,2,3,1]
hash_array = {}

for num in nums:
    if num in hash_array:
        hash_array[num] += 1
    else:
        hash_array[num] = 1

print(hash_array)
for num, count in hash_array.items():
    if count == 1:
        print("True")
    else:
        print("False")