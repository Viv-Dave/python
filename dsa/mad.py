n = int(input(""))
# n = 7
arr = []
for i in range(1, n+1):
    arr.append(i)

while len(arr) > 0:
    move_to_back = arr.pop(0)
    arr.append(move_to_back)
    
    print(arr.pop(0))