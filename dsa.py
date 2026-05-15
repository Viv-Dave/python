temperatures = [73,74,75,71,69,72,76,73]

n = len(temperatures)
days = [0] 
stack = []         

i = 0

while i < n:
    while len(stack) > 0 and temperatures[i] > temperatures[stack[-1]]:
        prev = stack.pop()
        days[prev] = i - prev

    stack.append(i)  
    i += 1

print(days)