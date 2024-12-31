# QM Method is diverse and can be applied to higher nummber of bits
seven_bits = ['0000', '0001', '0010','0011', '0100', '0101', '0110','0111']
decimal_values = [0,1,2,3,4,5,6,7]
# Comparing bits with decimal values
for i in range(len(seven_bits)):
    print(f"{seven_bits[i]} = {decimal_values[i]}")
# Minterms, Implement function to add minterms by choice
minterms = [0,1,3,4,5,7]
minterm_list = []
for i in range(len(minterms)):
    minterm_list.append(seven_bits[minterms[i]]);
print(f"Minterm list: {minterm_list}")
# To group minterms based on the number of true bits present
group = {0:[], 1:[], 2:[], 3:[]}
for i in range(len(minterm_list)):
    digit = '1' 
    count = minterm_list[i].count(digit)
    count_list = []
    count_list.append(count)
    for j in range(len(count_list)):
        if count == count_list[j]:
            group[count].append(minterm_list[i]) 
print(f"Group: {group}")
dissimilar = 'X'
# To keep track of grouping minterms
similar_grouping = {0: [], 1: [], 2: []}
# To keep track of which minterms were grouped
grouped_minterms = []
# Slice and Compare Strings for 1 and 2
similar = ''
dissimilar = 'X'
dissimilar_counter = 0
for i in range(4):
    if group[0][0][i] == group[3][0][i]:
        similar = similar+group[0][0][i]
    else:
        dissimilar_counter += 1
        similar=similar+dissimilar
        print(f"Dissimilarities: {dissimilar_counter}")
        if dissimilar_counter >=2:
            similar = ''
            break
similar_grouping[0].append(similar)
print(similar)
# Create a compare loop to iterate over the whole dictionary
    # Using print statements to simulate comparisons

# Use conditional operators to find change such as three similar and distinct in position
# Calculate the different elements first and setCount to them
# Use a count operator for count and increment to reach sucessive keys in number for comparison once the comparions are done
# Use a print statement for debugging