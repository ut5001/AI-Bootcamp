# Empty list
list1 = []

list1 = ['mouse', [2, 4, 6],  ['a']]
print(list1)

# How to access elements in list
list2 = ['p', 'r', 'o', 'b', 'l', 'e', 'm']
print(list2[4])

list1 = ['mouse', [2, 4, 6],  ['a']]
print(list1[1][1])

# Slicing in a list
list2 = ['p', 'r', 'o', 'b', 'l', 'e', 'm']
print(list2[:-5])

# List is mutable !!!!
odd = [2, 4, 6, 8]
odd[0] = 1
print(odd)

odd[1:4] = [3, 5, 7]
print(odd)

# Append and extend can be also done in list
odd.append(9)
print(odd)
odd.extend([11, 13])
print(odd)

# Insert an element into a list
odd = [1, 9]
odd.insert(1, 3)
print(odd)

odd[2:2] = [5,7]
print(odd)

# How to delete an element from a list?
del odd[0]
print(odd)

# Remove and pop are the same as in array !!!

# Clear
odd.clear()
print(odd)

# Sort a list
numbers = [1, 5, 2, 3]
numbers.sort()
print(numbers)

# An elegant way to create a list
pow2 = [2 ** x for x in range(10)]
print(pow2)

pow2 = [2 ** x for x in range(10) if x > 5]
print(pow2)

# Membership in list
print(2 in pow2)
print(2 not in pow2)

# Iterate through in a list
for fruit in ['apple', 'banana', 'orange']:
    print('I like', fruit)
