import array as arr

a = arr.array('d', [1.1, 3.5, 4.5, 5.6, 7.8, 8.9])
print(a)

# How to access elements in an array
print('First element:', a[0])
print('last element:', a[-1])

print('3rd to 5tf elements', a[2:5])
print('3rd to last elements', a[2:])
print('all elements', a[:])

# Append and extend in an array
numbers = arr.array('i', [1, 2, 3])
numbers.append(4)
print(numbers)

numbers.extend([5, 6, 7])
print(numbers)

# Concatenate two arrays using +
odd = arr.array('i', [1, 3, 5])
even = arr.array('i', [2, 4, 6])

numbers = arr.array('i') # Create an empty array
numbers = odd + even
print(numbers)

# Delete a specific element
del numbers[2]
print(numbers)

del numbers
# print(numbers)

# Remove an element in an array
numbers = arr.array('i', [2, 4, 6])
numbers.remove(4)
print(numbers)

# Pop a given index
print(numbers.pop(0))

"""
Use array only if we need it !
List is faster than array in python !!! 
"""
