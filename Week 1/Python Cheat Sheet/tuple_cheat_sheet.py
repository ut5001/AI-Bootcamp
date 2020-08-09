"""
- By convention, we generally use tuple for different datatypes
and list for similar datatypes
- Since tuple are immutable, then iterating through tuple
is faster than with list !!!
- Tuples that contain immutable elements can be used as key
for a dictionary. With list, this is NOT possible.
- If you have data that doesn't change, implementing
it as tuple will GUARANTEE that it remains write-protected
"""

# Empty tuple
tuple1 = ()
print(tuple1)

# Tuple containing integers
tuple1 = (1, 2, 3)
print(tuple1)

# Tuple with mixed datatypes
tuple1 = (1, 'hello', 3.4)
print(tuple1)

# Nested tuple
tuple1 = ('mouse', [8, 4, 6], (1, 2, 3))
print(tuple1)

# How to check 'type' in python
print(type(tuple1))

# Creating a tuple is not necessary to use '( )'
tuple2 = 1, 2, 3
print(type(tuple2))

# Creating a tuple with one element
tuple3 = (1, )
print(type(tuple3))

tuple4 = 1,
print(type(tuple4))

# How to access an element in a tuple

tuple1 = (1, 2, 3, 4, 5)

# Access the first element
print(tuple1[0])

# Access the last element
print(tuple1[-1])

# Slicing elements in a tuple

# Slicing the first element to the second last element
print(tuple1[:-2])

# Tuple is immutable !!!
# tuple1[0] = 0

# Delete a tuple
# del tuple1

tuple1 = (0, 1, 2)

# Concatenation
print((1, 2, 3) + (4, 5, 6))

# How to count a number of elements in a tuple
print(tuple1.count(1))

# How to index a given element in a tuple

print(tuple1.index(2))

# How to iterate elements in a tuple
for name in ('John', 'Kate'):
    print('Hello', name)
