import numpy as np

# pythonList = [0,1,2,3,4,5,6,7,8,9]
# npArray = np.array(pythonList)
# # npArray = np.reshape(pythonList, (2,5))
# print(pythonList)
# print(npArray)

# print(np.zeros(10))
# print(np.zeros((5,5)))

# a = np.arange(0,1.5,0.5)
# print(a)
# b = np.linspace(0, 1, 3)
# print(b)

# python = np.arange(10)**3
# print(python)
# print(python[1])
# print(python[3:6])
# print(python[4:])

python_1 = np.array([1,2,3,4,5,6]).reshape(2,3)
python_2 = np.array([7,8,9,10,11,12]).reshape(2,3)
print(python_1)
print(python_2)

print(np.hsplit(python_1, 3))
print(np.vsplit(python_1, 2))