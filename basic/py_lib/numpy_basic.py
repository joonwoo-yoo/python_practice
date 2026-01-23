import numpy as np

# a = np.arange(12).reshape(3,4)
# print(a)
# print(a.shape)
# print(a.dtype)
# print(a.itemsize)
# print(a.size)

# a = np.array([2,3,4])
# print(a)
# print(a.dtype)

# b = np.array([1.2, 3.5, 5.1])
# print(b)
# print(b.dtype)

# print(np.zeros((3,4), dtype=np.int64))
# print(np.ones((2,3,4), dtype=np.int64))
# print(np.empty((2,3)))

# print(np.arange(5))
# print(np.arange(5, 10))
# print(np.arange(10, 30, 5))
# print(np.arange(0,2,.3))

# x = np.linspace(0, 99, 100)
# print(x)

# a = np.arange(0, 1.25, 0.25)
# b = np.linspace(0,1,5)
# print(a)
# print(b)

# a = np.arange(12)
# print(a)
# print(a.reshape((4,3)))

# b = np.arange(24)
# print(b)
# print(b.reshape((2,3,4)))

# a = np.array([20,30,40,50])
# b = np.arange(4)
# print(a)
# print(b)

# c = a-b
# print(c)

# print(b**2)

# print(10*a)

# print(a < 35)

# a = np.array([[1,1],[0,1]])
# b = np.array([[2,0],[3,4]])
# print(a)
# print(b)

# print(a*b)
# print(a@b)

# a = np.ones(3, dtype=np.int64)
# b = np.linspace(0, np.pi,3)

# print(a)
# print(b)
# print(a.dtype)
# print(b.dtype)

# c = a + b
# print(c)
# print(c.dtype)

# d = np.exp(c*1j)
# print(d)
# print(d.dtype)

# a = np.arange(8).reshape(2,4)**2
# print(a)

# print(a.sum())
# print(a.cumsum())

# print(a.min())
# print(a.max())
# print(a.argmax())

# a = np.arange(12).reshape(3,4)
# print(a)

# print(a.sum(axis=1)) # 행기준
# print(a.sum(axis=0)) # 열기준

# # b = np.array([1,4,9])
# # print(b)
# # print(np.sqrt(b))

# a = np.arange(10)**2
# print(a)
# print(a[::-1])

# a[0:6:2] = 1000
# print(a)

# a = np.arange(8)**2
# print(a)

# i = np.array([1,1,3,5])
# print(a[i])

# j = np.array([[3,4],[2,5]])
# print(j)
# print(a[j])

# a = np.arange(12).reshape(3,4)
# print(a)

# b = a > 4
# print(b)

# print(a[b])
# a[b].shape

# a[b] = 0
# print(a)

# a = np.arange(12).reshape(3,4)
# print(a)
# print(a.shape)

# print(a.ravel())
# print(a.reshape(-1))

# print(a.T)
# print(a.T.shape)

# a = a.T
# print(a)

# a = np.array([1,2,3,4]).reshape(2,2)
# b = np.array([5,6,7,8]).reshape(2,2)
# print(a)
# print(b)

# print(np.vstack((a, b)))
# print(np.hstack((a, b)))

a = np.arange(12).reshape(2,6)
print(a)
print(np.hsplit(a, 3))
print(np.hsplit(a, (3,4)))