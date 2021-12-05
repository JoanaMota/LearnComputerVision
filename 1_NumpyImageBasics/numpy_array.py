import numpy as np

myList = [1, 2, 3]
type(myList)

myArray = np.array(myList)
print(myArray)
type(myArray)

myArray2 = np.arange(0, 10, 2)  # from 0 to 10 with a step size of 2
print(myArray2)

myArray3 = np.zeros(shape=(10, 5))  # rows,columns
print(myArray3)
myArray4 = np.ones(shape=(3, 2))  # rows,columns
print(myArray4)

##

np.random.seed(101)  # seed is used to create always the same random values
arr = np.random.randint(0, 100, 10)
print(arr)
arr2 = np.random.randint(0, 100, 10)
print(arr2)

print(arr.max())
print(arr.argmax())  # index of the max value
print(arr.min())
print(arr.argmin())  # index of the min value
print(arr.mean())

print(arr.shape)
arrReshaped = arr.reshape((5, 2))  # doing nothing
print(arrReshaped.shape)
print(arrReshaped)

##

mat = np.arange(0, 100).reshape(10, 10)
print(mat.shape)
print(mat)
print(mat[4, 6])
column1 = mat[:, 1].reshape(10, 1)  # all rows of column 1
print(column1)
row2 = mat[2, :]  # all columns of row 2
print(row2)
smallMat = mat[0:3, 0:4]  # row 0 to 3 and column 0 to 4
print(smallMat)

newMat = mat.copy()
newMat[0:6, :] = 0
print(newMat)
