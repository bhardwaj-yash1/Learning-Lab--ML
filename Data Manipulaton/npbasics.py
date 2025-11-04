# numpys are fast nd arrays means n dimension arrays


print("STAGE 2")
import numpy as np
arr1 = np.array([1,2,3])
arr2=np.array([[1,2],[3,4]])
arr3= np.array([[1,2], [3,4],[4,5]])

print(arr1)
print(arr2)
print(arr3)
print(np.zeros((2,3)))
print(np.ones((2,3)))
print(np.eye(3)) # identity matrix of 3x3
print(np.full((2,3),7)) # fills complete matrix with 7
print(np.random.rand(3,3)) # Random floats (0 to 1)
print(np.arange(0,10,2))
print(np.linspace(0,1,5))

# STAGE 3 - INDEXING

print("---------indexing------------")
print("1D array")

arr=np.array([1,2,3,4,5,6])
print(arr[0:5:2])
print(arr[::-1])


print("2D array")

mat = np.array([[1,2,3], [4,5,6]])
# print(row indexing , column indexing )
print(mat[0, 1])     # Element at row 0, col 1 → 2
print(mat[:, 1])     # All rows, col 1 → [2 5]
print(mat[1, :])     # Row 1 → [4 5 6]


# STAGE 4

# SHAPE AND RESHAPE 

a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.shape) # returns rows x columns
print(a.reshape(3,2))
print(a.flatten())


#STAGE 5: Mathematical Operations


print("mathematical operations :")

print("--element wise--")

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a+b)
print(a*b)
print(a**2)

print("--matrix multiplication--")
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print(np.dot(a, b))   # Dot product (2x2) gives same result as normal matrix multiplication

print("-- Sum, Mean, Std--")

arr = np.array([[1, 2], [3, 4]])
print(arr.sum())  # 10
print(arr.mean())  # 2.5
print(arr.std())  # Standard deviation
print(arr.sum(axis=0))  # Column-wise sum
# to get row wise sum set axis =row or axis = 1


print("STAGE 6: Broadcasting (ML Goldmine)")

print("addin scalar to a matrix directly")
a= np.array([1,2,3,4])
print(a+10)

print("adding a row / column vector ")

b = np.array([[1],[2]])
a=np.array([[10,20],[30,40]])
print(a+b) # adds row of b to row of a 

print("\n")
b = np.array([[1,2],[2,3]])
a=np.array([[10,20],[30,40]])
print(a+b)

print("STAGE 7: Boolean Indexing & Filtering ")

arr=np.array([10,20,30,40])
mask = arr>20
print(mask) # returns a boolean array which contains true and false for given condition after checkking foor each element

print(arr[mask]) # returns an array with values satisfying the given condition

print(" STAGE 8: Useful Functions for ML")

print("Index of max value")
arr=np.array([[1,2,5],[1,2,34],[2,34,5]])
print(arr)
print(np.argmax(arr))

print("Index of min value")
print(np.argmin(arr))

print("Limit values between a min/max")
print(np.clip(arr,15,30)) # a value less than min value becomes min value and
# a value larger than max value becomes max value
print("Get unique values (like labels)")
print(np.unique(arr))
print("Conditional selection")
print(np.where(arr>35,1,0))
