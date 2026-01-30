import numpy as np
print()
print("Numpy examples on numpy version:", np.__version__)
print()
print("creating and using the attributes of numpy arrays")
print()

array_from_list = np.array([1, 2, 3, 4]) # creating numpy array from a list
print("1-D array from list:", array_from_list)

array_from_tuple = np.array((1, 2, 3, 4)) # creating numpy array from a tuple
print("1-D array from tuple:", array_from_tuple)
print("Type of array:", type(array_from_tuple)) # type of numpy array
print("Dimension of array:", array_from_tuple.ndim) # dimension of numpy array
print("Size of array:", array_from_tuple.size) # number of elements in numpy array
print("Shape of array:", array_from_tuple.shape) # shape of numpy array (4,) means 4 elements in 1-D array can be either row or column
print("Data type of array elements:", array_from_tuple.dtype) # data type of numpy array elements
print()

array_2d = np.array([[1, 2, 3],     # Creating a 2-D array from a list of lists
[4, 5, 6],
[7, 8, 9]])
print("2-D array (matrix):\n", array_2d)
print("Dimension of 2-D array:", array_2d.ndim)
print("Size of 2-D array:", array_2d.size) # total number of elements
print("Shape of 2-D array:", array_2d.shape) # rows and columns
print("Data type of 2-D array elements:", array_2d.dtype)
print("Flattened 2-D array:", array_2d.flatten()) # flattening the 2-D array to 1-D
print()

array_3d = np.array([[[1, 2],       # Creating a 3-D array from a list of lists of lists
[3, 4]],
[[5, 6],
[7, 8]]])
print("3-D array:\n", array_3d)
print("Dimension of 3-D array:", array_3d.ndim)
print("Size of 3-D array:", array_3d.size) # total number of elements
print("Shape of 3-D array:", array_3d.shape) # depth, rows and columns
print("Data type of 3-D array elements:", array_3d.dtype)
print("Flattened 3-D array:", array_3d.flatten()) # flattening the 3-D array to 1-D
print()

nested_list = [[1, 2, 3],       # Nested list of integers
[4, 5, 6],
[7, 8, 9]]

array_float = np.array(nested_list, dtype=float)      # Creating a NumPy array with float dtype
print("Array with float dtype:\n", array_float)
print()
# Specify dtype explicitly
array2 = np.array([5, -7.4, 'a', 7.2], dtype=object) #if object was not used then python would try to convert all elements to a common type like string
print("Array with object dtype:", array2)   # if object was not used then output would be ['5' '-7.4' 'a' '7.2']
print("item size in bytes:", array2.itemsize) # size of each element in bytes
print("Total size of array in bytes:", array2.nbytes) # total size of array in bytes
print("starting memory address of array:", array2.data) # starting memory address of array
print()
print("Array slicing and indexing examples")
print()
array3 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90]) # 1-D array works like a list
print("Original array:", array3)
print("Element at index 2:", array3[2]) # accessing element at index 2
print("Elements from index 2 to 5:", array3[2:6]) # slicing from index 2 to 5
print("Elements from start to index 4:", array3[:5]) # slicing from start to index 4
print("Elements from index 5 to end:", array3[5:]) # slicing from index 5 to end
print("Elements with step 2:", array3[::2]) # slicing with step 2
print("Reversed array:", array3[::-1]) # reversing the array
print()
array_2d_example = np.array([[1, 2, 3, 4, 5],       # 2-D array is different from list of lists 
[6, 7, 8, 9, 10],                                   # 2-D does not support nested indexing line list[][] its has arr[row, col]
[11, 12, 13, 14, 15]])
print("Original 2-D array:\n", array_2d_example)
print("Element at row 1, column 2:", array_2d_example[1, 2]) # accessing element at row 1, column 2
print("Elements in row 0:", array_2d_example[0, :]) # accessing all elements in row 0
print("Elements in column 3:", array_2d_example[:, 3]) # accessing all elements in column 3
print("Sub-array from row 0-1 in reverse and column 1-3:\n", array_2d_example[2:0:-1, 1:4]) # slicing sub-array from row 2-0 in reverse and column 1-3
print()
array_3d_example = np.array([[[1, 2, 3],       # 3-D array has (depth, row, col) each part between the commas have start:end:step slicing
[4, 5, 6]],
[[7, 8, 9],
[10, 11, 12]]])
print("Original 3-D array:\n", array_3d_example)
print("Element at depth 0, row 1, column 2:", array_3d_example[0, 1, 2]) # accessing element at depth 0, row 1, column 2
print("Elements in depth 0:", array_3d_example[0]) # accessing all elements in depth 0 
print("Elements in row 1 across all depths:", array_3d_example[:, 1, :]) # accessing all elements in row 1 across all depths
print("Sub-array from depth 0-1, row 0-1 and column 1-2:\n", array_3d_example[0:2, 0:2, 1:3]) # slicing sub-array from depth 0-1, row 0-1 and column 1-2
print()
print("Ellipsis (...) example")
array_4d_example = np.array([[[[1, 2],       # 4-D array has (block, depth, row, col)
[3, 4]],
[[5, 6],
[7, 8]]],
[[[9, 10],
[11, 12]],
[[13, 14],
[15, 16]]]])
print("Original 4-D array:\n", array_4d_example)
print("Element at block 1, depth 0, row 1, column 1:", array_4d_example[1, 0, 1, 1]) # accessing element at block 1, depth 0, row 1, column 1
print("Using ellipsis to access element at block 0, depth 1, row 0, column 1:", array_4d_example[0, 1, ..., 1]) # using ellipsis to access element at block 0, depth 1, row 0, column 1
print("Using ellipsis to access all elements in block 1:", array_4d_example[1, ...]) # using ellipsis to access all elements in block 1
print("Using ellipsis to access all elements in row 0 across all blocks and depths:", array_4d_example[..., 0, :]) # here the structure is block, depth, row, col so ... means all blocks and depths
print()
print("boolean indexing example")
bool_array = np.array([True, False, True, False, True, False, True, False, True]) # boolean array
print("Boolean array:", bool_array)
print("Elements where boolean array is True:", array3[bool_array]) # boolean indexing
print()
print("Fancy indexing example")
fancy_indices = [0, 2, 4, 6, 8] # list of indices
print("Fancy indices:", fancy_indices)
print("Elements at fancy indices:", array3[fancy_indices]) # fancy indexing
print()
print("Arithmetic operations on a 2-D arrays")
print()
array_a = np.array([[1, 2, 3],       # Creating two 2-D arrays for arithmetic operations
[4, 5, 6],
[7, 8, 9]])
array_b = np.array([[9, 8, 7],       # Creating two 2-D arrays for arithmetic operations
[6, 5, 4],
[3, 2, 1]])
print("Array A:\n", array_a)
print("Array B:\n", array_b)
print("Addition (A + B):\n", array_a + array_b) # element-wise addition
print("Subtraction (A - B):\n", array_a - array_b) #
print("Multiplication (A * B):\n", array_a * array_b) # element-wise multiplication
print("Division (A / B):\n", array_a / array_b) # element wise division
print("Exponentiation (A ** 2):\n", array_a ** 2) # element-wise exponentiation
print("Matrix Multiplication (A @ B):\n", array_a @ array_b) # matrix multiplication
print("Transpose of A:\n", array_a.T) # transpose of array A
print("Sum of all elements in A:", np.sum(array_a)) # sum of all elements in array A
print("Mean of all elements in B:", np.mean(array_b)) # mean of all elements in array B
print("Standard Deviation of all elements in A:", np.std(array_a)) # standard deviation of all elements in array A
print("Maximum element in B:", np.max(array_b)) # maximum element in array B
print("Minimum element in A:", np.min(array_a)) # minimum element in array A
print("Row-wise sum of A:", np.sum(array_a, axis=1)) # row-wise sum
print("Column-wise sum of B:", np.sum(array_b, axis=0)) # column-wise sum
print("Dot product of A and B:\n", np.dot(array_a, array_b)) # dot product
print("Element-wise square root of A:\n", np.sqrt(array_a)) # element-wise square root
print("Element-wise natural logarithm of B:\n", np.log(array_b)) # element-wise natural logarithm
print("Element-wise exponential of A:\n", np.exp(array_a)) # element-wise exponential
print("Numpy broadcasting examples")
print()
array_broadcast_a = np.array([[1, 2, 3],       # first array for broadcasting
[4, 5, 6],
[7, 8, 9]])
array_broadcast_b = np.array([10, 20, 30]) # second array for broadcasting
print("Array A:\n", array_broadcast_a)
print("Array B:\n", array_broadcast_b)
broadcast_sum = array_broadcast_a + array_broadcast_b # broadcasting addition means array_broadcast_b is treated as [[10,20,30],[10,20,30],[10,20,30]] because of broadcasting rules which states that when operating on two arrays of different shapes, numpy will 'stretch' the smaller array across the larger array so that they have compatible shapes
print("Broadcasted addition (A + B):\n", broadcast_sum)
print()
print("Sorting examples")
print()
unsorted_array = np.array([5, 2, 9, 1, 5, 6]) # unsorted 1-D array
print("Unsorted array:", unsorted_array)
sorted_array = np.sort(unsorted_array) # sorting the array using np.sort() does not change the original array
print("Sorted array:", sorted_array)
print("Reverse sorted array:", np.sort(unsorted_array)[::-1]) # reverse sorting
print("Indices that would sort the array:", np.argsort(unsorted_array)) # indices that would sort the array
print("Sort using indexes:", unsorted_array[np.argsort(unsorted_array)]) # sorting using the indices from argsort
sorted_array.sort() # in-place sorting
print("In-place sorted array:", unsorted_array) # original array has been changed and is now sorted 
print()
unsorted_2d_array = np.array([[369, 32, 11],       # unsorted 2-D array
[6, 10, 4],
[9, 8, 47]])
print("Unsorted 2-D array:\n", unsorted_2d_array)
sorted_2d_array_row = np.sort(unsorted_2d_array, axis=1) # sorting along rows
print("Row-wise sorted 2-D array:\n", sorted_2d_array_row)
sorted_2d_array_col = np.sort(unsorted_2d_array, axis=0) # sorting along columns
print("Column-wise sorted 2-D array:\n", sorted_2d_array_col)
print("sorted rows and columns at the same time:\n", np.sort(np.sort(unsorted_2d_array, axis=0), axis=1)) # sorting rows and columns at the same time
sorted_2d_array_flat = np.sort(unsorted_2d_array, axis=None) # sorting the flattened array
print("Fully sorted flattened 2-D array:\n", sorted_2d_array_flat)
print("Indices that would sort each row:\n", np.argsort(unsorted_2d_array, axis=1)) # indices that would sort each row
print("Indices that would sort each column:\n", np.argsort(unsorted_2d_array, axis=0)) # indices that would sort each column
print("Axis is -1 means last axis, so sorting along last axis (columns here):\n", np.sort(unsorted_2d_array, axis=-1)) # negative axis means counting from the end not the last accessed axis if row was accessed last then -1 would mean rows
print("Axis is -2 means second last axis, so sorting along second last axis (rows here):\n", np.sort(unsorted_2d_array, axis=-2))
print()
print("Conditional sorting using np.where()")
condition = unsorted_2d_array > 10
print("Condition (elements > 10):", condition)
print("Elements satisfying condition:", unsorted_2d_array[condition])
print("Elements not satisfying condition:", unsorted_2d_array[~condition])
print()
print("Concatenation examples") 
print()
array1 = np.array([[1, 2, 3],       # first 2-D array
[4, 5, 6]])
array2 = np.array([[7, 8, 9],       # second 2-D array
[10, 11, 12]])
print("Array 1:\n", array1)
print("Array 2:\n", array2)
concat_axis0 = np.concatenate((array1, array2), axis=0) # concatenation along rows
print("Concatenation along axis 0 (rows):\n", concat_axis0)
concat_axis1 = np.concatenate((array1, array2), axis=1) # concatenation along columns
print("Concatenation along axis 1 (columns):\n", concat_axis1)
print("matrix off different shapes cannot be concatenated along certain axis:")
array3 = np.array([[13, 14],       # third 2-D array with different shape
[15, 16]])
print("Array 3:\n", array3)
try:
    concat_invalid = np.concatenate((array1, array3), axis=0) # this will raise an error
except ValueError as e:
    print("Error during concatenation along axis 1:", e)
concat_valid = np.concatenate((array1, array3), axis=1) # transposing array3 to make shapes compatible
print("Concatenation of Array 1 and transposed Array 3 along axis 0:\n", concat_valid)
print()
print("Reshape examples")
print()
original_array = np.array([[1, 2, 3, 4, 5, 6],       # original 2-D array
[7, 8, 9, 10, 11, 12]]) 
print("Original array:\n", original_array)
reshaped_array2 = original_array.reshape((3, 4)) # reshaping to 3 rows and 4 columns
print("Reshaped array to (3, 4):\n", reshaped_array2)
reshaped_array3 = original_array.reshape((-1, 2)) # reshaping to 2 columns and inferring rows (means python will calculate rows automatically)
print("Reshaped array to (-1, 2):\n", reshaped_array3)
reshaped_array4 = original_array.reshape((2, -1)) # reshaping to 2 rows and inferring columns
print("Reshaped array to (2, -1):\n", reshaped_array4)
try:
    invalid_reshape = original_array.reshape((5, 5)) # this will raise an error because total elements do not match
except ValueError as e:
    print("Error during reshaping to (5, 5):", e)
print("Flattened array using reshape(-1):", original_array.reshape(-1)) # flattening the array 
print()
print("Spliting examples")
print()
array_to_split = np.array([[1, 2, 3, 4, 5, 6],       # array to be split
[7, 8, 9, 10, 11, 12]])
print("Original array to split:\n", array_to_split)
split_axis0 = np.array_split(array_to_split, 2, axis=0) # splitting into 2 sub-arrays along rows
print("Split into 2 sub-arrays along axis 0 (rows):")
for i, sub_array in enumerate(split_axis0):
    print(f"Sub-array {i+1}:\n", sub_array)
split_axis1 = np.array_split(array_to_split, 3, axis=1) # splitting into 3 sub-arrays along columns
print("Split into 3 sub-arrays along axis 1 (columns):")
for i, sub_array in enumerate(split_axis1):
    print(f"Sub-array {i+1}:\n", sub_array)
print()
print("Numpy array creation functions examples")
print()
array_zeros = np.zeros((2, 3)) # creating a 2-D array of zeros
print("Array of zeros (2, 3):\n", array_zeros)
array_ones = np.ones((3, 2)) # creating a 2-D array of ones
print("Array of ones (3, 2):\n", array_ones)
array_full = np.full((2, 4), 7) # creating a 2-D array filled with 7
print("Array full of 7s (2, 4):\n", array_full)
array_eye = np.eye(4, dtype=int) # creating a 4x4 identity matrix
array_eye1 = np.eye(3,4,-1) # creating a 3x4 matrix where diagonal is shifted down by 1
array_identity = np.identity(4) # creating a 4x4 identity matrix using identity()
print("Identity matrix (4x4):\n", array_eye)
print("Identity matrix with diagonal shifted down by 1 (3x4):\n", array_eye1)
print("Identity matrix using identity() (4x4):\n", array_identity)
array_arange = np.arange(10, 21, 2) # creating a 1-D array with values from 10 to 20 with step 2
print("Array using arange (10 to 20 with step 2):", array_arange)
array_linspace = np.linspace(0, 1, 5) # creating a 1-D array with 5 values evenly spaced between 0 and 1 endpoint included
print("Array using linspace (0 to 1 with 5 values):", array_linspace)
array_linspace2 = np.linspace(0, 10, 4, endpoint=False) # creating a 1-D array with 4 values evenly spaced between 0 and 10 excluding endpoint
print("Array using linspace (0 to 10 with 4 values, endpoint=False):", array_linspace2)
array_linspace3 = np.linspace(0, 1, 5, retstep=True) # creating a 1-D array with 5 values evenly spaced between 0 and 1 and returning the step size
print("Array using linspace (0 to 1 with 5 values) and step size:", array_linspace3)
array_random = np.random.rand(3, 3) # creating a 3x3
print("Array of random values (3x3):\n", array_random)
print()

