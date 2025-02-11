"""
DOMAIN 1: Importing and Exporting Data
NumPy provides functions to import and export data efficiently, allowing analysts to work with various file formats like text files and CSV files.

Key Functions:
np.loadtxt('file.txt')
Load data from a text file.
np.genfromtxt('file.csv', delimiter=',')
Load data from a CSV file (supports missing values).
np.savetxt('file.txt', arr, delimiter=' ')
Save NumPy arrays to a text file.
np.savetxt('file.csv', arr, delimiter=',')
Save NumPy arrays to a CSV file.

"""
import numpy as np

# Save an array to a text file
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
np.savetxt('data.txt', data, delimiter=' ')
print("Array saved to 'data.txt'.")

# Load the data back
loaded_data = np.loadtxt('data.txt')
print("Loaded Data:\n", loaded_data)

# Save to CSV
np.savetxt('data.csv', data, delimiter=',')
print("Array saved to 'data.csv'.")

# Load from CSV
csv_data = np.genfromtxt('data.csv', delimiter=',')
print("Loaded CSV Data:\n", csv_data)

"""
Data Analysts: Often retrieve data from CSVs for Excel-like analysis.
Data Scientists: Load and save datasets during preprocessing workflows.
"""
#####################################################################################################################################################################################################
"""
DOMAIN 2: Creating Arrays
Creating arrays is fundamental in NumPy. Arrays can represent datasets, matrices, or other structures used in data analysis.

Key Functions:
np.array()
Create arrays (1D or 2D).
np.zeros()
Generate arrays filled with zeros.
np.ones()
Generate arrays filled with ones.
np.eye()
Identity matrix (1s on the diagonal).
np.linspace()
Create evenly spaced values.
np.arange()
Create arrays with steps.
np.full()
Create arrays filled with a specific value.
np.random.rand()
Create arrays of random floats.
np.random.randint()
Create arrays of random integers.

"""
# 1D and 2D arrays
arr1D = np.array([1, 2, 3])
arr2D = np.array([(1, 2, 3), (4, 5, 6)])
print("1D Array:", arr1D)
print("2D Array:\n", arr2D)

# Zeros and Ones
zeros = np.zeros(3) # It will Create a  Matrix of 3 Columns and and all elements in that Matrix will be zero..[0 0 0]
ones = np.ones((3, 4)) # It will Create a Matrix having 3 Rows and 4 Columns and all elements in that Matrix will be one..see in the Output
print("Zeros Array:", zeros)
print("Ones Array:\n", ones)

# Identity Matrix
identity = np.eye(5)  ## # It will Create a 5x5 Matrix  and all diagonal elements in that Matrix will be one..see in the Output
print("Identity Matrix:\n", identity)

# Evenly spaced values
linspace_arr = np.linspace(0, 100, 6)
print("Linspace Array:", linspace_arr) ## Divide the Array in 6 Parts.. Between 0 to 100 
# The Output is :-  Linspace Array: [  0.  20.  40.  60.  80. 100.]

# ************** Random floats and integers ****************
random_floats = np.random.rand(4, 5) * 100  ## Here 4 Rows and 5 Columns ka Matrix Create hoga.. and Values will Picked Randomly in Floats.. and 100 sai multiply kiya so (All randomly picked values are inside 100)

random_integers = np.random.randint(5, size=(2, 3)) ## Create An Array having 2 Rows and 3 Columns.. and Take Randomly Any Integet value as a Element in that Matrix..
# Now What- ever Value it will choose Randomly for Array Elemnts that elements will Never Greater then 5..(Joh Bhi Elements woh Randomly Select Karega Un-Elements Mai sai Koi Bhi Element 5 sai Bada Nahi hoga )

print("Random Floats:\n", random_floats)
print("Random Integers:\n", random_integers)



######################################################################################################################################################################################################################

"""
DOMAIN 3: Inspecting Properties of Arrays
Understanding array properties is crucial for debugging and ensuring compatibility with other systems or libraries.

Key Functions:
arr.size
Total number of elements in an array.
arr.shape
Dimensions of the array (rows, columns).
arr.dtype
Data type of elements in the array.
arr.astype(dtype)
Convert array elements to a specific type.
arr.tolist()
Convert an array to a Python list.

"""
# Create a 2D array
arr = np.array([[1.5, 2.5], [3.5, 4.5]])

# Inspect properties
print("Array Size:", arr.size)  # Total elements
print("Array Shape:", arr.shape)  # Dimensions
print("Array Data Type:", arr.dtype)  # Data type

# Convert dtype and to a list
converted_arr = arr.astype(int)
print("Converted Array:\n", converted_arr)
print("Array as List:", arr.tolist()) #Convert the Array into in the List..

##############################################################################################################################################################################################

"""
DOMAIN 4: Reshaping, Sorting, and Copying Arrays
Reshaping and sorting arrays is essential for preparing data for analysis and visualization.

Key Functions:
arr.reshape()
Change the shape of an array without modifying data.
arr.resize()
Change the shape of an array and modify data as needed.
arr.sort()
Sort an array.
arr.flatten()
Convert a multi-dimensional array into 1D.
arr.T
Transpose an array.

"""
# Create a 2D array
arr = np.array([[4, 3], [2, 1]])

# Reshape
reshaped = arr.reshape(1, 4)
print("Reshaped Array:\n", reshaped)

# Sort
sorted_arr = np.sort(arr, axis=1)
print("Sorted Array:\n", sorted_arr)

# Flatten and Transpose
flattened = arr.flatten()
transposed = arr.T # Just to make the Tranpose of the Matrix.. which is Given...
print("Flattened Array:", flattened)
print("Transposed Array:\n", transposed)

# Data Analysts: Prepare data for graphing and pivot tables.
#Data Scientists: Reshape arrays for training/testing ML models.

#################################################################################################################################################################
"""
DOMAIN 5: Adding, Removing, and Combining Arrays
Manipulating arrays (adding, removing, or combining) is vital for handling real-world datasets.

Key Functions:
np.append()
Add values to an array.
np.insert()
Insert values at a specific index.
np.delete()
Remove rows/columns from an array.
np.concatenate()
Combine two arrays.
np.split()
Split an array into sub-arrays.

"""
# Append and Insert
arr = np.array([1, 2, 3])
appended = np.append(arr, [4, 5])
inserted = np.insert(arr, 1, 99)
print("Appended Array:", appended)
print("Inserted Array:", inserted)

# Delete and Concatenate
arr2D = np.array([[1, 2], [3, 4]])
deleted = np.delete(arr2D, 1, axis=0)  # Delete row
combined = np.concatenate((arr2D, arr2D), axis=1) #  You Can Change the (axis) and See the Diffrence In Outputs 

""" (Axis =0):- it means Row.. To Performe Operations On Row at Particular Row index which is Mentioned.."""
""" (Axis =1):- it means Column.. To Performe Operations On Column at Particular Column index which is Mentioned.."""

print("Deleted Array:\n", deleted)
print("Combined Array:\n", combined)


# Splitting
split_arr = np.split(arr, 3)  ## To Split the Array After every 3 Elements 
print("Split Arrays:", split_arr)

##############################################################################################################################################################################

"""
1. np.linspace(start, stop, num)
This function creates an array of evenly spaced numbers between start and stop.

Parameters:

start: The starting value of the sequence.
stop: The end value of the sequence.
num: The number of evenly spaced points (default is 50).

"""
import numpy as np

# Example 1: Generate 5 numbers between 0 and 10
array1 = np.linspace(0, 10, 5)
print(array1)  # Output: [ 0.   2.5  5.   7.5 10. ]

# Example 2: Generate 4 numbers between -5 and 5
array2 = np.linspace(-5, 5, 4)
print(array2)  # Output: [-5.  -1.66666667  1.66666667  5. ]

##############################################################################################################################################################################

"""
2. np.average(a)
This function calculates the average (mean) of all elements in an array.
"""
# Example 1: Average of a simple 1D array
array = np.array([1, 2, 3, 4])
print(np.average(array))  # Output: 2.5

# Example 2: Average of a 2D array
array2D = np.array([[1, 2], [3, 4]]) # [1,2]:- 1+2/2 =2  and [3,4]:- 3+4/2 = 3.5=4 now 
                                    # [[1, 2], [3, 4]] = 1+4/2 = 2.5
print(np.average(array2D))  # Output: 2.5

##############################################################################################################################################################################
"""
3. <slice> = <val>
This replaces values in an array using slicing.
"""
## Here we can Assigned Values , on that Particular index by slicing 
# Example 1: Replace first two elements with 99
array = np.array([0, 1, 2, 3])
array[:2] = 99
print(array)  # Output: [99 99  2  3]

# Example 2: Replace all even numbers with -1
array = np.array([0, 1, 2, 3, 4])
array[array % 2 == 0] = -1
print(array)  # Output: [-1  1 -1  3 -1]

##############################################################################################################################################################################
"""
4. np.var(a)
This calculates the variance of the array (how far the numbers are spread out).
"""
# Example 1: Variance of 1D array
array = np.array([1, 2, 3, 4])
print(np.var(array))  # Output: 1.25

# Example 2: Variance of 2D array
array2D = np.array([[1, 2], [3, 4]])
print(np.var(array2D))  # Output: 1.25

##############################################################################################################################################################################

"""5. np.std(a)
This calculates the standard deviation (how much data varies from the average).
"""

# Example 1: Standard deviation of 1D array
array = np.array([1, 2, 3, 4])
print(np.std(array))  # Output: 1.118033988749895

# Example 2: Standard deviation of 2D array
array2D = np.array([[1, 2], [3, 4]])
print(np.std(array2D))  # Output: 1.118033988749895

##############################################################################################################################################################################

"""6. np.diff(a)
This calculates the difference between consecutive elements."""

# Example 1: Difference of a simple array
array = np.array([10, 20, 30, 40])
print(np.diff(array))  # Output: [10 10 10]

# Example 2: Difference of Fibonacci numbers
fibs = np.array([0, 1, 1, 2, 3, 5])
print(np.diff(fibs))  # Output: [1 0 1 1 2]

##############################################################################################################################################################################

"""1. * (Element-wise Multiplication)
The asterisk (*) operator performs element-wise multiplication of two arrays. This means each element in one array is multiplied by the corresponding element in the other array.
"""
import numpy as np

# Example 1: Element-wise multiplication of 1D arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = a * b
print(result)  # Output: [ 4 10 18 ]

a = np.array([1, 2, 3])
b = np.array([6, 8, 4])
result = a * b
print(result)  # Output: [ 6 16 12 ]


# Example 2: Element-wise multiplication of 2D arrays
a = np.array([[2, 0], [0, 2]])
b = np.array([[1, 1], [1, 1]])
result = a * b
print(result)  # Output: [[2 0] [0 2]]

##############################################################################################################################################################################
"""2. np.matmul(a, b) or a @ b (Matrix Multiplication)
This performs matrix multiplication, which is different from element-wise multiplication. In matrix multiplication:

Rows of the first matrix are multiplied by columns of the second matrix.

The number of columns in the first matrix must match the number of rows in the second matrix.
"""
# Here @ is also used to Denote Matrix Multiplication in Numpy-Arrays By-using (matmul) this function ..
# Example 1: Matrix multiplication of 2D arrays
a = np.array([[1, 2],
               [3, 4]])
b = np.array([[2, 0], 
              [1, 2]])
result = np.matmul(a, b)
print(result)
# Output:
# [[ 4  4]
#  [10  8]]

# Example 2: Using @ operator for matrix multiplication
result = a @ b
print(result)
# Output:
# [[ 4  4]
#  [10  8]]




##############################################################################################################################################################################

"""
3. np.arange([start,] stop [,step])
This creates a 1D array of numbers starting from start (default is 0) and ending before stop, with an optional step."""

# Basically this is Same as the "Range" function in list.. Here In Arrays We used "Arange" this function in this...Numpy-Arrays
# Example 1: Generate numbers from 0 to 9
array = np.arange(10)
print(array)  # Output: [0 1 2 3 4 5 6 7 8 9]

# Example 2: Generate even numbers from 0 to 8
array = np.arange(0, 10, 2)
print(array)  # Output: [0 2 4 6 8]

# Example 3: Generate numbers from -5 to 5 with a step of 2
array = np.arange(-5, 6, 2)
print(array)  # Output: [-5 -3 -1  1  3  5]

##############################################################################################################################################################################

"""1. Scalar Math
This domain involves performing mathematical operations on every element of an array by applying a scalar (a single value).

Relevance to Data Analysis and Science
Scalar math is essential for tasks like normalization, scaling, or transforming data, which are common preprocessing steps in data analysis and machine learning.

"""
#Common Functions and Examples
#1.1 Add

import numpy as np

arr = np.array([2, 4, 6, 8])
result = np.add(arr, 1)  # Adds 1 to each element
print(result)
#Output: [3, 5, 7, 9]
#Use Case: Adding a constant to normalize data or shift its range.

#1.2 Subtract

result = np.subtract(arr, 2)  # Subtracts 2 from each element
print(result)
#Output: [0, 2, 4, 6]
#Use Case: Removing a bias from sensor data.

#1.3 Multiply

result = np.multiply(arr, 3)  # Multiplies each element by 3
print(result)
#Output: [6, 12, 18, 24]
#Use Case: Scaling data by a factor.

#1.4 Divide

result = np.divide(arr, 4)  # Divides each element by 4
print(result)
#Output: [0.5, 1.0, 1.5, 2.0]
#Use Case: Converting units (e.g., from millimeters to meters).

#1.5 Power

result = np.power(arr, 2)  # Raises each element to the power of 2
print(result)
#Output: [4, 16, 36, 64]
#Use Case: Computing squares of values, e.g., for statistical variance calculations.
    
"""
2. Vector Math
This domain focuses on elementwise operations between two arrays.

Relevance to Data Analysis and Science
Vector math is fundamental for tasks like calculating residuals (differences between observed and predicted values), vectorized computations, and feature engineering.
"""
#Common Functions and Examples
#2.1 Elementwise Add

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
result = np.add(arr1, arr2)  # Elementwise addition
print(result)
#Output: [5, 7, 9]
#Use Case: Combining multiple features for prediction models.

#2.2 Elementwise Subtract

result = np.subtract(arr1, arr2)  # Elementwise subtraction
print(result)
#Output: [-3, -3, -3]
#Use Case: Calculating residuals in regression.

#2.3 Elementwise Multiply

result = np.multiply(arr1, arr2)  # Elementwise multiplication
print(result)
#Output: [4, 10, 18]
#Use Case: Weighting features by importance.

#2.4 Elementwise Divide

result = np.divide(arr1, arr2)  # Elementwise division
print(result)
#Output: [0.25, 0.4, 0.5]
#Use Case: Normalizing multiple features simultaneously.

#2.5 Power

result = np.power(arr1, arr2)  # Elementwise power
print(result)
#Output: [1, 32, 729]
#Use Case: Exponential transformations in feature scaling.

#2.6 Square Root

result = np.sqrt(arr1)  # Square root of each element
print(result)
#Output: [1.0, 1.41, 1.73]
#Use Case: Scaling features for distance metrics in clustering.

#2.7 Trigonometric Functions

angles = np.array([0, np.pi/2, np.pi])
result = np.sin(angles)  # Sine of each angle
print(result)
#Output: [0.0, 1.0, 0.0]
#Use Case: Signal processing or periodic data analysis.

#2.8 Logarithm

arr = np.array([1, np.e, np.e**2])
result = np.log(arr)  # Natural log of each element
print(result)
#Output: [0.0, 1.0, 2.0]
#Use Case: Logarithmic scaling for skewed data.
    
"""
3. Statistics
This domain covers functions to calculate statistical properties of arrays.

Relevance to Data Analysis and Science
Statistical methods are critical for summarizing data, detecting patterns, and validating hypotheses in analytics and research."""

#Common Functions and Examples
#3.1 Mean
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = np.mean(arr, axis=0)  # Mean along columns
print(result)
#Output: [2.5, 3.5, 4.5]
#Use Case: Average sales across regions.

#3.2 Sum

result = arr.sum()  # Sum of all elements
print(result)
#Output: 21
#Use Case: Total revenue calculation.

#3.3 Min and Max

result_min = arr.min()  # Minimum value
result_max = arr.max()  # Maximum value
print(result_min, result_max)
#Output: 1, 6
#Use Case: Finding outliers in data.

#3.4 Variance and Standard Deviation

result_var = np.var(arr)  # Variance
result_std = np.std(arr)  # Standard deviation
print(result_var, result_std)
#Output: 2.9167, 1.7078
#Use Case: Measuring data dispersion in experiments.

#3.5 Correlation Coefficient

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
result = np.corrcoef(arr1, arr2)  # Correlation coefficient
print(result)
#Output:
"""
[[1. 1.]
 [1. 1.]]"""








