import pandas as pd 
import numpy as np

print("Pandas examples on Pandas Version:", pd.__version__)
print("========================================")
print()
# syntax to create a Series object in pandas
# pd.Series(data, index=list_index)
# Creating a Series from a list

# data = [10, 20, 30, 40, 50]
# series = pd.Series(data)
# print("Series from list:")
# print(series)
# print()
# list_index = ['a', 'b', 'c', 'd', 'e']
# series_with_index = pd.Series(data, index=list_index)
# print("Series from list with custom index:")
# print(series_with_index)
# print()
# # Creating a series from a dictionary
# data_dict = {'a': 1, 'b': 2, 'c': 3}
# series_dict = pd.Series(data_dict)
# print("Series from dictionary:")
# print(series_dict)
# print()
# print("Accessing elements in Series:")
# print("Element at index 2:", series[2])  # Accessing by position
# print("Accessing multiple elements:", series[[1, 3, 4]])  # Accessing multiple by position
# print("Element with index 'c':", series_with_index['c'])  # Accessing by label
# print("Accessing multiple elements by label:", series_with_index[['a', 'd']])  # Accessing multiple by label
# print()
# print("positional Slicing in Series (exclusive):")
# print("First 3 elements:\n", series[:3])
# print("Last 3 elements:\n", series[-3:])
# print("Elements from index 1 to 4:\n",series[1:4])
# print("Labled Slicing in Series (inclusive):")
# print("First 3 elements:\n",series_with_index[:3]) # number indexing also work even if labels are present and it is positional and exclusive
# print("Last 3 elements:\n",series_with_index[-3:])
# print("Elements from series_dict:\n",series_dict["b":"c"]) # labeled slicing is inclusive
# print("Elements from index 'a' to 'd':\n",series_with_index['a':'d'])
# print()
# print("Attributes of Series:") # attributes has no parentheses ()
# print("Index of series_with_index:", series_with_index.index)
# print("Values of series_with_index:", series_with_index.values)
# print("Data type of series_with_index:", series_with_index.dtype)
# print("Size of series_with_index:", series_with_index.size)
# print("Shape of series_with_index:", series_with_index.shape)
# print("Dimensions of series_with_index:", series_with_index.ndim)
# print("Is series_with_index empty?:", series_with_index.empty)
# print("Is series_with_index unique?:", series_with_index.is_unique)
# print("Hasnan in series_with_index?:", series_with_index.hasnans) # check if there are any NaN values
# series_with_index.name = "sample_series"
# print("Name of series_with_index:", series_with_index.name)
# print()
# print("Methods of Series:") # methods has parentheses ()
# print("Heading 3 elements of series_with_index:\n", series_with_index.head(3)) # first n elements
# print("Tail 2 elements of series_with_index:\n", series_with_index.tail(2)) # last n elements
# print("Descriptive statistics of series_with_index:\n", series_with_index.describe()) # descriptive statistics
# print("Counting unique values in series_with_index:\n", series_with_index.value_counts()) # count of unique values
# print("Sorting series_with_index by index:\n", series_with_index.sort_index()) # sorting by index
# print("Sorting series_with_index by values:\n", series_with_index.sort_values()) # sorting by values
# print("Dropping index 'b' from series_with_index:\n", series_with_index.drop('b')) # dropping an index
# print("Replacing value 30 with 35 in series_with_index:\n", series_with_index.replace(30, 35)) # replacing values
# print("Checking for null values in series_with_index:\n", series_with_index.isnull()) # check for null values
# print("Checking for non-null values in series_with_index:\n", series_with_index.notnull()) # check for non-null values
# print("Applying a function (squaring) to series_with_index:\n", series_with_index.apply(lambda x: x**2)) # applying a function to each element
# print("Getting unique values from series_with_index:\n", series_with_index.unique()) # getting unique values
# print("Getting index positions of series_with_index:\n", series_with_index.index.tolist()) # getting index positions as a list
# print("Getting values of series_with_index as a list:\n", series_with_index.tolist()) # getting values as a list
# print("Checking if all elements are greater than 15 in series_with_index:\n", series_with_index.gt(15)) # greater than comparison
# print("Checking if any element is equal to 30 in series_with_index:\n", series_with_index.eq(30)) # equality comparison
# print("Cumulative sum of series_with_index:\n", series_with_index.cumsum()) # cumulative sum
# print("Cumulative maximum of series_with_index:\n", series_with_index.cummax()) # cumulative maximum
# print("Cumulative minimum of series_with_index:\n", series_with_index.cummin()) # cumulative minimum
# print("Cumulative product of series_with_index:\n", series_with_index.cumprod()) # cumulative product
# print("Calculating mean of series_with_index:", series_with_index.mean()) # mean of the series
# print("Calculating median of series_with_index:", series_with_index.median()) # median of the series
# print("Calculating standard deviation of series_with_index:", series_with_index.std()) # standard deviation of the series
# print("Calculating variance of series_with_index:", series_with_index.var()) # variance of the series
# print("Calculating sum of series_with_index:", series_with_index.sum()) # sum of the series
# print("Calculating minimum of series_with_index:", series_with_index.min()) # minimum value
# print("Calculating maximum of series_with_index:", series_with_index.max()) # maximum value
# print("Calculating product of series_with_index:", series_with_index.prod()) # product of the series
# print("Calculating quantiles of series_with_index:\n", series_with_index.quantile([0.25, 0.5, 0.75])) # quantiles 
# print("Calculating skewness of series_with_index:", series_with_index.skew()) # skewness
# print("Calculating kurtosis of series_with_index:", series_with_index.kurt()) # kurtosis
# print("Calculating correlation of series_with_index with itself:", series_with_index.corr(series_with_index)) # correlation with itself
# print("Calculating covariance of series_with_index with itself:", series_with_index.cov(series_with_index)) # covariance with itself
# print()
# print("Math operations on Series:")
# print("Original series:")
# print(series)
# print("Adding 5 to each element:\n", series + 5)
# print("Integer division of each element by 4:\n", series // 4)
# print("Math functions on two series:")
# print()
# series2 = pd.Series([5, 15, 25, 35, 45])
# print("Second series:")
# print(series2)
# print("Addition of two series:\n", series + series2)
# print("Modulus of first series by second series:\n", series % series2)
# print()
# print("If the indexes do not match, the result will have NaN for non-matching indexes:")
# series3 = pd.Series([100, 200, 300], index=['a', 'b', 'c'])
# print("Third series with different index:")
# print(series3)
# print("Addition of series_with_index and series3:\n", series_with_index + series3)
# print()
# print("Handling missing values:")
# series_with_nan = pd.Series([10, None, 30, None, 50], index=['a', 'b', 'c', 'd', 'e'])
# print("Series with NaN values:")
# print(series_with_nan)
# print("Filling NaN values with 0:\n", series_with_nan.fillna(0))
# print("Filling NaN values with the mean of the series:\n", series_with_nan.fillna(series_with_nan.mean()))
# print("Dropping NaN values:\n", series_with_nan.dropna())
# print("Checking for NaN values:\n", series_with_nan.isnull())
# print("Checking for non-NaN values:\n", series_with_nan.notnull())
# print()

# print("Database Examples")
# print()
# # Creating an empty DataFrame
# empty_df = pd.DataFrame()
# print(empty_df)         # Displaying the empty DataFrame

# data = np.array([       # Example data
# [25, 'New York'],
# [30, 'Los Angeles'],
# [35, 'Chicago'],
# [40, 'Houston']
# ])
# df = pd.DataFrame(data, columns=['Age', 'City'])    # Creating DataFrame from ndarray
# print(df)

# data = [                                    
# {'Name': 'Alice', 'Age': 25, 'City': 'New York'},
# {'Name': 'Bob', 'Age': 30, 'City': 'Los Angeles'},
# {'Name': 'Charlie', 'Age': 35, 'City': 'Chicago'},
# {'Name': 'David', 'Age': 40, 'City': 'Houston'}
# ]   
# df = pd.DataFrame(data)                             # Creating DataFrame from dictionary here the indices = keys of the dictionary
# print("Dataframe from dictionary:\n",df)        
print()

# print("Reading an excel and csv file")
# print()
# print("Use try except block to open file safely")
# print()
# try:
#     Customers = pd.read_excel("customers.xlsx")
#     Customers_Product_Sheet = pd.read_excel("customers.xlsx" , sheet_name = 1) # used index to access sheet
#     Customers_Purchases_Sheet = pd.read_excel("customers.xlsx" , sheet_name = "purchases") # used name in sting to access sheet
#     Products = pd.read_excel("D:\programming\Python\products.xlsx")
#     Purchases = pd.read_csv("purchases.csv")
#     print("Customer DataFrame")
#     print(Customers.head(5))
#     print()
#     print("Customer_Product_Sheet DataFrame")
#     print(Customers_Product_Sheet.head(5))
#     print()
#     print("Customer_Purchases_Sheet DataFrame")
#     print(Customers_Purchases_Sheet.head(5))
#     print()
#     print("Products DataFrame")
#     print(Products.head(5))
#     print()
#     print("Purchases DataFrame")
#     print(Purchases.head(5))
#     print()
# except FileNotFoundError:
#     print("File not found. Please check the file path.")
# except Exception as e:
#     print(f"An error occurred: {e}")

# print("DataFrame Indexing, Slicing, and Viewing")
# print("="*50)

# # Create a sample DataFrame for demonstration
# df_sample = pd.DataFrame({
#     'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
#     'Age': [25, 30, 35, 40, 28],
#     'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
#     'Salary': [50000, 60000, 75000, 80000, 55000]
# })

# print("\nOriginal DataFrame:")
# print(df_sample)

# print("\n--- Viewing Methods ---")
# print("\nHead (first 3 rows):")
# print(df_sample.head(3))

# print("\nTail (last 2 rows):")
# print(df_sample.tail(2))

# print("\nInfo about DataFrame:")
# print(df_sample.info())

# print("\nDescriptive Statistics:")
# print(df_sample.describe())

# print("\nShape (rows, columns):", df_sample.shape)
# print("Columns:", df_sample.columns.tolist())
# print("Index:", df_sample.index.tolist())

# print("\n--- Column Access ---")
# print("\nAccessing single column 'Name':")
# print(df_sample['Name'])

# print("\nAccessing multiple columns ['Name', 'Age']:")
# print(df_sample[['Name', 'Age']])

# print("\n--- Row Access ---")
# print("\nUsing iloc (position-based):")
# print("Row at position 0:")
# print(df_sample.iloc[0])

# print("\nRows from position 1 to 3:")
# print(df_sample.iloc[1:3])

# print("\nUsing loc (label-based):")
# print("Row at index 2:")
# print(df_sample.loc[2])

# print("\nRows 1 to 3 (inclusive):")
# print(df_sample.loc[1:3])

# print("\n--- Cell Access ---")
# print("\nAccessing cell [0, 'Name']:", df_sample.loc[0, 'Name'])
# print("Accessing cell using iloc [0, 0]:", df_sample.iloc[0, 0])

# print("\n--- Conditional Filtering ---")
# print("\nRows where Age > 30:")
# print(df_sample[df_sample['Age'] > 30])

# print("\nRows where City is 'New York':")
# print(df_sample[df_sample['City'] == 'New York'])

# print("\nRows where City is 'New York and Age id 25:")
# print(df_sample[(df_sample['City'] == 'New York') & (df_sample['Age'] == 25)])

# print("\nRows where City is 'New York or Age id 35:")
# print(df_sample[(df_sample['City'] == 'New York') | (df_sample['Age'] == 35)])
# print()

print("DataFrame Row and Column Manipulation")
print("="*50)

# Create a sample DataFrame for demonstration
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
})

print("\nOriginal DataFrame:")
print(df)

# ===== INSERTING COLUMNS =====
print("\n--- INSERTING COLUMNS ---")

# Method 1: Direct assignment (simplest way)
df['Salary'] = [50000, 60000, 75000] # While making a column make sure that the no. of rows given match the rows in DataFrame o 
print("\nAfter adding 'Salary' column:")
print(df)

# Method 2: Using assign() - creates a new DataFrame (does not modify original)
df_new = df.assign(Department=['IT', 'HR', 'Finance'])
print("\nUsing assign() method (original unchanged):")
print(df_new)

# Method 3: Insert at specific position using insert()
df.insert(1, 'Country', ['USA', 'USA', 'USA'])  # Insert at position 1
print("\nAfter inserting 'Country' at position 1:")
print(df)

# Method 4: Concatenating two columns
# df['NameCity'] = df['Name'] + " " + df["City"] 
print("\nAfter adding 'NameCity' by concatenating Name & City column:")
print(df)

# ===== INSERTING ROWS =====
print("\n--- INSERTING ROWS ---")

# Method 1: Using loc with a new index (simple but can cause issues if index exists)
df.loc[3] = ['David', 'USA', 40, 'Chicago' , 80000]
print("\nAfter adding row with index 3 using loc:")
print(df)

# Method 2: Using concat() - RECOMMENDED for adding multiple rows
new_row = pd.DataFrame({
    'Name': ['Eve'],
    'Country': ['Canada'],
    'Age': [28],
    'Salary': [55000],
    'Department': ['Marketing']
})
df = pd.concat([df, new_row], ignore_index=False)  # ignore_index=True resets index to 0,1,2...
print("\nAfter using concat() to add a row:")
print(df)

# Method 3: Using append()-like functionality with concat
rows_to_add = pd.DataFrame({
    'Name': ['Frank', 'Grace'],
    'Country': ['UK', 'Australia'],
    'Age': [32, 29],
    'Salary': [70000, 65000],
    'Department': ['Operations', 'Sales']
}, index=[5, 6])
df = pd.concat([df, rows_to_add])
print("\nAfter adding multiple rows with concat():")
print(df)

# ===== CONCATENATION =====
print("\n--- CONCATENATION ---")

# Create two DataFrames to concatenate
df1 = pd.DataFrame({
    'Name': ['Alice', 'Bob'],
    'Age': [25, 30]
})

df2 = pd.DataFrame({
    'Name': ['Charlie', 'David'],
    'Age': [35, 40]
})

# Method 1: Concatenate vertically (stacking rows) - axis=0
df_vertical = pd.concat([df1, df2], ignore_index=True)
print("\nConcatenate vertically (axis=0):")
print(df_vertical)

# Method 2: Concatenate horizontally (adding columns) - axis=1
df_horizontal = pd.concat([df1, df2], axis=1)
print("\nConcatenate horizontally (axis=1):")
print(df_horizontal)

# ===== DELETING/DROPPING COLUMNS =====
print("\n--- DELETING COLUMNS ---")

df_drop = df.copy()  # Create a copy to avoid modifying original

# Method 1: Using drop() - most common
df_drop = df_drop.drop('Department', axis=1)  # axis=1 means column
print("\nAfter dropping 'Department' column using drop():")
print(df_drop)

# Method 2: Drop multiple columns
df_drop2 = df_drop.drop(['Country', 'Salary'], axis=1)
print("\nAfter dropping multiple columns:")
print(df_drop2)

# Method 3: Using del (modifies DataFrame in place) - WARNING: affects original
df_temp = df.copy()
del df_temp['Country']
print("\nAfter using del to remove 'Country' column:")
print(df_temp)

# ===== DELETING/DROPPING ROWS =====
print("\n--- DELETING ROWS ---")

df_row_drop = df.copy()

# Method 1: Using drop() with index
df_row_drop = df_row_drop.drop(0)  # Default axis=0 means row
print("\nAfter dropping row with index 0:")
print(df_row_drop)

# Method 2: Drop multiple rows by index
df_row_drop2 = df.copy()
df_row_drop2 = df_row_drop2.drop([1, 2])
print("\nAfter dropping rows with index 1 and 2:")
print(df_row_drop2)

# Method 3: Drop rows by condition
df_row_drop3 = df.copy()
df_row_drop3 = df_row_drop3[df_row_drop3['Age'] > 30]  # Keep only Age > 30
print("\nAfter dropping rows where Age <= 30:")
print(df_row_drop3)

# ===== OVERWRITING/MODIFYING ROWS =====
print("\n--- OVERWRITING ROWS ---")

df_modify = df.copy()

# Method 1: Modify entire row using loc
df_modify.loc[0] = ['Updated', 'Updated_Country', 99, 'Chicago' , 99999, 'Updated_Dept']
print("\nAfter overwriting row at index 0:")
print(df_modify)

# Method 2: Modify specific cell in a row
df_modify.loc[1, 'Name'] = 'Bob_Modified'
print("\nAfter modifying 'Name' cell in row 1:")
print(df_modify)

# Method 3: Modify multiple cells in a row
df_modify.loc[2, ['Name', 'Age']] = ['Charlie_Mod', 40]
print("\nAfter modifying multiple cells in row 2:")
print(df_modify)

# ===== OVERWRITING/MODIFYING COLUMNS =====
print("\n--- OVERWRITING COLUMNS ---")

df_col_modify = df.copy()

# Method 1: Overwrite entire column
df_col_modify['Age'] = [100, 101, 102, 103, 104, 105 , 106]
print("\nAfter overwriting 'Age' column:")
print(df_col_modify)

# Method 2: Modify column based on condition
df_col_modify['Salary'] = df_col_modify['Salary'] * 1.1  # Increase salary by 10%
print("\nAfter increasing all salaries by 10%:")
print(df_col_modify)

# Method 3: Apply function to column values
df_col_modify['Name'] = df_col_modify['Name'].str.upper()  # Convert names to uppercase
print("\nAfter converting 'Name' column to uppercase:")
print(df_col_modify)

print("\n" + "="*50 , "\n")

# ===== RENAMING COLUMNS =====
print("\n--- RENAMING COLUMNS ---")

df_rename = df.copy()

# Method 1: Using rename() with a dictionary (most common and flexible)
# rename() returns a new DataFrame, doesn't modify original unless inplace=True
df_renamed = df_rename.rename(columns={'Name': 'Full_Name', 'Age': 'Years_Old'})
print("\nAfter renaming columns using rename():")
print(df_renamed)

# Method 2: Using rename() with inplace=True (modifies original DataFrame)
df_rename.rename(columns={'City': 'Location'}, inplace=True)
print("\nAfter renaming with inplace=True:")
print(df_rename)

# Method 3: Rename all columns at once using list (order matters!)
df_all_rename = df.copy()
df_all_rename.columns = ['Full_Name', 'Nation', 'Age', 'Residence', 'Monthly_Pay', 'Dept']
print("\nAfter renaming all columns using list:")
print(df_all_rename)

# Method 4: Rename with a function (e.g., convert all to uppercase)
df_func_rename = df.copy()
df_func_rename = df_func_rename.rename(columns=str.upper)  # Convert all column names to uppercase
print("\nAfter renaming columns using function (uppercase):")
print(df_func_rename)

# IMPORTANT: If you try to rename a column that doesn't exist, it won't raise error but won't rename anything
df_safe_rename = df.copy()
df_safe_rename = df_safe_rename.rename(columns={'NonExistent': 'NewName'})  # No error, just ignored
print("\nAfter trying to rename non-existent column (no error, just ignored):")
print(df_safe_rename)

# ===== RENAMING ROWS/INDEX =====
print("\n--- RENAMING ROWS/INDEX ---")

df_row_rename = df.copy()

# Method 1: Using rename() to rename index
df_row_rename = df_row_rename.rename(index={0: 'Row_A', 1: 'Row_B', 2: 'Row_C'})
print("\nAfter renaming index using rename():")
print(df_row_rename)

# Method 2: Reset index and set custom index
df_reset = df.copy()
df_reset.index = ['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth', 'Seventh']
print("\nAfter setting custom index directly:")
print(df_reset)

# ===== UNIQUE() METHOD =====
print("\n--- UNIQUE() METHOD ---")

df_unique = df.copy()

# Method 1: Get unique values from a single column
print("\nUnique values in 'Name' column:")
print(df_unique['Name'].unique())

# Method 2: unique() returns a NumPy array (not a Series)
unique_names = df_unique['Name'].unique()
print(f"\nType of unique values: {type(unique_names)}")  # <class 'numpy.ndarray'>

# Method 3: Get count of unique values using len()
print(f"\nNumber of unique names: {len(unique_names)}")

# Method 4: unique() maintains order of first appearance (unlike set)
df_order = pd.DataFrame({'Fruit': ['Apple', 'Banana', 'Apple', 'Orange', 'Banana']})
print("\nUnique fruits (order of first appearance preserved):")
print(df_order['Fruit'].unique())

# Method 5: Get unique values and convert to list
unique_list = df_unique['Country'].unique().tolist()
print("\nUnique countries as list:")
print(unique_list)

# IMPORTANT: unique() doesn't work on entire DataFrame, only on Series (single column)
# This would cause error: df_unique.unique()  # AttributeError!

# Method 6: Get unique rows (entire DataFrame)
print("\nUsing drop_duplicates() to get unique rows:")
df_with_dupes = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Alice'],
    'Age': [25, 30, 25]
})
print("Original DataFrame with duplicates:")
print(df_with_dupes)
print("\nUnique rows only:")
print(df_with_dupes.drop_duplicates())

# ===== NUNIQUE() METHOD =====
print("\n--- NUNIQUE() METHOD ---")

df_nunique = df.copy()

# Method 1: Count unique values in a single column (returns integer)
print("\nNumber of unique names:", df_nunique['Name'].nunique())

# Method 2: Count unique values for all columns at once (returns Series)
print("\nNumber of unique values per column:")
print(df_nunique.nunique())

# Method 3: nunique() ignores NaN values by default
df_with_nan = pd.DataFrame({
    'Name': ['Alice', 'Bob', None, 'Alice'],
    'Age': [25, 30, 35, 25]
})
print("\nDataFrame with NaN values:")
print(df_with_nan)
print("\nUnique count (NaN ignored by default):")
print(df_with_nan.nunique())

# Method 4: dropna=False to count NaN as a unique value
print("\nUnique count (including NaN as unique):")
print(df_with_nan.nunique(dropna=False))

# Method 5: nunique() on entire DataFrame by column
print("\nUnique values per column in full DataFrame:")
print(df_nunique.nunique())

# IMPORTANT: nunique() returns 0 for empty Series, not an error
empty_series = pd.Series([], dtype='object')
print(f"\nUnique count of empty Series: {empty_series.nunique()}")

# ===== TYPE CONVERSION USING astype() =====
print("\n--- TYPE CONVERSION USING astype() ---")

df_types = df.copy()

# Method 1: Convert single column to different type (int, float, str, etc.)
print("\nOriginal DataFrame:")
print(df_types)
print("\nOriginal data types:")
print(df_types.dtypes)

df_types['Age'] = df_types['Age'].astype(str)  # Convert Age to string
print("\nAfter converting 'Age' to string:")
print(df_types)
print(df_types.dtypes)

# Method 2: Convert multiple columns at once
df_convert = df.copy()
df_convert = df_convert.astype({'Age': 'str', 'Salary': 'float'})
print("\nAfter converting multiple columns:")
print(df_convert.dtypes)

# Method 3: Convert all columns to specific type
df_all_str = df.copy()
df_all_str = df_all_str.astype(str)  # Convert all to string
print("\nAfter converting all columns to string:")
print(df_all_str.dtypes)

# Method 4: Convert int to float
df_float = df.copy()
df_float['Age'] = df_float['Age'].astype(float)
print("\nAfter converting 'Age' to float:")
print(df_float['Age'])
print(df_float['Age'].dtype)

# Method 5: Convert string to int (only if string contains valid numbers)
df_str_to_int = pd.DataFrame({'Numbers': ['10', '20', '30']})
print("\nBefore converting string to int:")
print(df_str_to_int['Numbers'].dtype)
df_str_to_int['Numbers'] = df_str_to_int['Numbers'].astype(int)
print("\nAfter converting string to int:")
print(df_str_to_int['Numbers'].dtype)
print(df_str_to_int)

# IMPORTANT ERROR: Converting invalid string to int raises ValueError
# This would cause error:
# df_invalid = pd.DataFrame({'Numbers': ['10', 'ABC', '30']})
# df_invalid['Numbers'] = df_invalid['Numbers'].astype(int)  # ValueError!

# Method 6: Using errors parameter to handle invalid conversions
df_invalid = pd.DataFrame({'Numbers': ['10', 'ABC', '30']})
print("\nUsing errors='coerce' to convert invalid values to NaN:")
df_invalid['Numbers'] = pd.to_numeric(df_invalid['Numbers'], errors='coerce')
print(df_invalid)
print(df_invalid.dtypes)

# Method 7: Convert to categorical type (useful for memory optimization)
df_cat = df.copy()
df_cat['Country'] = df_cat['Country'].astype('category')
print("\nAfter converting 'Country' to categorical:")
print(df_cat['Country'].dtype)
print(df_cat)

# Method 8: Convert to boolean type
df_bool = pd.DataFrame({'Active': ['True', 'False', 'True']})
print("\nBefore converting to boolean:")
print(df_bool['Active'].dtype)
# NOTE: String 'True'/'False' won't directly convert to bool, need custom mapping
df_bool['Active'] = df_bool['Active'].map({'True': True, 'False': False})
print("\nAfter converting to boolean using map():")
print(df_bool['Active'].dtype)
print(df_bool)

# Method 9: Convert to datetime type
df_datetime = pd.DataFrame({'Date': ['2023-01-15', '2023-02-20', '2023-03-10']})
print("\nBefore converting to datetime:")
print(df_datetime['Date'].dtype)
df_datetime['Date'] = pd.to_datetime(df_datetime['Date'])
print("\nAfter converting to datetime:")
print(df_datetime['Date'].dtype)
print(df_datetime)

# IMPORTANT: Use pd.to_numeric() and pd.to_datetime() for safer conversions
# Method 10: Using errors='ignore' (returns original if conversion fails)
df_ignore = pd.DataFrame({'Values': ['10', 'ABC', '30']})
df_ignore['Values'] = pd.to_numeric(df_ignore['Values'], errors='ignore')
print("\nUsing errors='ignore' (keeps original if conversion fails):")
print(df_ignore)

# Method 11: Check data types before conversion
print("\nData types in DataFrame:")
print(df.dtypes)

# Method 12: Convert float with NaN to int safely (convert NaN to value first)
df_nan_int = pd.DataFrame({'Numbers': [10.5, 20.3, None, 40.1]})
print("\nDataFrame with float and NaN:")
print(df_nan_int)
print("\nConverting float with NaN to int (fillna first, then convert):")
df_nan_int['Numbers'] = df_nan_int['Numbers'].fillna(0).astype(int)
print(df_nan_int)
print(df_nan_int.dtypes)

print("\n" + "="*50)