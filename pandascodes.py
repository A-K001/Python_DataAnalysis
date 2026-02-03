import pandas as pd 
import numpy as np
from sqlalchemy import create_engine

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
# print()

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

# print("DataFrame Row and Column Manipulation")
# print("="*50)

# # Create a sample DataFrame for demonstration
# df = pd.DataFrame({
#     'Name': ['Alice', 'Bob', 'Charlie'],
#     'Age': [25, 30, 35],
#     'City': ['New York', 'Los Angeles', 'Chicago']
# })

# print("\nOriginal DataFrame:")
# print(df)

# # ===== INSERTING COLUMNS =====
# print("\n--- INSERTING COLUMNS ---")

# # Method 1: Direct assignment (simplest way)
# df['Salary'] = [50000, 60000, 75000] # While making a column make sure that the no. of rows given match the rows in DataFrame o 
# print("\nAfter adding 'Salary' column:")
# print(df)

# # Method 2: Using assign() - creates a new DataFrame (does not modify original)
# df_new = df.assign(Department=['IT', 'HR', 'Finance'])
# print("\nUsing assign() method (original unchanged):")
# print(df_new)

# # Method 3: Insert at specific position using insert()
# df.insert(1, 'Country', ['USA', 'USA', 'USA'])  # Insert at position 1
# print("\nAfter inserting 'Country' at position 1:")
# print(df)

# # Method 4: Concatenating two columns
# # df['NameCity'] = df['Name'] + " " + df["City"] 
# print("\nAfter adding 'NameCity' by concatenating Name & City column:")
# print(df)

# # ===== INSERTING ROWS =====
# print("\n--- INSERTING ROWS ---")

# # Method 1: Using loc with a new index (simple but can cause issues if index exists)
# df.loc[3] = ['David', 'USA', 40, 'Chicago' , 80000]
# print("\nAfter adding row with index 3 using loc:")
# print(df)

# # Method 2: Using concat() - RECOMMENDED for adding multiple rows
# new_row = pd.DataFrame({
#     'Name': ['Eve'],
#     'Country': ['Canada'],
#     'Age': [28],
#     'Salary': [55000],
#     'Department': ['Marketing']
# })
# df = pd.concat([df, new_row], ignore_index=False)  # ignore_index=True resets index to 0,1,2...
# print("\nAfter using concat() to add a row:")
# print(df)

# # Method 3: Using append()-like functionality with concat
# rows_to_add = pd.DataFrame({
#     'Name': ['Frank', 'Grace'],
#     'Country': ['UK', 'Australia'],
#     'Age': [32, 29],
#     'Salary': [70000, 65000],
#     'Department': ['Operations', 'Sales']
# }, index=[5, 6])
# df = pd.concat([df, rows_to_add])
# print("\nAfter adding multiple rows with concat():")
# print(df)

# # ===== DELETING/DROPPING COLUMNS =====
# print("\n--- DELETING COLUMNS ---")

# df_drop = df.copy()  # Create a copy to avoid modifying original

# # Method 1: Using drop() - most common
# df_drop = df_drop.drop('Department', axis = 1)  # axis=1 means column
# print("\nAfter dropping 'Department' column using drop():")
# print(df_drop)

# # Method 2: Drop multiple columns
# df_drop2 = df_drop.drop(['Country', 'Salary'], axis=1)
# print("\nAfter dropping multiple columns:")
# print(df_drop2)

# # Method 3: Using del (modifies DataFrame in place) - WARNING: affects original
# df_temp = df.copy()
# del df_temp['Country']
# print("\nAfter using del to remove 'Country' column:")
# print(df_temp)

# # ===== DELETING/DROPPING ROWS =====
# print("\n--- DELETING ROWS ---")

# df_row_drop = df.copy()

# # Method 1: Using drop() with index
# df_row_drop = df_row_drop.drop(0)  # Default axis=0 means row
# print("\nAfter dropping row with index 0:")
# print(df_row_drop)

# # Method 2: Drop multiple rows by index
# df_row_drop2 = df.copy()
# df_row_drop2 = df_row_drop2.drop([1, 2])
# print("\nAfter dropping rows with index 1 and 2:")
# print(df_row_drop2)

# # Method 3: Drop rows by condition
# df_row_drop3 = df.copy()
# df_row_drop3 = df_row_drop3[df_row_drop3['Age'] > 30]  # Keep only Age > 30
# print("\nAfter dropping rows where Age <= 30:")
# print(df_row_drop3)

# # ===== OVERWRITING/MODIFYING ROWS =====
# print("\n--- OVERWRITING ROWS ---")

# df_modify = df.copy()

# # Method 1: Modify entire row using loc
# df_modify.loc[0] = ['Updated', 'Updated_Country', 99, 'Chicago' , 99999, 'Updated_Dept']
# print("\nAfter overwriting row at index 0:")
# print(df_modify)

# # Method 2: Modify specific cell in a row
# df_modify.loc[1, 'Name'] = 'Bob_Modified'
# print("\nAfter modifying 'Name' cell in row 1:")
# print(df_modify)

# # Method 3: Modify multiple cells in a row
# df_modify.loc[2, ['Name', 'Age']] = ['Charlie_Mod', 40]
# print("\nAfter modifying multiple cells in row 2:")
# print(df_modify)

# # ===== OVERWRITING/MODIFYING COLUMNS =====
# print("\n--- OVERWRITING COLUMNS ---")

# df_col_modify = df.copy()

# # Method 1: Overwrite entire column
# df_col_modify['Age'] = [100, 101, 102, 103, 104, 105 , 106]
# print("\nAfter overwriting 'Age' column:")
# print(df_col_modify)

# # Method 2: Modify column based on condition
# df_col_modify['Salary'] = df_col_modify['Salary'] * 1.1  # Increase salary by 10%
# print("\nAfter increasing all salaries by 10%:")
# print(df_col_modify)

# # Method 3: Apply function to column values
# df_col_modify['Name'] = df_col_modify['Name'].str.upper()  # Convert names to uppercase
# print("\nAfter converting 'Name' column to uppercase:")
# print(df_col_modify)

# print("\n" + "="*50 , "\n")

# # ===== RENAMING COLUMNS =====
# print("\n--- RENAMING COLUMNS ---")

# df_rename = df.copy()

# # Method 1: Using rename() with a dictionary (most common and flexible)
# # rename() returns a new DataFrame, doesn't modify original unless inplace=True
# df_renamed = df_rename.rename(columns={'Name': 'Full_Name', 'Age': 'Years_Old'})
# print("\nAfter renaming columns using rename():")
# print(df_renamed)

# # Method 2: Using rename() with inplace=True (modifies original DataFrame)
# df_rename.rename(columns={'City': 'Location'}, inplace=True)
# print("\nAfter renaming with inplace=True:")
# print(df_rename)

# # Method 3: Rename all columns at once using list (order matters!)
# df_all_rename = df.copy()
# df_all_rename.columns = ['Full_Name', 'Nation', 'Age', 'Residence', 'Monthly_Pay', 'Dept']
# print("\nAfter renaming all columns using list:")
# print(df_all_rename)

# # Method 4: Rename with a function (e.g., convert all to uppercase)
# df_func_rename = df.copy()
# df_func_rename = df_func_rename.rename(columns=str.upper)  # Convert all column names to uppercase
# print("\nAfter renaming columns using function (uppercase):")
# print(df_func_rename)

# # IMPORTANT: If you try to rename a column that doesn't exist, it won't raise error but won't rename anything
# df_safe_rename = df.copy()
# df_safe_rename = df_safe_rename.rename(columns={'NonExistent': 'NewName'})  # No error, just ignored
# print("\nAfter trying to rename non-existent column (no error, just ignored):")
# print(df_safe_rename)

# # ===== RENAMING ROWS/INDEX =====
# print("\n--- RENAMING ROWS/INDEX ---")

# df_row_rename = df.copy()

# # Method 1: Using rename() to rename index
# df_row_rename = df_row_rename.rename(index={0: 'Row_A', 1: 'Row_B', 2: 'Row_C'})
# print("\nAfter renaming index using rename():")
# print(df_row_rename)

# # Method 2: Reset index and set custom index
# df_reset = df.copy()
# df_reset.index = ['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth', 'Seventh']
# print("\nAfter setting custom index directly:")
# print(df_reset)

# # ===== UNIQUE() METHOD =====
# print("\n--- UNIQUE() METHOD ---")

# df_unique = df.copy()

# # Method 1: Get unique values from a single column
# print("\nUnique values in 'Name' column:")
# print(df_unique['Name'].unique())

# # Method 2: unique() returns a NumPy array (not a Series)
# unique_names = df_unique['Name'].unique()
# print(f"\nType of unique values: {type(unique_names)}")  # <class 'numpy.ndarray'>

# # Method 3: Get count of unique values using len()
# print(f"\nNumber of unique names: {len(unique_names)}")

# # Method 4: unique() maintains order of first appearance (unlike set)
# df_order = pd.DataFrame({'Fruit': ['Apple', 'Banana', 'Apple', 'Orange', 'Banana']})
# print("\nUnique fruits (order of first appearance preserved):")
# print(df_order['Fruit'].unique())

# # Method 5: Get unique values and convert to list
# unique_list = df_unique['Country'].unique().tolist()
# print("\nUnique countries as list:")
# print(unique_list)

# # IMPORTANT: unique() doesn't work on entire DataFrame, only on Series (single column)
# # This would cause error: df_unique.unique()  # AttributeError!

# # Method 6: Get unique rows (entire DataFrame) using drop_duplicate()
# print("\nUsing drop_duplicates() to get unique rows:")
# df_with_dupes = pd.DataFrame({
#     'Name': ['Alice', 'Bob', 'Alice'],
#     'Age': [25, 30, 25]
# })
# print("Original DataFrame with duplicates:")
# print(df_with_dupes)
# print("\nUnique rows only:")
# print(df_with_dupes.drop_duplicates())

# # ===== NUNIQUE() METHOD =====
# print("\n--- NUNIQUE() METHOD ---")

# df_nunique = df.copy()

# # Method 1: Count unique values in a single column (returns integer)
# print("\nNumber of unique names:", df_nunique['Name'].nunique())

# # Method 2: Count unique values for all columns at once (returns Series)
# print("\nNumber of unique values per column:")
# print(df_nunique.nunique())

# # Method 3: nunique() ignores NaN values by default
# df_with_nan = pd.DataFrame({
#     'Name': ['Alice', 'Bob', None, 'Alice'],
#     'Age': [25, 30, 35, 25]
# })
# print("\nDataFrame with NaN values:")
# print(df_with_nan)
# print("\nUnique count (NaN ignored by default):")
# print(df_with_nan.nunique())

# # Method 4: dropna=False to count NaN as a unique value
# print("\nUnique count (including NaN as unique):")
# print(df_with_nan.nunique(dropna=False))

# # Method 5: nunique() on entire DataFrame by column
# print("\nUnique values per column in full DataFrame:")
# print(df_nunique.nunique())

# # IMPORTANT: nunique() returns 0 for empty Series, not an error
# empty_series = pd.Series([], dtype='object')
# print(f"\nUnique count of empty Series: {empty_series.nunique()}")

# # ===== TYPE CONVERSION USING astype() =====
# print("\n--- TYPE CONVERSION USING astype() ---")

# df_types = df.copy()

# # Method 1: Convert single column to different type (int, float, str, etc.)
# print("\nOriginal DataFrame:")
# print(df_types)
# print("\nOriginal data types:")
# print(df_types.dtypes)

# df_types['Age'] = df_types['Age'].astype(str)  # Convert Age to string
# print("\nAfter converting 'Age' to string:")
# print(df_types)
# print(df_types.dtypes)

# # Method 2: Convert multiple columns at once
# df_convert = df.copy()
# df_convert = df_convert.astype({'Age': 'str', 'Salary': 'float'})
# print("\nAfter converting multiple columns:")
# print(df_convert.dtypes)

# # Method 3: Convert all columns to specific type
# df_all_str = df.copy()
# df_all_str = df_all_str.astype(str)  # Convert all to string
# print("\nAfter converting all columns to string:")
# print(df_all_str.dtypes)

# # Method 4: Convert int to float
# df_float = df.copy()
# df_float['Age'] = df_float['Age'].astype(float)
# print("\nAfter converting 'Age' to float:")
# print(df_float['Age'])
# print(df_float['Age'].dtype)

# # Method 5: Convert string to int (only if string contains valid numbers)
# df_str_to_int = pd.DataFrame({'Numbers': ['10', '20', '30']})
# print("\nBefore converting string to int:")
# print(df_str_to_int['Numbers'].dtype)
# df_str_to_int['Numbers'] = df_str_to_int['Numbers'].astype(int)
# print("\nAfter converting string to int:")
# print(df_str_to_int['Numbers'].dtype)
# print(df_str_to_int)

# # IMPORTANT ERROR: Converting invalid string to int raises ValueError
# # This would cause error:
# # df_invalid = pd.DataFrame({'Numbers': ['10', 'ABC', '30']})
# # df_invalid['Numbers'] = df_invalid['Numbers'].astype(int)  # ValueError!

# # Method 6: Using errors parameter to handle invalid conversions
# df_invalid = pd.DataFrame({'Numbers': ['10', 'ABC', '30']})
# print("\nUsing errors='coerce' to convert invalid values to NaN:")
# df_invalid['Numbers'] = pd.to_numeric(df_invalid['Numbers'], errors='coerce')
# print(df_invalid)
# print(df_invalid.dtypes)

# # Method 7: Convert to categorical type (useful for memory optimization)
# df_cat = df.copy()
# df_cat['Country'] = df_cat['Country'].astype('category')
# print("\nAfter converting 'Country' to categorical:")
# print(df_cat['Country'].dtype)
# print(df_cat)

# # Method 8: Convert to boolean type
# df_bool = pd.DataFrame({'Active': ['True', 'False', 'True']})
# print("\nBefore converting to boolean:")
# print(df_bool['Active'].dtype)
# # NOTE: String 'True'/'False' won't directly convert to bool, need custom mapping
# df_bool['Active'] = df_bool['Active'].map({'True': True, 'False': False})
# print("\nAfter converting to boolean using map():")
# print(df_bool['Active'].dtype)
# print(df_bool)

# # Method 9: Convert to datetime type using pd.to_datetime()
# df_datetime = pd.DataFrame({'Date': ['2023-01-15', '2023-02-20', '2023-03-10']})
# print("\nBefore converting to datetime:")
# print(df_datetime['Date'].dtype)
# df_datetime['Date'] = pd.to_datetime(df_datetime['Date'])
# print("\nAfter converting to datetime:")
# print(df_datetime['Date'].dtype)
# print(df_datetime)

# # IMPORTANT: Use pd.to_numeric() and pd.to_datetime() for safer conversions
# df_ignore = pd.DataFrame({'Values': ['10', 'ABC', '30']})
# df_ignore['Values'] = pd.to_numeric(df_ignore['Values'], errors='coerce')
# print("\nUsing errors='coerce' (Fills with NaN if error type can't be converted):")
# print(df_ignore)
# print(df_ignore['Values'].dtypes)

# # Method 10: Using errors='ignore' (returns original if conversion fails)
# df_ignore = pd.DataFrame({'Values': ['10', 'ABC', '30']})
# df_ignore['Values'] = df_ignore.astype(int, errors='ignore')
# print("\nUsing errors='ignore' (keeps original if conversion fails):")
# print(df_ignore)
# print(df_ignore['Values'].dtypes)

# # Method 11: Check data types before conversion
# print("\nData types in DataFrame:")
# print(df.dtypes)

# # Method 12: Convert float with NaN to int safely (convert NaN to value first)
# df_nan_int = pd.DataFrame({'Numbers': [10.5, 20.3, None, 40.1]})
# print("\nDataFrame with float and NaN:")
# print(df_nan_int)
# print("\nConverting float with NaN to int (fillna first, then convert):")
# df_nan_int['Numbers'] = df_nan_int['Numbers'].fillna(0).astype(int)
# print(df_nan_int)
# print(df_nan_int.dtypes)

# print("\n" + "="*50)


# print("\nJoining of tables\n")
# # ===== CONCATENATION =====
# print("\n--- CONCATENATION ---")

# # Create two DataFrames to concatenate
# df1 = pd.DataFrame({
#     'Name': ['Alice', 'Bob'],
#     'Age': [25, 30]
# })

# df2 = pd.DataFrame({
#     'Name': ['Charlie', 'David'],
#     'Age': [35, 40]
# })

# # Method 1: Concatenate vertically (stacking rows) - axis=0
# df_vertical = pd.concat([df1, df2], ignore_index=True)
# print("\nConcatenate vertically (axis=0):")
# print(df_vertical)

# # Method 2: Concatenate horizontally (adding columns) - axis=1
# df_horizontal = pd.concat([df1, df2], axis=1)
# print("\nConcatenate horizontally (axis=1):")
# print(df_horizontal)

# # ===== MERGE AND JOIN =====
# print("\n--- MERGE AND JOIN ---")

# # Create sample DataFrames for merging
# df_employees = pd.DataFrame({
#     'EmployeeID': [1, 2, 3, 4],
#     'Name': ['Alice', 'Bob', 'Charlie', 'David'],
#     'DepartmentID': [10, 20, 10, 30]
# })

# df_departments = pd.DataFrame({
#     'DepartmentID': [10, 20, 30, 40],
#     'DepartmentName': ['IT', 'HR', 'Finance', 'Sales']
# })

# print("\nEmployees DataFrame:")
# print(df_employees)
# print("\nDepartments DataFrame:")
# print(df_departments)

# # Method 1: Inner Join (only matching rows from both DataFrames)
# print("\n--- INNER JOIN (default) ---")
# df_inner = pd.merge(df_employees, df_departments, on='DepartmentID', how='inner')
# print("Inner join result (only matching DepartmentIDs):")
# print(df_inner)

# # Method 2: Left Join (all rows from left DataFrame, matching from right)
# print("\n--- LEFT JOIN ---")
# df_left = pd.merge(df_employees, df_departments, on='DepartmentID', how='left')
# print("Left join result (all employees, matched departments):")
# print(df_left)

# # Method 3: Right Join (all rows from right DataFrame, matching from left)
# print("\n--- RIGHT JOIN ---")
# df_right = pd.merge(df_employees, df_departments, on='DepartmentID', how='right')
# print("Right join result (all departments, matched employees):")
# print(df_right)

# # Method 4: Outer Join (all rows from both DataFrames, NaN where no match)
# print("\n--- OUTER JOIN ---")
# df_outer = pd.merge(df_employees, df_departments, on='DepartmentID', how='outer')
# print("Outer join result (all rows from both, NaN for non-matches):")
# print(df_outer)

# # Method 5: Merge on different column names
# print("\n--- MERGE WITH DIFFERENT COLUMN NAMES ---")
# df_dept_renamed = df_departments.rename(columns={'DepartmentID': 'DeptID'})
# df_merged_renamed = pd.merge(df_employees, df_dept_renamed, left_on='DepartmentID', right_on='DeptID', how='inner')
# print("Merge using left_on and right_on:")
# print(df_merged_renamed)

# # Method 6: Merge on multiple columns
# print("\n--- MERGE ON MULTIPLE COLUMNS ---")
# df_sales1 = pd.DataFrame({
#     'Region': ['North', 'South', 'East'],
#     'Quarter': ['Q1', 'Q1', 'Q2'],
#     'Sales': [1000, 1500, 2000]
# })

# df_sales2 = pd.DataFrame({
#     'Region': ['North', 'South', 'East'],
#     'Quarter': ['Q1', 'Q1', 'Q2'],
#     'Target': [900, 1400, 1800]
# })

# df_multi_merge = pd.merge(df_sales1, df_sales2, on=['Region', 'Quarter'], how='inner')
# print("Merge on multiple columns (Region and Quarter):")
# print(df_multi_merge)

# # Method 7: Join using index (concat with axis=1 and join parameter)
# print("\n--- JOIN USING INDEX ---")
# df_index1 = pd.DataFrame({'A': [10, 20, 30]}, index=['a', 'b', 'c'])
# df_index2 = pd.DataFrame({'B': [100, 200, 300]}, index=['a', 'b', 'c'])
# df_join = df_index1.join(df_index2)
# print("Join on index:")
# print(df_join)

# # Method 8: Join with mismatched indexes
# print("\n--- JOIN WITH MISMATCHED INDEXES ---")
# df_index3 = pd.DataFrame({'C': [1000, 2000]}, index=['a', 'd'])
# df_join_mismatch = df_index1.join(df_index3, how='left')  # Left join keeps all from df_index1
# print("Left join with mismatched index (all from left):")
# print(df_join_mismatch)

# print("\n" + "="*50)

# # ===== HANDLING MISSING VALUES =====
# print("\n--- HANDLING MISSING VALUES ---")

# # Create DataFrame with missing values
# df_missing = pd.DataFrame({
#     'Name': ['Alice', 'Bob', None, 'David', 'Eve'],
#     'Age': [25, None, 35, 40, None],
#     'Salary': [50000, 60000, 75000, None, 55000],
#     'Department': ['IT', 'HR', 'Finance', 'Sales', None]
# })

# print("\nDataFrame with missing values (None/NaN):")
# print(df_missing)
# print("\nData types:")
# print(df_missing.dtypes)

# # Method 1: Detect missing values using isnull()
# print("\n--- DETECTING NULL VALUES USING isnull() ---")
# print("\nNull values in DataFrame:")
# print(df_missing.isnull())

# print("\nCount of null values per column:")
# print(df_missing.isnull().sum())

# print("\nTotal null values in DataFrame:")
# print(df_missing.isnull().sum().sum())

# # Check for null in specific column
# print("\nNull values in 'Age' column:")
# print(df_missing['Age'].isnull())

# # Method 2: Detect non-null values using notnull()
# print("\n--- DETECTING NON-NULL VALUES USING notnull() ---")
# print("\nNon-null values in DataFrame:")
# print(df_missing.notnull())

# print("\nCount of non-null values per column:")
# print(df_missing.notnull().sum())

# # Method 3: Filtering rows with missing values
# print("\n--- FILTERING ROWS WITH MISSING VALUES ---")

# # Keep only rows with NO missing values
# print("\nRows with NO missing values:")
# print(df_missing[df_missing.notnull().all(axis=1)])  # axis=1 checks all columns

# # Keep only rows where specific column is not null
# print("\nRows where 'Age' is not null:")
# print(df_missing[df_missing['Age'].notnull()])

# # Method 4: Drop missing values using dropna()
# print("\n--- DROPPING MISSING VALUES USING dropna() ---")

# # Drop rows with ANY missing value
# print("\nAfter dropna() - removes ALL rows with any NaN:")
# df_dropped_any = df_missing.dropna()
# print(df_dropped_any)

# # Drop rows only if ALL values are NaN (rare case)
# print("\nAfter dropna(how='all') - removes only if ALL values are NaN:")
# df_dropped_all = df_missing.dropna(how='all')
# print(df_dropped_all)

# # Drop rows where specific column has NaN
# print("\nAfter dropna(subset=['Age']) - removes rows where Age is NaN:")
# df_dropped_subset = df_missing.dropna(subset=['Age'])
# print(df_dropped_subset)

# # Drop missing values in multiple specific columns
# print("\nAfter dropna(subset=['Age', 'Salary']) - removes rows where Age OR Salary is NaN:")
# df_dropped_multi = df_missing.dropna(subset=['Age', 'Salary'])
# print(df_dropped_multi)

# # Drop columns with missing values (axis=1)
# print("\nAfter dropna(axis=1) - removes columns with ANY NaN:")
# df_dropped_columns = df_missing.dropna(axis=1)
# print(df_dropped_columns)

# # Drop columns that have missing values greater than threshold
# print("\nAfter dropna(thresh=3) - keeps only rows with at least 3 non-null values:")
# df_thresh = df_missing.dropna(thresh=3)
# print(df_thresh)

# print("\n" + "="*50)

# # ===== FILLING MISSING VALUES =====
# print("\n--- FILLING MISSING VALUES ---")

# print("\nOriginal DataFrame with missing values:")
# print(df_missing)

# # Method 1: Fill with a specific value
# print("\n--- FILL WITH SPECIFIC VALUE ---")
# df_fill_value = df_missing.copy()
# df_fill_value = df_fill_value.fillna(0)  # Fill all NaN with 0
# print("After fillna(0) - fills all NaN with 0:")
# print(df_fill_value)

# # Fill specific column with value
# print("\nFilling only 'Age' column with value 30:")
# df_fill_col = df_missing.copy()
# df_fill_col['Age'] = df_fill_col['Age'].fillna(30)
# print(df_fill_col)

# # Fill different columns with different values using dictionary
# print("\nFilling different columns with different values:")
# df_fill_dict = df_missing.copy()
# df_fill_dict = df_fill_dict.fillna({'Age': 35, 'Salary': 65000, 'Name': 'Unknown'})
# print(df_fill_dict)

# # Method 2: Forward Fill (ffill) - propagate last valid value forward
# print("\n--- FORWARD FILL (ffill/pad) ---")
# df_ffill = df_missing.copy()
# print("After fillna(method='ffill') - fills NaN with previous row's value:")
# df_ffill = df_ffill.ffill()     # fillna(method = 'ffill') is the wrong syntax now df.ffill() is the correct one
# print(df_ffill)

# # Forward fill only specific column
# print("\nForward fill only 'Age' column:")
# df_ffill_col = df_missing.copy()
# df_ffill_col['Age'] = df_ffill_col['Age'].ffill(axis= 0)
# print(df_ffill_col)

# # IMPORTANT: In newer pandas (2.0+), use method parameter differently or use shift()
# # df_ffill = df_missing.ffill()  # Alternative syntax

# # Method 3: Backward Fill (bfill) - propagate next valid value backward
# print("\n--- BACKWARD FILL (bfill/backfill) ---")
# df_bfill = df_missing.copy()
# print("After bfill() - fills NaN with next row's value:")
# df_bfill = df_bfill.bfill()

# # Method 4: Fill with mean, median, or other statistics
# print("\n--- FILL WITH STATISTICAL VALUES ---")
# df_stat = df_missing.copy()
# print("Original Age column:", df_stat['Age'].tolist())

# # Fill with mean
# mean_age = df_missing['Age'].mean()
# print(f"\nMean of Age: {mean_age}")
# df_stat['Age'] = df_stat['Age'].fillna(mean_age)
# print("After filling Age with mean:")
# print(df_stat['Age'].tolist())

# # Fill with median
# print("\nFilling Salary with median:")
# df_stat2 = df_missing.copy()
# median_salary = df_missing['Salary'].median()
# print(f"Median of Salary: {median_salary}")
# df_stat2['Salary'] = df_stat2['Salary'].fillna(median_salary)
# print(df_stat2)

# # Method 5: Interpolate - estimate missing values based on surrounding values
# print("\n--- INTERPOLATE ---")
# df_interp = pd.DataFrame({
#     'Value': [10, None, None, 40, None, 60]
# })
# print("Original DataFrame with gaps:")
# print(df_interp)

# print("\nAfter interpolate() - linear interpolation:")
# df_interp_linear = df_interp.interpolate()
# print(df_interp_linear)

# # Different interpolation methods
# print("\nInterpolate with method='ffill' (forward fill):")
# df_interp_pad = df_interp.copy().ffill()
# print(df_interp_pad)

# # IMPORTANT: interpolate() is useful for time series data
# df_time_series = pd.DataFrame({
#     'Date': pd.date_range('2023-01-01', periods=6),
#     'Value': [100, None, None, 400, None, 600]
# })
# print("\nTime series data with missing values:")
# print(df_time_series)

# print("\nAfter interpolate() on time series:")
# df_time_interp = df_time_series.copy()
# df_time_interp['Value'] = df_time_interp['Value'].interpolate()
# print(df_time_interp)

# # Method 6: Fill with limit parameter (limit number of fills)
# print("\n--- FILL WITH LIMIT ---")
# df_limit = df_missing.copy()
# print("After fillna(0, limit=2) - fills only first 2 NaN values per column:")
# df_limit = df_limit.fillna(0, limit=2)
# print(df_limit)

# # Method 7: In-place filling (modifies original DataFrame)
# print("\n--- IN-PLACE FILLING ---")     
# df_inplace = df_missing.copy()
# print("Before inplace fillna:")
# print(df_inplace)
# s = df_inplace.fillna("999")
# print("\nAfter fillna(999, inplace=True):")
# print(s)

# print("\n" + "="*50)

# # ===== DETECTING DUPLICATES =====
# print("\n--- DETECTING DUPLICATES ---")

# # Create DataFrame with duplicate rows
# df_dupes = pd.DataFrame({
#     'Name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob', 'David'],
#     'Age': [25, 30, 25, 35, 30, 40],
#     'City': ['NYC', 'LA', 'NYC', 'Chicago', 'LA', 'Boston']
# })

# print("\nDataFrame with duplicate rows:")
# print(df_dupes)

# # Method 1: Detect duplicates using duplicated()
# print("\n--- DETECTING DUPLICATES USING duplicated() ---")
# print("\nDuplicate rows (returns boolean):")
# print(df_dupes.duplicated())

# # IMPORTANT: duplicated() marks the SECOND and subsequent occurrences as True
# print("\nDuplicate row details:")
# print("Row 0 (Alice): False - First occurrence")
# print("Row 1 (Bob): False - First occurrence")
# print("Row 2 (Alice): True - DUPLICATE of row 0")
# print("Row 3 (Charlie): False - First occurrence")
# print("Row 4 (Bob): True - DUPLICATE of row 1")
# print("Row 5 (David): False - First occurrence")

# # Get only duplicate rows
# print("\nShow only duplicate rows:")
# print(df_dupes[df_dupes.duplicated()])

# # Show all occurrences of duplicates (including first)
# print("\nShow all duplicates (including first occurrence):")
# print(df_dupes[df_dupes.duplicated(keep=False)])

# # Detect duplicates considering only specific columns
# print("\n--- DETECT DUPLICATES IN SPECIFIC COLUMNS ---")
# print("Duplicates based on 'Name' column only:")
# print(df_dupes.duplicated(subset=['Name']))

# print("\nRows with duplicate names:")
# print(df_dupes[df_dupes.duplicated(subset=['Name'], keep=False)])

# # Keep first occurrence
# print("\nDuplicates with keep='first' (mark duplicates after first):")
# print(df_dupes.duplicated(keep='first'))

# # Keep last occurrence
# print("\nDuplicates with keep='last' (mark duplicates before last):")
# print(df_dupes.duplicated(keep='last'))

# # Keep none (mark all duplicates)
# print("\nDuplicates with keep=False (mark ALL duplicates):")
# print(df_dupes.duplicated(keep=False))

# # Method 2: Drop duplicates using drop_duplicates()
# print("\n--- DROP DUPLICATES USING drop_duplicates() ---")
# print("After drop_duplicates() - removes all duplicate rows (keeps first):")
# df_no_dupes = df_dupes.drop_duplicates()
# print(df_no_dupes)

# # Drop duplicates keeping last occurrence
# print("\nAfter drop_duplicates(keep='last') - keeps last occurrence:")
# df_keep_last = df_dupes.drop_duplicates(keep='last')
# print(df_keep_last)

# # Drop duplicates on specific columns
# print("\nAfter drop_duplicates(subset=['Name']) - based on Name column only:")
# df_subset_dupes = df_dupes.drop_duplicates(subset=['Name'])
# print(df_subset_dupes)

# # Drop duplicates on multiple columns
# print("\nAfter drop_duplicates(subset=['Name', 'Age']):")
# df_multi_dupes = df_dupes.drop_duplicates(subset=['Name', 'Age'])
# print(df_multi_dupes)

# # In-place drop duplicates
# print("\nUsing drop_duplicates(inplace=True) - modifies original:")
# df_temp_dupes = df_dupes.copy()
# df_temp_dupes.drop_duplicates(inplace=True)
# print(df_temp_dupes)

# print("\n" + "="*50)

# # ===== AGGREGATION & GROUPING =====
# print("\n--- AGGREGATION & GROUPING ---")

# # Create sample data for grouping
# df_sales = pd.DataFrame({
#     'Department': ['IT', 'HR', 'IT', 'Finance', 'HR', 'Finance', 'IT', 'HR'],
#     'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Henry'],
#     'Salary': [50000, 60000, 55000, 75000, 62000, 80000, 58000, 61000],
#     'Bonus': [5000, 6000, 5500, 8000, 6200, 8500, 5800, 6100]
# })

# print("\nOriginal Sales DataFrame:")
# print(df_sales)

# # Method 1: Basic groupby with sum()
# print("\n--- BASIC GROUPBY WITH SUM() ---")
# group_sum = df_sales.groupby('Department')['Salary'].sum()
# print("Total salary by department:")
# print(group_sum)

# # IMPORTANT: groupby() returns a GroupBy object, need aggregation function to get result

# # Method 2: Groupby with mean()
# print("\n--- GROUPBY WITH MEAN() ---")
# group_mean = df_sales.groupby('Department')['Salary'].mean()
# print("Average salary by department:")
# print(group_mean)

# # Method 3: Groupby with min()
# print("\n--- GROUPBY WITH MIN() ---")
# group_min = df_sales.groupby('Department')['Salary'].min()
# print("Minimum salary by department:")
# print(group_min)

# # Method 4: Groupby with max()
# print("\n--- GROUPBY WITH MAX() ---")
# group_max = df_sales.groupby('Department')['Salary'].max()
# print("Maximum salary by department:")
# print(group_max)

# # Method 5: Groupby with count()
# print("\n--- GROUPBY WITH COUNT() ---")
# group_count = df_sales.groupby('Department')['Name'].count()
# print("Number of employees per department:")
# print(group_count)

# # Alternative: Use size() for counting
# print("\nUsing size() for count:")
# group_size = df_sales.groupby('Department').size()
# print(group_size)

# # Method 6: Multiple aggregations at once
# print("\n--- MULTIPLE AGGREGATIONS ---")
# group_agg = df_sales.groupby('Department')['Salary'].agg(['sum', 'mean', 'min', 'max', 'count'])
# print("Multiple statistics by department:")
# print(group_agg)

# # Method 7: Aggregate multiple columns
# print("\n--- AGGREGATE MULTIPLE COLUMNS ---")
# group_multi = df_sales.groupby('Department')[['Salary', 'Bonus']].sum()
# print("Total salary and bonus by department:")
# print(group_multi)

# # Method 8: Aggregate different columns differently
# print("\n--- DIFFERENT FUNCTIONS FOR DIFFERENT COLUMNS ---")
# group_dict_agg = df_sales.groupby('Department').agg({
#     'Salary': 'sum',
#     'Bonus': 'mean',
#     'Name': 'count'
# })
# print("Sum of salary, mean of bonus, count of employees by department:")
# print(group_dict_agg)

# # Method 9: Groupby on multiple columns
# print("\n--- GROUPBY ON MULTIPLE COLUMNS ---")
# # First, add a gender column for demonstration
# df_sales['Gender'] = ['F', 'M', 'M', 'M', 'F', 'M', 'F', 'M']
# group_multi_col = df_sales.groupby(['Department', 'Gender'])['Salary'].sum()
# print("Total salary by department and gender:")
# print(group_multi_col)

# # Method 10: Reset index to get groupby result as regular DataFrame
# print("\n--- RESET INDEX ---")
# group_reset = df_sales.groupby('Department')['Salary'].sum().reset_index()
# group_reset.columns = ['Department', 'Total_Salary']
# print("Groupby result as DataFrame with reset index:")
# print(group_reset)

# # Method 11: Apply custom function in groupby
# print("\n--- APPLY CUSTOM FUNCTION ---")
# def salary_range(group):
#     return group.max() - group.min()

# group_custom = df_sales.groupby('Department')['Salary'].apply(salary_range)
# print("Salary range (max - min) by department:")
# print(group_custom)

# # Method 12: Get group as separate DataFrames
# print("\n--- ITERATE THROUGH GROUPS ---")
# print("Iterating through groups:")
# for dept, group in df_sales.groupby('Department'):
#     print(f"\nDepartment: {dept}")
#     print(group)

# # Method 13: Filter groups
# print("\n--- FILTER GROUPS ---")
# # Get groups with average salary > 60000
# group_filter = df_sales.groupby('Department').filter(lambda x: x['Salary'].mean() > 60000)
# print("Departments with average salary > 60000:")
# print(group_filter)

# # Method 14: Transform - apply function to each group and return same-shaped DataFrame
# print("\n--- TRANSFORM ---")
# df_sales_copy = df_sales.copy()
# df_sales_copy['Salary_Diff'] = df_sales.groupby('Department')['Salary'].transform(lambda x: x - x.mean())
# print("Salary difference from department average:")
# print(df_sales_copy[['Name', 'Department', 'Salary', 'Salary_Diff']])

# # IMPORTANT: Understanding groupby():
# # - groupby() returns a GroupBy object (not a DataFrame)
# # - Must apply aggregation function: .sum(), .mean(), .min(), .max(), .count(), etc.
# # - Can chain multiple operations
# # - Use .reset_index() to convert result back to DataFrame
# # - Use .agg() for multiple aggregations
# # - Use .apply() for custom functions
# # - Use .transform() to return same-shaped data

# print("\n" + "="*50)

# # ===== AGGREGATE() METHOD =====
# print("\n--- AGGREGATE() METHOD ---")

# # Create sample DataFrame
# df_products = pd.DataFrame({
#     'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones'],
#     'Cost': [800, 25, 45, 300, 150],
#     'Price': [1200, 40, 70, 450, 250],
#     'Stock': [15, 150, 120, 45, 80]
# })

# print("\nOriginal DataFrame:")
# print(df_products)

# # Method 1: Single aggregation function
# print("\n--- SINGLE AGGREGATION FUNCTION ---")
# print("Maximum cost:")
# print(df_products['Cost'].aggregate('max'))

# print("\nMinimum cost:")
# print(df_products['Cost'].aggregate('min'))

# # Method 2: Multiple aggregation functions (returns Series)
# print("\n--- MULTIPLE AGGREGATION FUNCTIONS ---")
# print("Cost aggregation (max, min, mean, sum):")
# result = df_products['Cost'].aggregate(['max', 'min', 'mean', 'sum'])
# print(result)
# print(f"Type: {type(result)}")  # Returns Series

# # Method 3: Aggregate on multiple columns
# print("\n--- AGGREGATE ON MULTIPLE COLUMNS ---")
# print("Multiple columns with multiple functions:")
# result_multi = df_products[['Cost', 'Price', 'Stock']].aggregate(['max', 'min', 'mean'])
# print(result_multi)

# # Method 4: Different functions for different columns using dictionary
# print("\n--- DIFFERENT FUNCTIONS FOR DIFFERENT COLUMNS ---")
# result_dict = df_products.aggregate({
#     'Cost': ['max', 'min'],
#     'Price': 'mean',
#     'Stock': 'sum'
# })
# print(result_dict)

# # Method 5: Using named aggregation (more readable) - pandas 0.25+
# print("\n--- NAMED AGGREGATION ---")
# result_named = df_products.aggregate(
#     max_cost=('Cost', 'max'),
#     min_cost=('Cost', 'min'),
#     avg_price=('Price', 'mean'),
#     total_stock=('Stock', 'sum')
# )
# print(result_named)

# # Method 6: Using custom function with aggregate
# print("\n--- CUSTOM FUNCTION WITH AGGREGATE ---")
# def price_range(x):
#     return x.max() - x.min()

# result_custom = df_products['Price'].aggregate(price_range)
# print(f"Price range (max - min): {result_custom}")

# # Method 7: Multiple custom functions
# print("\n--- MULTIPLE CUSTOM FUNCTIONS ---")
# result_custom_multi = df_products['Cost'].agg({
#     'maximum': 'max',
#     'minimum': 'min',
#     'range': lambda x: x.max() - x.min(),
#     'variance': 'var'
# })
# print(result_custom_multi)

# # IMPORTANT: Difference between aggregate() and other methods:
# # - aggregate() / agg() - flexible, works on Series and DataFrames
# # - Can use string names: 'max', 'min', 'mean', 'sum', 'std', 'var', 'count', 'median', etc.
# # - Can use functions: np.max, lambda functions, custom functions
# # - Can specify different functions for different columns
# # - Returns Series or DataFrame depending on input

# print("\n" + "="*50)

# # ===== SORTING DATAFRAME =====
# print("\n--- SORTING DATAFRAME ---")

# # Create sample DataFrame
# df_sort = pd.DataFrame({
#     'Name': ['Charlie', 'Alice', 'Bob', 'David', 'Eve'],
#     'Age': [35, 25, 30, 28, 32],
#     'Salary': [75000, 50000, 60000, 55000, 65000],
#     'Department': ['IT', 'HR', 'Finance', 'IT', 'HR']
# })

# print("\nOriginal DataFrame:")
# print(df_sort)

# # Method 1: Sort by single column (ascending by default)
# print("\n--- SORT BY SINGLE COLUMN ---")
# print("Sort by Age (ascending):")
# sorted_asc = df_sort.sort_values('Age')
# print(sorted_asc)

# # Sort descending
# print("\nSort by Age (descending):")
# sorted_desc = df_sort.sort_values('Age', ascending=False)
# print(sorted_desc)

# # Method 2: Sort by multiple columns
# print("\n--- SORT BY MULTIPLE COLUMNS ---")
# print("Sort by Department (asc) then Salary (desc):")
# sorted_multi = df_sort.sort_values(['Department', 'Salary'], ascending=[True, False])
# print(sorted_multi)

# # Method 3: Sort by column with missing values (NaN)
# print("\n--- SORT WITH MISSING VALUES ---")
# df_with_nan = pd.DataFrame({
#     'Name': ['Alice', 'Bob', 'Charlie', 'David'],
#     'Score': [85, None, 92, 78]
# })
# print("DataFrame with NaN values:")
# print(df_with_nan)

# print("\nSort by Score (NaN values at end by default):")
# sorted_nan = df_with_nan.sort_values('Score', na_position='last')
# print(sorted_nan)

# print("\nSort by Score (NaN values at start):")
# sorted_nan_first = df_with_nan.sort_values('Score', na_position='first')
# print(sorted_nan_first)

# # Method 4: Sort by index
# print("\n--- SORT BY INDEX ---")
# df_index_sort = df_sort.copy()
# df_index_sort.index = ['E', 'A', 'C', 'B', 'D']
# print("DataFrame with custom index:")
# print(df_index_sort)

# print("\nSort by index (ascending):")
# sorted_index = df_index_sort.sort_index()
# print(sorted_index)

# print("\nSort by index (descending):")
# sorted_index_desc = df_index_sort.sort_index(ascending=False)
# print(sorted_index_desc)

# # Method 5: In-place sorting (modifies original DataFrame)
# print("\n--- IN-PLACE SORTING ---")
# df_inplace = df_sort.copy()
# print("Before inplace sort:")
# print(df_inplace)
# df_inplace.sort_values('Age', inplace=True)
# print("\nAfter sort_values(inplace=True):")
# print(df_inplace)

# # Method 6: Sort with custom comparator (using key parameter)
# print("\n--- SORT WITH CUSTOM KEY FUNCTION ---")
# print("Sort Name column by string length:")
# sorted_key = df_sort.sort_values('Name', key=lambda x: x.str.len())
# print(sorted_key)

# # IMPORTANT: Sorting Rules:
# # - sort_values() - sorts by column values
# # - sort_index() - sorts by row index
# # - ascending=True (default), ascending=False for descending
# # - For multiple columns, use list of column names and list of ascending values
# # - na_position='last' (default) or 'first' for NaN handling
# # - inplace=True modifies original, False (default) returns new DataFrame
# # - Sorting returns NEW DataFrame by default, doesn't modify original

# print("\n" + "="*50)

# # ===== INDEX MANIPULATION =====
# print("\n--- INDEX MANIPULATION ---")

# # Create sample DataFrame
# df_index = pd.DataFrame({
#     'Name': ['Alice', 'Bob', 'Charlie', 'David'],
#     'Age': [25, 30, 35, 40],
#     'City': ['NYC', 'LA', 'Chicago', 'Boston']
# })

# print("\nOriginal DataFrame with default index (0, 1, 2, 3):")
# print(df_index)

# # Method 1: Set a column as index
# print("\n--- SET COLUMN AS INDEX ---")
# df_set_index = df_index.set_index('Name')
# print("After set_index('Name'):")
# print(df_set_index)

# # Set multiple columns as index (MultiIndex)
# print("\n--- MULTIINDEX ---")
# df_multi_index = pd.DataFrame({
#     'Department': ['IT', 'IT', 'HR', 'HR'],
#     'Employee': ['Alice', 'Bob', 'Charlie', 'David'],
#     'Salary': [50000, 60000, 55000, 65000]
# })
# df_multi = df_multi_index.set_index(['Department', 'Employee'])
# print("After set_index(['Department', 'Employee']):")
# print(df_multi)

# # Method 2: Reset index (convert index back to column)
# print("\n--- RESET INDEX ---")
# df_reset = df_set_index.reset_index()
# print("After reset_index():")
# print(df_reset)

# # Reset index but drop it (don't create column)
# print("\nAfter reset_index(drop=True) - index removed, not converted to column:")
# df_reset_drop = df_set_index.reset_index(drop=True)
# print(df_reset_drop)

# # Method 3: Rename index
# print("\n--- RENAME INDEX ---")
# df_rename_idx = df_index.copy()
# df_rename_idx.index = ['Row_A', 'Row_B', 'Row_C', 'Row_D']
# print("After manually setting index:")
# print(df_rename_idx)

# # Using rename method for index
# df_renamed = df_index.rename(index={0: 'First', 1: 'Second', 2: 'Third', 3: 'Fourth'})
# print("\nAfter rename() with index mapping:")
# print(df_renamed)

# # Method 4: Create custom range index
# print("\n--- CREATE CUSTOM INDEX ---")
# df_custom = df_index.copy()
# df_custom.index = pd.RangeIndex(start=1, stop=5)  # Index from 1 to 4
# print("After setting RangeIndex(1, 5):")
# print(df_custom)

# # Method 5: Using Index with custom values
# print("\n--- SET INDEX WITH LIST ---")
# df_list_idx = df_index.copy()
# df_list_idx.index = ['a', 'b', 'c', 'd']
# print("After setting index with list:")
# print(df_list_idx)

# # Method 6: Reindex - reorder rows by new index
# print("\n--- REINDEX ---")
# df_reindex = df_index.copy()
# df_reindex = df_reindex.reindex([3, 1, 0, 2])  # Reorder rows
# print("After reindex([3, 1, 0, 2]) - reorder by position:")
# print(df_reindex)

# # Reindex with new index values (creates NaN for missing)
# print("\nReindex with new values (unknown indices become NaN):")
# df_reindex_new = df_index.reindex([0, 1, 5, 10])
# print(df_reindex_new)

# # Method 7: Sort by index
# print("\n--- SORT BY INDEX ---")
# df_sort_idx = df_list_idx.copy()
# print("Sort by index:")
# print(df_sort_idx.sort_index())

# # IMPORTANT: Index Rules:
# # - Every DataFrame has an index (row labels)
# # - set_index() converts column to index
# # - reset_index() converts index back to column
# # - Can have MultiIndex (multiple columns as index)
# # - Index must be unique (generally, but not enforced)
# # - Use index for fast lookup operations

# print("\n" + "="*50)

# # ===== PIVOT TABLES =====
# print("\n--- PIVOT TABLES ---")

# # Create sample data for pivot
# df_pivot = pd.DataFrame({
#     'Date': ['2023-01', '2023-01', '2023-01', '2023-02', '2023-02', '2023-02'],
#     'Region': ['North', 'South',    'East',     'North',    'South', 'East'],
#     'Product': ['A',    'A',        'B',        'A',        'B',        'B'],
#     'Sales': [100,      150,        200,        120,        180,        220]
# })

# print("\nOriginal DataFrame:")
# print(df_pivot)

# # Method 1: Basic pivot table
# print("\n--- BASIC PIVOT TABLE ---")
# # Syntax: df.pivot_table(values='column_to_aggregate', index='rows', columns='new columns', aggfunc='function') This is the method to make a pivot table 
# # Syntax: pd.pivot_table(DataFrame, values= 'column_to_aggregate' , index= ''rows', columns= 'new columns', aggfunc= 'function') This is the function to make pivot table
# pivot_basic = df_pivot.pivot_table(
#     values='Sales',
#     index='Region',
#     columns='Product',
#     aggfunc='sum'
# )
# print("Pivot table (Region as rows, Product as columns, sum of Sales):")
# print(pivot_basic)

# # Method 2: Pivot table with different aggregation function
# print("\n--- PIVOT TABLE WITH MEAN ---")
# pivot_mean = df_pivot.pivot_table(
#     values='Sales',
#     index='Region',
#     columns='Date',
#     aggfunc='mean'
# )
# print("Pivot table (mean of Sales):")
# print(pivot_mean)

# # Method 3: Pivot table with multiple aggregation functions
# print("\n--- PIVOT TABLE WITH MULTIPLE AGGREGATIONS ---")
# pivot_multi_agg = df_pivot.pivot_table(
#     values='Sales',
#     index='Region',
#     columns='Product',
#     aggfunc=['sum', 'mean', 'count']
# )
# print("Pivot table (sum, mean, count of Sales):")
# print(pivot_multi_agg)

# # Method 4: Pivot table with multiple values columns
# print("\n--- PIVOT TABLE WITH MULTIPLE VALUES AND AGGREGATE FUNCTIONS ---")
# df_pivot_ext = df_pivot.copy()
# df_pivot_ext['Profit'] = [20, 30, 50, 24, 45, 55]
# pivot_multi_val = df_pivot_ext.pivot_table(
#     values=['Sales', 'Profit'],
#     index='Region',
#     columns='Product',
#     aggfunc=['sum','mean']
# )
# print("Pivot table (multiple value columns and aggregate functions):")
# print(pivot_multi_val)

# # Method 5: Pivot table with margins (total rows/columns)
# print("\n--- PIVOT TABLE WITH MARGINS (TOTALS) ---")
# pivot_margins = df_pivot.pivot_table(
#     values='Sales',
#     index='Region',
#     columns='Product',
#     aggfunc='sum',
#     margins=True  # Adds 'All' row and column with totals
# )
# print("Pivot table with margins (totals):")
# print(pivot_margins)

# # IMPORTANT: Pivot Table Rules:
# # - pivot_table() is more flexible than pivot()
# # - Handles duplicate index-column combinations with aggfunc
# # - Can specify multiple aggregation functions
# # - margins=True adds total rows and columns
# # - fill_value parameter replaces NaN with specific value
# # - aggfunc default is 'mean'

# print("\n" + "="*50)

# # ===== CROSS TABULATION (CROSSTAB) =====
# print("\n--- CROSS TABULATION (CROSSTAB) ---")

# # Create sample data
# df_crosstab = pd.DataFrame({
#     'Gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M'],
#     'Product': ['A', 'A', 'B', 'A', 'A', 'B', 'A', 'B', 'A'],
#     'Sales': [100, 150, 200, 120, 180, 220, 110, 140 , 123]
# })

# print("\nOriginal DataFrame:")
# print(df_crosstab)

# # Method 1: Basic crosstab (counts)
# print("\n--- BASIC CROSSTAB (COUNTS) ---")
# # Syntax: pd.crosstab(index, columns)
# crosstab_basic = pd.crosstab(df_crosstab['Gender'], df_crosstab['Product'])
# print("Crosstab (count of occurrences):")
# print(crosstab_basic)

# # Method 2: Crosstab with values and aggregation
# print("\n--- CROSSTAB WITH VALUES AND AGGREGATION ---")
# # Syntax: pd.crosstab(index, columns, values=aggcolumn, aggfunc='function')
# crosstab_sum = pd.crosstab(
#     df_crosstab['Gender'],
#     df_crosstab['Product'],
#     values=df_crosstab['Sales'],
#     aggfunc='sum'
# )
# print("Crosstab (sum of Sales by Gender and Product):")
# print(crosstab_sum)

# # Method 3: Crosstab with mean
# print("\n--- CROSSTAB WITH MEAN ---")
# crosstab_mean = pd.crosstab(
#     df_crosstab['Gender'],
#     df_crosstab['Product'],
#     values=df_crosstab['Sales'],
#     aggfunc='mean'
# )
# print("Crosstab (mean of Sales):")
# print(crosstab_mean)

# # Method 4: Crosstab with margins
# print("\n--- CROSSTAB WITH MARGINS ---")
# crosstab_margins = pd.crosstab(
#     df_crosstab['Gender'],
#     df_crosstab['Product'],
#     values=df_crosstab['Sales'],
#     aggfunc='sum',
#     margins=True  # Adds row and column totals
# )
# print("Crosstab with margins (totals):")
# print(crosstab_margins)

# # Method 5: Crosstab with normalize (percentages)
# print("\n--- CROSSTAB WITH NORMALIZE ---")
# crosstab_norm = pd.crosstab(
#     df_crosstab['Gender'],
#     df_crosstab['Product'],
#     normalize=True  # Shows proportions/percentages
# )
# print("Crosstab normalized (as proportions):")
# print(crosstab_norm)

# # IMPORTANT: Crosstab vs Pivot Table:
# # - crosstab() works with Series/arrays, pivot_table() works with DataFrames
# # - crosstab() is simpler for frequency tables
# # - crosstab() default aggfunc is 'count'
# # - pivot_table() default aggfunc is 'mean'
# # - Both can use margins for totals

# print("\n" + "="*50)

# # ===== MELT (UNPIVOT) =====
# print("\n--- MELT (UNPIVOT) ---")

# # Create sample wide-format DataFrame
# df_wide = pd.DataFrame({
#     'Name': ['Alice', 'Bob', 'Charlie'],
#     'Q1_Sales': [100, 150, 200],
#     'Q2_Sales': [120, 180, 220],
#     'Q3_Sales': [140, 190, 230]
# })

# print("\nOriginal Wide Format DataFrame:")
# print(df_wide)

# # Method 1: Basic melt (convert wide to long format)
# print("\n--- BASIC MELT ---")
# # Syntax: melt(frame, id_vars='column_to_keep', value_vars='columns_to_unpivot', var_name='new_col_name', value_name='new_value_col')
# melted = pd.melt(
#     df_wide,
#     id_vars='Name',  # Column to keep as identifier
#     value_vars=['Q1_Sales', 'Q2_Sales', 'Q3_Sales']  # Columns to unpivot
# )
# print("After melt (default names):")
# print(melted)

# # Method 2: Melt with custom variable and value names
# print("\n--- MELT WITH CUSTOM NAMES ---")
# melted_named = pd.melt(
#     df_wide,
#     id_vars='Name',
#     value_vars=['Q1_Sales', 'Q2_Sales', 'Q3_Sales'],
#     var_name='Quarter',  # Name for the unpivoted columns
#     value_name='Sales'   # Name for the values
# )
# print("After melt with custom names:")
# print(melted_named)

# # Method 3: Melt without specifying value_vars (all other columns melted)
# print("\n--- MELT WITHOUT SPECIFYING value_vars ---")
# melted_auto = pd.melt(df_wide, id_vars='Name')
# print("Melt with automatic value_vars selection:")
# print(melted_auto)

# # Method 4: Melt with multiple id_vars
# print("\n--- MELT WITH MULTIPLE id_vars ---")
# df_multi_id = pd.DataFrame({
#     'Name': ['Alice', 'Bob', 'Alice', 'Bob'],
#     'Department': ['IT', 'HR', 'IT', 'HR'],
#     'Jan': [100, 150, 200, 250],
#     'Feb': [120, 180, 220, 280],
#     'Mar': [140, 190, 240, 300]
# })
# print("DataFrame with multiple id columns:")
# print(df_multi_id)

# melted_multi_id = pd.melt(
#     df_multi_id,
#     id_vars=['Name', 'Department'],
#     var_name='Month',
#     value_name='Sales'
# )
# print("\nAfter melt with multiple id_vars:")
# print(melted_multi_id)

# # IMPORTANT: Melt Rules:
# # - melt() converts wide format to long format (opposite of pivot)
# # - id_vars - columns to keep (identifiers)
# # - value_vars - columns to unpivot (if not specified, all non-id columns used)
# # - var_name - name for new column containing old column names (default: 'variable')
# # - value_name - name for new column containing values (default: 'value')
# # - Useful for preparing data for analysis and visualization

# print("\n" + "="*50)

# # ===== EXPORTING DATA =====
# print("\n--- EXPORTING DATA ---")

# # Create sample DataFrame for exporting
# df_export = pd.DataFrame({
#     'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
#     'Age': [25, 30, 35, 40, 28],
#     'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
#     'Salary': [50000, 60000, 75000, 80000, 55000],
#     'Department': ['IT', 'HR', 'Finance', 'IT', 'Sales']
# })

# print("\nDataFrame to export:")
# print(df_export)

# # ===== EXPORT TO CSV =====
# print("\n--- EXPORT TO CSV ---")

# # Method 1: Basic CSV export
# # Syntax: df.to_csv('filename.csv')
# # IMPORTANT: to_csv() saves the file to disk, doesn't return anything visible
# df_export.to_csv('employees.csv')
# print("Exported to 'employees.csv' (includes index by default)")

# # Method 2: Export without index
# # IMPORTANT: index=False removes the row numbers from the output
# df_export.to_csv('employees_no_index.csv', index=False)
# print("Exported to 'employees_no_index.csv' (without index)")

# # Method 3: Export specific columns only
# df_export[['Name', 'Salary']].to_csv('employee_names_salary.csv', index=False)
# print("Exported specific columns to 'employee_names_salary.csv'")

# # Method 4: Export with custom delimiter
# # IMPORTANT: sep parameter changes the delimiter (default is comma)
# df_export.to_csv('employees_semicolon.csv', sep=';', index=False)
# print("Exported with semicolon delimiter to 'employees_semicolon.csv'")

# # Method 5: Export with custom encoding
# # IMPORTANT: encoding='utf-8' ensures special characters are handled correctly
# df_export.to_csv('employees_utf8.csv', index=False, encoding='utf-8')
# print("Exported with UTF-8 encoding to 'employees_utf8.csv'")

# # Method 6: Export without header
# # IMPORTANT: header=False removes column names from first row
# df_export.to_csv('employees_no_header.csv', index=False, header=False)
# print("Exported without header to 'employees_no_header.csv'")

# # Method 7: Export with specific columns order
# # IMPORTANT: Select columns in desired order before exporting
# df_export[['Name', 'Department', 'Salary', 'Age', 'City']].to_csv('employees_reordered.csv', index=False)
# print("Exported with reordered columns to 'employees_reordered.csv'")

# # ===== EXPORT TO EXCEL =====
# print("\n--- EXPORT TO EXCEL ---")

# # Method 1: Basic Excel export (.xlsx format)
# # Syntax: df.to_excel('filename.xlsx', sheet_name='SheetName')
# # IMPORTANT: Requires openpyxl library installed: pip install openpyxl
# try:
#     df_export.to_excel('employees.xlsx', index=False, sheet_name='Employees')
#     print("Exported to 'employees.xlsx'")
# except Exception as e:
#     print(f"Could not export to Excel: {e}")

# # Method 2: Export multiple DataFrames to different sheets
# # IMPORTANT: Use ExcelWriter to write multiple sheets in same file
# try:
#     with pd.ExcelWriter('employees_multiple_sheets.xlsx') as writer:
#         df_export.to_excel(writer, sheet_name='All Employees', index=False)
#         df_export[df_export['Department'] == 'IT'].to_excel(writer, sheet_name='IT Department', index=False)
#         df_export[df_export['Salary'] > 60000].to_excel(writer, sheet_name='High Earners', index=False)
#     print("Exported multiple sheets to 'employees_multiple_sheets.xlsx'")
# except Exception as e:
#     print(f"Could not export multiple sheets: {e}")

# # Method 3: Export to specific cell location
# # IMPORTANT: startrow and startcol parameters position data in sheet
# try:
#     with pd.ExcelWriter('employees_positioned.xlsx') as writer:
#         df_export.to_excel(writer, sheet_name='Data', startrow=5, startcol=2, index=False)
#     print("Exported starting at row 5, column 2 to 'employees_positioned.xlsx'")
# except Exception as e:
#     print(f"Could not export with positioning: {e}")

# # Method 4: Export with formatting (colors, fonts, etc.)
# # IMPORTANT: Use Styler object for more advanced formatting
# try:
#     styled_df = df_export.style.background_gradient(cmap='viridis', subset=['Salary'])
#     styled_df.to_excel('employees_styled.xlsx', sheet_name='Formatted', index=False)
#     print("Exported styled DataFrame to 'employees_styled.xlsx'")
# except Exception as e:
#     print(f"Could not export styled DataFrame: {e}")

# # ===== EXPORT TO JSON =====
# print("\n--- EXPORT TO JSON ---")

# # Method 1: Basic JSON export (default: list of dictionaries)
# # Syntax: df.to_json('filename.json')
# df_export.to_json('employees.json')
# print("Exported to 'employees.json' (default orient='columns')")

# # Method 2: JSON with different orientation
# # IMPORTANT: orient parameter changes JSON structure
# # 'split' - {index, columns, data}
# # 'records' - [{col1, col2}, {col1, col2}] - MOST COMMON
# # 'index' - {index: {col1, col2}, ...}
# # 'columns' - {col1: {index: value}, ...} - DEFAULT
# # 'values' - just the values as nested lists
# df_export.to_json('employees_records.json', orient='records')
# print("Exported as records (orient='records') to 'employees_records.json'")

# # Method 3: JSON with pretty printing
# # IMPORTANT: indent parameter makes JSON readable (pretty-printed)
# df_export.to_json('employees_pretty.json', orient='records', indent=4)
# print("Exported with indentation to 'employees_pretty.json'")

# # Method 4: JSON without index
# # IMPORTANT: index=False removes index information (works with some orientations)
# df_export.to_json('employees_no_index.json', orient='split', index=False)
# print("Exported without index to 'employees_no_index.json'")

# # ===== EXPORT TO HTML =====
# print("\n--- EXPORT TO HTML ---")

# # Method 1: Basic HTML export
# # Syntax: df.to_html('filename.html')
# df_export.to_html('employees.html')
# print("Exported to 'employees.html'")

# # Method 2: HTML without index
# df_export.to_html('employees_no_index.html', index=False)
# print("Exported to 'employees_no_index.html' (without index)")

# # Method 3: HTML with custom table attributes
# # IMPORTANT: classes parameter adds CSS classes for styling
# df_export.to_html('employees_styled.html', index=False, classes='my_table')
# print("Exported with custom CSS classes to 'employees_styled.html'")

# # Method 4: HTML with border
# # IMPORTANT: border parameter controls table border
# df_export.to_html('employees_bordered.html', index=False, border=1)
# print("Exported with border to 'employees_bordered.html'")

# # Method 5: Get HTML as string (not save to file)
# # IMPORTANT: Use to_html() without filename parameter to get string
# html_string = df_export.to_html(index=False)
# print("\nHTML as string (first 100 characters):")
# print(html_string[:100])

# # ===== EXPORT TO SQL DATABASE =====
# print("\n--- EXPORT TO SQL DATABASE ---")

# # IMPORTANT: Requires SQLAlchemy library: pip install sqlalchemy
# # Also requires database driver: pip install sqlite3 (built-in), pip install pymysql, pip install psycopg2, etc.

# # Method 1: Export to SQLite database (file-based)
# try:
    
#     # Create SQLite engine (creates database if doesn't exist)
#     # Syntax: create_engine('sqlite:///filename.db')
#     engine = create_engine('sqlite:///employees.db')
    
#     # Write DataFrame to SQL table
#     # Syntax: df.to_sql('table_name', con=engine, if_exists='replace', index=False)
#     # IMPORTANT: if_exists parameter options:
#     # 'fail' - raise error if table exists (default)
#     # 'replace' - drop and recreate table
#     # 'append' - add rows to existing table
#     df_export.to_sql('employees', con=engine, if_exists='replace', index=False)
#     print("Exported to SQLite database 'employees.db' table 'employees'")
    
# except ImportError:
#     print("SQLAlchemy not installed. Install with: pip install sqlalchemy")
# except Exception as e:
#     print(f"Could not export to SQL: {e}")

# # ===== EXPORT TO PICKLE =====
# print("\n--- EXPORT TO PICKLE ---")

# # Method 1: Basic pickle export
# # Syntax: df.to_pickle('filename.pkl')
# # IMPORTANT: Pickle format preserves DataFrame exactly (including dtypes)
# df_export.to_pickle('employees.pkl')
# print("Exported to 'employees.pkl' (pickle format preserves exact DataFrame)")

# # Method 2: Read pickle back
# # Note: Showing how to read for reference
# df_from_pickle = pd.read_pickle('employees.pkl')
# print("Successfully read pickle file (dtypes and format preserved)")

# # ===== EXPORT TO TEXT/TXT =====
# print("\n--- EXPORT TO TEXT ---")

# # Method 1: Export to text file (space-separated by default)
# # IMPORTANT: sep parameter controls delimiter
# df_export.to_csv('employees.txt', sep='\t', index=False)
# print("Exported to 'employees.txt' (tab-separated)")

# # Method 2: Export to fixed-width text
# df_export.to_string(open('employees_fixed_width.txt', 'w'))
# print("Exported to 'employees_fixed_width.txt' (fixed-width format)")

# # ===== EXPORT TO CLIPBOARD =====
# print("\n--- EXPORT TO CLIPBOARD ---")

# # Method 1: Copy DataFrame to clipboard
# # IMPORTANT: to_clipboard() copies to system clipboard (Windows/Mac/Linux)
# try:
#     df_export.to_clipboard(index=False)
#     print("Copied DataFrame to clipboard (paste with Ctrl+V)")
# except Exception as e:
#     print(f"Could not copy to clipboard: {e}")

# # ===== EXPORT TO PARQUET =====
# print("\n--- EXPORT TO PARQUET ---")

# # Method 1: Export to Parquet format
# # IMPORTANT: Parquet is efficient for big data, preserves dtypes
# # Requires: pip install pyarrow or pip install fastparquet
# try:
#     df_export.to_parquet('employees.parquet')
#     print("Exported to 'employees.parquet' (efficient columnar format)")
# except ImportError:
#     print("PyArrow not installed. Install with: pip install pyarrow")
# except Exception as e:
#     print(f"Could not export to Parquet: {e}")

# # ===== EXPORT TO OTHER FORMATS =====
# print("\n--- EXPORT TO OTHER FORMATS ---")

# # Method 1: Export to HDF5 (hierarchical data format)
# # IMPORTANT: Requires: pip install tables
# try:
#     df_export.to_hdf('employees.h5', key='df')
#     print("Exported to 'employees.h5' (HDF5 format)")
# except ImportError:
#     print("Tables not installed. Install with: pip install tables")
# except Exception as e:
#     print(f"Could not export to HDF5: {e}")

# # Method 2: Export to Stata format (.dta)
# try:
#     df_export.to_stata('employees.dta')
#     print("Exported to 'employees.dta' (Stata format)")
# except Exception as e:
#     print(f"Could not export to Stata: {e}")

# # ===== GENERAL EXPORT SUMMARY =====
# print("\n--- EXPORT SUMMARY ---")
# print("""
# COMMON EXPORT FORMATS:
# 1. CSV (.csv) - Most portable, universal
# 2. Excel (.xlsx) - Business standard
# 3. JSON (.json) - Web/API friendly
# 4. HTML (.html) - Web display
# 5. SQL (.db) - Database storage
# 6. Pickle (.pkl) - Python preservation
# 7. Parquet (.parquet) - Big data efficient
# 8. HDF5 (.h5) - Scientific data

# IMPORTANT RULES:
# - to_csv(): Universal, human-readable, use index=False to remove row numbers
# - to_excel(): Requires openpyxl, good for business reports
# - to_json(): Best for API integration, use orient='records' for readability
# - to_html(): For web display, add CSS classes for styling
# - to_sql(): For database storage, use if_exists='replace/append'
# - to_pickle(): Preserves exact DataFrame state, Python-specific
# - to_parquet(): Most efficient for large datasets
# - All to_*() methods don't return values, they save to disk/clipboard
# - Use index=False to exclude row index from export (usually desired)
# - header=True (default) keeps column names
# - sep/delimiter parameter controls field separator
# """)

# print("\n" + "="*50)