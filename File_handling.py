print("File Handling in Python")
print("------------------------")
print()

# Opening a file also creates it if it doesn't exist
file = open('example.txt', 'w') # write mode always creates a new file or truncates an existing file if seek is used then it can write at a specific location and not truncate the file
# Writing to a file "w"
file.write('Hello, World!\n')
file.write('This is a sample file.\n')
# Closing the file
file.close()

# location of the file
import os
print("File location:", os.path.abspath('example.txt'))
print()

print("Demonstrating different file modes like r, w, a, r+, w+, a+")
# Reading from a file "r"
print("Reading the file after writing:")
file = open('example.txt', 'r')
content = file.read()
print(content)
# Closing the file
file.close()
print()

# Appending to a file "a"
file = open('example.txt', 'a')
file.write('Appending a new line.\n')
file.close()

# Reading the updated file "r"
print("Reading the file after appending:")
file = open('example.txt', 'r')
content = file.read()
print(content)
file.close()
print()

# Using read and write mode "r+" it allows both reading and writing but first reads the existing content co can't create a new file
print("Using r+ mode to read and write:")
file = open('example.txt', 'r+')
content = file.read()
print(content)
file.write('New content added with r+ mode.')
content = file.read()
print(content)  # This will print an empty string because the cursor is at the end after writing
file.seek(0)  # Move the cursor to the beginning of the file
content = file.read()
print(content)
file.close()
print()

print("Using w+ mode to write and read:") # w+ allows both writing and reading but truncates the file if it exists or creates a new one
file = open('example.txt', 'w+')    # w+ first writes (truncates) then allows reading u can't read what was there before
content = file.read()
print(content)  # This will print an empty string because the file was truncated
file.write('Overwriting with w+ mode.\n')
file.write('This is a sample file.')
content = file.read() # This will still print an empty string because the cursor is at the end after writing
print(content)
file.seek(0)  # Move the cursor to the beginning of the file
content = file.read()
print(content)
file.close()
print()

print("Using seek() to move the cursor:")
file = open('example.txt', 'r')
content = file.read(10)  # Read first 10 characters
print("First 10 characters:", content)
file.seek(0)  # Move the cursor back to the beginning
content = file.read(20)  # Read first 20 characters
print("First 20 characters after seek:", content)
file.close()
print()

print("Using readlines() and writelines():")
file = open('example.txt', 'a')
file.writelines(['\nAdding multiple lines using writelines().\n', 'Second line added.\n'])
file.close()
file = open('example.txt', 'r')
lines = file.readlines()
print("Lines read using readlines():", lines)
for line in lines:
    print(line.strip())
file.close()
print()

print("Using a+ mode to append and read:") # a+ allows both appending and reading, creates the file if it doesn't exist
file = open('example.txt', 'a+')   # a+ first appends then allows reading u can't read what was there before appending
content = file.read()
print(content)  # This will print an empty string because the cursor is at the end when opened in append mode
file.write('\nAppending with a+ mode.')
file.seek(0)  # Move the cursor to the beginning of the file
content = file.read()
print(content)
file.close()
print()

print("Demonstrating binary file modes like rb, wb, ab, rb+, wb+, ab+")
print("Using write binary mode wb and reading in read binary mode rb")
file = open('example.bin', 'wb') # write binary mode does not support string data only bytes like b'example'
file.write(b'Hello, World!')
file.close()

file = open('example.bin', 'rb')
content = file.read()
print(content)
file.close()
print()