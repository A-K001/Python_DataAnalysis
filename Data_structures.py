# # Nested Dictionary Example: Storing information about students
# students = {
#  "student1": {"name": "Alice", "age": 20, "major": "Computer Science"},3
#  "student2": {"name": "Bob", "age": 22, "major":"Engineering"},
#  "student3": {"name": "Charlie", "age": 21, "major":"Mathematics"}
# }
# # Accessing information about student2
# print("Name:", students["student2"]["name"])
# print("Age:", students["student2"]["age"])
# print("Major:", students["student2"]["major"])

# # Zipping Two Tuples into a Dictionary
# t1 = (1, 2, 3)
# t2 = (4, 5, 6)
# l1 = [1, 2, 3]
# l2 = [4, 5, 6]
# S1 = {1, 2, 3}
# S2 = {4, 5, 6}
# my_tdict = dict(zip(t1, t2))
# my_ldict = dict(zip(l1, l2))
# my_sdict = dict(zip(S1, S2))
# print("Zipped TDictionary:", my_tdict)
# print("Zipped LDictionary:", my_ldict)
# print("Zipped SDictionary:", my_sdict)

# l1 = [1, 2, 3, 4]
# l2 = ['a', 'b', 'c']
# my_list = list(zip(l1, l2))
# my_tuple = tuple(zip(l1, l2))
# my_set = set(zip(l1, l2))
# print("Zipped Set:", my_set)
# print("Zipped Tuple:", my_tuple)
# print("Zipped List:", my_list)

# nested_list = [[i*j for j in range(3)] for i in range(4)]
# print(nested_list)

# i = 1
# while i < 4:
#     if i == 3: # else will not be printed
#     # if i == 5: # else will be printed
#         break
#     print(i, end=" ")
#     i += 1
# else:
#     print("Else is also printed")

