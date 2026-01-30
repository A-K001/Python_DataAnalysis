def square(x):
    return x ** 2

def cube(x):
    return x ** 3

def factorial(n):
    if n < 0:
        raise ValueError("Negative number")
    if n == 0:
        return 1
    return n * factorial(n - 1)