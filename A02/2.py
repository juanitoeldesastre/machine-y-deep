#1

import numpy as np

array = np.array([[1,2,3],[4,5,6]])
print(array)

#2

a = np.array([1,2,3])
b = np.array([4,5,6])

print("Suma:", a +b)
print("Resta:", a - b)
print("Multiplicacion:", a * b)
print("Division:", a /b)

#3

array = np.array([[10,10,20],[40,50,60],[70,80,90]])

print(array[1, 1])
print(array[0, :])
print(array[:, 1])

#4

array =np.array([1, 2, 3])
print(array + 5)

#5

array = np.array([1, 2, 3, 4, 5])
np.savetxt("mi_array.txt", array)

#6 

import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.show()