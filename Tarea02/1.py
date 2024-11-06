def programa(numeros):
    suma = 0
    for numero in numeros:
        if numero % 2 != 0:
            suma += numero
    return suma

n = int(input("Ingrese la cantidad de números: "))

numeros = []

for i in range(n):
    numero = int(input(f"Ingrese el número {i + 1}: "))
    numeros.append(numero)

resultado = programa(numeros)
print("La suma de los números impares es:", resultado)