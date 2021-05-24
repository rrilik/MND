import math
import random

coefficient = [random.randint(1, 10) for _ in range(4)]

x = [[random.randint(0, 20) for _ in range(3)] for _ in range(8)]

y = [coefficient[0] + coefficient[1] * x[i][0] +
     coefficient[2] * x[i][1] + coefficient[3] * x[i][2] for i in range(len(x))]

x0 = [(max([xi[i] for xi in x]) + max([xi[i] for xi in x])) / 2 for i in range(3)]

dx = [max([xi[i] for xi in x]) - x0[i] for i in range(3)]

normalized_x = [[(x[i][j] - x0[j]) / dx[j] for j in range(3)] for i in range(8)]

reference_y = coefficient[0] + coefficient[1] * x0[0] + coefficient[2] * x0[1] + coefficient[3] * x0[2]

min_y = math.inf
min_y_index = 0
for i in range(len(y)):
    element = (y[i] - reference_y) ** 2
    if element < min_y:
        min_y = element
        min_y_index = i

print(f'Список довільно вибраних коефіцієнтів: {coefficient}')
print(f'Матриця планування: {x}')
print(f'Значення функції відгуків для кожної точки плану: {y}')
print(f'Центральні моменти експерименту: {x0}')
print(f'Інтервали зміни фактора: {dx}')
print(f'Матриця нормалізованих значень X: {normalized_x}')
print(f'Значення функції відгуку для нульових рівнів факторів (еталонний Y): {reference_y}')
print(f'Для варіанту #214: min((Y - Yет)²) = {min_y}; точка плану, що задовольняє критерій вибору оптимальності: {x[min_y_index]}')
