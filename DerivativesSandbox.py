import matplotlib.pyplot as plt
import numpy as np

'''here i'm trying to understand how to draw tangent lines for the nonlinear
function so that these lines are representative of the impact that the x has on y;

So essentially i'm calculating the 'slope' of the function which is basically 
the impact itself but since i can't calculate the slope for the nonlinear function
with the big step (2 point for example) because it won't be accurate obviously,
so i'm creating multiple tangent lines with the way smaller step so that i can 
calculate their slopes separately and then combine???
'''


def f(x):
    return 2 * x ** 2


x = np.arange(0, 5, 0.001)
y = f(x)

plt.plot(x, y)

colors = ['k', 'g', 'r', 'b', 'c']


def approximate_tangent_line(x, approximate_derivative):
    return (approximate_derivative * x) + b


for i in range(5):
    p2_delta = 0.0001
    x1 = i
    x2 = x1 + p2_delta

    y1 = f(x1)
    y2 = f(x2)

    print((x1, y1), (x2, y2))

    approximate_derivative = (y2 - y1) / (x2 - x1)
    b = y2 - approximate_derivative * x2

    to_plot = [x1 - 0.9, x1, x1 + 0.9]

    plt.scatter(x1, y1, c=colors[i])
    plt.plot([point for point in to_plot],
             [approximate_tangent_line(point, approximate_derivative)
              for point in to_plot],
             c=colors[i])
    print('Approximate derivative for f(x)', f'where x = {x1} is {approximate_derivative}')

plt.show()
