#!/usr/bin/env python
# coding: utf-8

# ##  1

# # n*(k*m) times multiply and
# # n*(k*(m-1)addition and resulting matrix will be of order (n*m)

# ## 2

# In[26]:


def matrix_multiply(a, b):
    n = len(matrix_a)
    k = len(matrix_a[0])
    m = len(matrix_b[0])

    result = [[0 for _ in range(m)] for _ in range(n)]

    # Perform matrix multiplication
    for i in range(n):
        for j in range(m):
            for l in range(k):
                result[i][j] += matrix_a[i][l] * matrix_b[l][j]

    return result

matrix_a = [
    [4, 8, 4],
    [2, 7, 7],
    [3, 8, 9],
    [4, 7, 9]
]

matrix_b = [
    [5, 8],
    [8, 10],
    [4, 8]
]

result_list = matrix_multiply(matrix_a, matrix_b)
print(result_list)


# In[27]:


# 2
import numpy as np
matrix_a = [
    [4, 8, 4],
    [2, 7, 7],
    [3, 8, 9],
    [4, 7, 9]
]

matrix_b = [
    [5, 8],
    [8, 10],
    [4, 8]
]

print(np.dot(matrix_a, matrix_b))


# ## numpy will be faster than using list of list and numpy is an inbuild libraray that allow matrix multiplication much faster
# ## than using list of list

# In[38]:


pip install jax


# ## 3

# In[107]:


##3import jax
import jax
import jax.numpy as jnp

def f(x, y):
    return x**2 * y + y**3 * jnp.sin(x)


grad_f = jax.grad(f, argnums=(0, 1))  
x_values = jnp.array([3.0, 8.0, 2.0])
y_values = jnp.array([2.0, 5.0, 3.0])

for x, y in zip(x_values, y_values):
    jax_grad = grad_f(x, y)
    
    analytical_grad_x = 2*x*y + y**3 * jnp.cos(x)
    analytical_grad_y = x**2 + 3*y**2 * jnp.sin(x)
    
    print(f"Evaluating at x={x}, y={y}:")
    print(f"  JAX gradient: df/dx = {jax_grad[0]}, ∂f/∂y = {jax_grad[1]}")
    print(f"  Analytical gradient: df/dx = {analytical_grad_x}, df/dy = {analytical_grad_y}")
    print()


# ## 7

# In[108]:


## 7
import sympy as sp

x, y = sp.symbols('x y')

f = x**2 * y + y**3 * sp.sin(x)


grad_f_x = sp.diff(f, x) 
grad_f_y = sp.diff(f, y)  


print("SymPy Gradient:")
print(f"df/dx = {grad_f_x}")
print(f"df/dy = {grad_f_y}")


grad_f_x_simplified = sp.simplify(grad_f_x)
grad_f_y_simplified = sp.simplify(grad_f_y)

print("\nSimplified Gradient:")
print(f"df/dx = {grad_f_x_simplified}")
print(f"df/dy = {grad_f_y_simplified}")


# ## 8

# In[109]:


## 8
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0.5, 100, 0.5)  

y = x

plt.figure(figsize=(10, 6))
plt.plot(x, y, label="y = x", color="blue")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Plot of y = x")
plt.legend()

plt.grid(True)

plt.show()


# In[110]:


import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0.5, 100, 0.5)  

y = x**2

plt.figure(figsize=(10, 6))
plt.plot(x, y, label="y = x^2", color="blue")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Plot of y = x^2")
plt.legend()

plt.grid(True)

plt.show()


# In[111]:


import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0.5, 100, 0.5)  

y = (x**3)/100

plt.figure(figsize=(10, 6))
plt.plot(x, y, label="y = (x^3)/100", color="blue")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Plot of (x^3)/100")
plt.legend()

plt.grid(True)

plt.show()


# In[112]:


import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0.5, 100, 0.5)  

y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label="y = sin(x)", color="blue")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Plot of y = sin(x)")
plt.legend()

plt.grid(True)

plt.show()


# In[113]:


import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0.5, 100, 0.5)  

y = (np.sin(x))/ x

plt.figure(figsize=(10, 6))
plt.plot(x, y, label="y = sin(x)/x", color="blue")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Plot of y = sin(x)/x")
plt.legend()

plt.grid(True)

plt.show()


# In[114]:


import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0.5, 100, 0.5)  

y = np.log(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label="y = log(x)", color="blue")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Plot of y = log(x)")
plt.legend()

plt.grid(True)

plt.show()


# In[115]:


import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0.5, 100, 0.5)  

y = np.exp(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label="y = exp(x)", color="blue")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Plot of y = exp(x)")
plt.legend()

plt.grid(True)

plt.show()


# ## 10

# In[116]:


## 10
import numpy as np
x = np.random.rand(20, 5)
print(x)


# In[117]:


import pandas as pd
df = pd.DataFrame(x, columns=['a', 'b', 'c', 'd', 'e'])

std_dev = df.std()
column_with_highest_std = std_dev.idxmax()
highest_std_dev = std_dev.max()
print(f"Column with the highest standard deviation: {column_with_highest_std} (Standard Deviation: {highest_std_dev: f})")

means = df.mean(axis = 1)
row_with_lowest_mean = means.idxmin()
lowest_mean = means.min()
print(f"Row with the lowest mean: {row_with_lowest_mean} (Mean: {lowest_mean: f})")


# ## 7

# In[123]:


student_records = {
    2022: {
        'Branch 1': {
            1: {
                'Name': 'Dinesh',
                'Marks': {
                    'Maths': 100,
                    'English': 70,
                    'Science': 85
                }
            },
            2: {
                'Name': 'Sawai',
                'Marks': {
                    'Maths': 90,
                    'English': 80,
                    'Science': 75
                }
            }
        },
        'Branch 2': {
            1: {
                'Name': 'Bhushan',
                'Marks': {
                    'Maths': 95,
                    'English': 85,
                    'Science': 80
                }
            }
        }
    },
    2023: {
        'Branch 1': {
            1: {
                'Name': 'Akash',
                'Marks': {
                    'Maths': 88,
                    'English': 78,
                    'Science': 82
                }
            }
        },
        'Branch 2': {
            1: {
                'Name': 'Ritish',
                'Marks': {
                    'Maths': 92,
                    'English': 81,
                    'Science': 77
                }
            }
        }
    },
    2024: {
        'Branch 1': {
            1: {
                'Name': 'Pritam',
                'Marks': {
                    'Maths': 85,
                    'English': 70,
                    'Science': 90
                }
            }
        },
        'Branch 2': {
            1: {
                'Name': 'Rishabh',
                'Marks': {
                    'Maths': 91,
                    'English': 79,
                    'Science': 88
                }
            }
        }
    },
    2025: {
        'Branch 1': {
            1: {
                'Name': 'Pranav',
                'Marks': {
                    'Maths': 87,
                    'English': 74,
                    'Science': 82
                }
            }
        },
        'Branch 2': {
            1: {
                'Name': 'Rudraksh',
                'Marks': {
                    'Maths': 93,
                    'English': 80,
                    'Science': 85
                }
            }
        }
    }
}

print(student_records)


# ## 12

# In[125]:


import numpy as np
a = np.array([1,2,3,4])
b = np.array([2,3,5])
print(a+b)


# ## Fixed broadcasting

# In[127]:


import numpy as np
a = np.array([1,2,3,4])
b = np.array([2,3,5,6])
print(a+b)


# In[130]:


x = np.array([1, 2, 3]) ## 1*3 matrix
y = np.array([[3], [4], [5]]) ##3*1 matrix
print(x+y)


# In[ ]:




