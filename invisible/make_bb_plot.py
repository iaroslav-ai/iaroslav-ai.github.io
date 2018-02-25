import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Space
from skopt.plots import partial_dependence

np.random.seed(1)

def objective(x):
    v = (np.sin(x[0] ** 2)-0.5)*0.7+x[0]/3.0
    return v

sol = gp_minimize(objective, [Real(-3.0, 3.0)], n_calls=16, n_random_starts=8, noise=0.0)
space = sol.space
model = sol.models[-1]
estimate = lambda x: model.predict(space.transform([[x]]), return_std=True)


x = np.linspace(-3.0, 3.0, num=5000)
y = np.array([objective([v]) for v in x])
yp = np.array([estimate(v) for v in x])

ym = np.array([v[0] for v in yp])
yv = np.array([v[1] for v in yp])

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))

# plot evaluations of the function
plt.scatter(sol.x_iters, sol.func_vals, label='Evaluations')

plt.plot(x, y,"r--", label="Actual function", c='b')
plt.plot(x, ym, label="Function estimate", c='r')

plt.fill(
    np.concatenate([x, x[::-1]]), # x coordinates for fill
    np.concatenate([ym-yv, ym[::-1]+yv[::-1]]), # y coordinates of fill
    alpha=.2, fc="r", ec="None", label='Uncertainty'
)

plt.grid()
plt.legend()
plt.tight_layout()

plt.show()