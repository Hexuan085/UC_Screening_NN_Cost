import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import random
import csv
from datetime import datetime
import time
import warnings

warnings.filterwarnings("ignore")

# Load the known parameters
from pypsa.linopt import define_variables, get_var
with open('PTDF39.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    # print(rows)
    a_ln = np.array(rows, dtype=float)
    print(a_ln[45][38])

with open('GEN39.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    # print(rows)
    g_index = np.array(rows, dtype=int)
    print(g_index[1][0])

with open('linear_marginal_cost.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    # print(rows)
    C = np.array(rows, dtype=float)
    # print(C)

#  marginal cost
C = C[0][0:10]
print(C)

# Set the number of generated samples
num_samples = 10000

# Topology of used network
num_of_gen = 10
num_buses = 39
num_lines = 46

# Set the constraint coefficients
m = np.ones((1, num_of_gen), dtype=float)
e_right = np.diag(m[0])*5  # Upper bound of generation
e_left = np.diag(m[0])*0  # Lower bound of generation
D = np.diag(m[0])  # Generator coefficient in bounds

b = np.zeros((num_buses, num_of_gen), dtype=float)  # Initial generator coefficient in net demand equation

for g in range(num_of_gen):  # Modify generator coefficient in net demand equation
    b[g_index[g][0]-1][g] = 1

line_flow_limit = np.ones((1, num_lines), dtype=float) * 3.5  # Upper\lower bound of line flow

load_status = -1
num = 0

for i in range(num_samples):

    load_status = -load_status
    load_left = np.ones((1, 20), dtype=float) * 30 / 39
    load_right = np.ones((1, 19), dtype=float) * 30 / 39
    # print(load_left)
    load_random_left = np.random.uniform(0, 0.2, (1, 20))
    load_random_right = load_random_left[0][0:19] + load_random_left[0][19] / 19
    # print(load_random_left)
    load_left = load_left + load_random_left * load_status
    load_right = load_right - load_random_right * load_status
    # print(load_left[0])
    load = list(load_left[0]) + list(load_right[0])
    load = np.array(load)

    x = cp.Variable(num_of_gen)
    u = cp.Variable(num_of_gen, integer=True)
    q = cp.Variable(num_buses)

    # Original UC problem
    # Initialize decision variables
    start_time = time.time()
    cost = cp.sum(C * x)

    constraint = [q == b * x - load.reshape(-1,),
                  cp.sum(q) == 0,
                  e_left * u <= D * x,
                  D * x <= e_right * u,
                  u <= 1, u >= 0,
                  a_ln * q >= -line_flow_limit.reshape(-1, ),
                  line_flow_limit.reshape(-1, ) >= a_ln * q]
    try:
        prob = cp.Problem(cp.Minimize(cost), constraint)
        prob.solve(solver='GLPK_MI')
        if x.value is None:
            # print("No solution!!!!!!")
            # print("Impossible load combination", load)
            print("Load sum", np.sum(load))
            continue
    except KeyError:
        print("KeyError")
        continue

    # print('status of g', u.value)
    # print('generation of g', x.value)
    # print('summation of g', np.sum(x.value))
    # print('summation of l', np.sum(load))

    print('cost', cost.value)

    # Save generated data
    if num == 0:
        load_all=load.reshape(1, -1)
    else:
        load_all=np.concatenate((load_all, load.reshape(1, -1)))
    if num == 0:
        x_val=[]
        u_val=[]
        l_sum=[]
        l_sum.append(np.sum(load))
        x_val.append(x.value)
        u_val.append(u.value)
        line_flow_all = np.dot(a_ln, q.value).reshape(1, -1)
        cost_all=[]
        cost_all.append(prob.value)
    else:
        x_val.append(x.value)
        u_val.append(u.value)
        l_sum.append(np.sum(load))
        line_flow_all = np.vstack((line_flow_all, np.dot(a_ln, q.value).reshape(1, -1)))
        cost_all.append(prob.value)
    num += 1

print(np.max(l_sum))
print(np.min(l_sum))
print(np.max(cost_all))
cost_all = np.array(cost_all, dtype=float).reshape(-1, 1)
print(np.shape(load_all))
print(np.shape(x_val))
print(np.shape(u_val))
# unique_u = np.copy(u_val)
# unique_u = np.array(list(set([tuple(t) for t in unique_u])))
# print(np.shape(unique_u))
print(np.shape(line_flow_all))
print(np.shape(cost_all))
data_all = np.concatenate((load_all, u_val, x_val, line_flow_all, cost_all), axis=1)
data_all = np.round(data_all, 4)
with open('linear_data_all39_UC.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data_all)
# with open('linear_data_all39_pattern_u.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerows(unique_u)
