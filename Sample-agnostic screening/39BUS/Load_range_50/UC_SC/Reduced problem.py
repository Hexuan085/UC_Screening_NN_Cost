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

with open('PTDF39.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    # print(rows)
    a_ln = np.array(rows, dtype=float)
    # print(a_ln[185][117])

with open('GEN39.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    # print(rows)
    g_index = np.array(rows, dtype=int)
    # print(g_index[1][0])

with open('linear_marginal_cost.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    # print(rows)
    C = np.array(rows, dtype=float)
    # print(C)

with open('UC_sta_line_max.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    # print(rows)
    sta_line_max = np.array(rows, dtype=int)
    # print(C)

with open('UC_sta_line_min.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    # print(rows)
    sta_line_min = np.array(rows, dtype=int)
    # print(C)

with open('UC_sta_line_bound_max.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    # print(rows)
    sta_line_bound_max = np.array(rows, dtype=int)
    # print(C)

with open('UC_sta_line_bound_min.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    # print(rows)
    sta_line_bound_min = np.array(rows, dtype=int)
    # print(C)
C = C[0][0:10]

num_samples = 200
num_of_gen = 10
num_buses = 39
num_lines = 46

m = np.ones((1, num_of_gen), dtype=float)
e_right = np.diag(m[0])*5
e_left = np.diag(m[0])*0.5
D = np.diag(m[0])

b = np.zeros((num_buses, num_of_gen), dtype=float)

for g in range(num_of_gen):
    b[g_index[g][0]-1][g] = 1

reduce_max = sta_line_max.shape[1]
reduce_min = sta_line_min.shape[1]
reduce_min_bound = sta_line_bound_min.shape[1]
reduce_max_bound = sta_line_bound_max.shape[1]

line_flow_limit = np.ones((1, num_lines), dtype=float) * 3.5
line_flow_limit_max = np.ones((1, reduce_max), dtype=float) * 3.5
line_flow_limit_min = np.ones((1, reduce_min), dtype=float) * 3.5
line_flow_limit_max_bound = np.ones((1, reduce_max_bound), dtype=float) * 3.5
line_flow_limit_min_bound = np.ones((1, reduce_min_bound), dtype=float) * 3.5

a_ln_max = []
a_ln_min = []
a_ln_min_bound = []
a_ln_max_bound = []

for l_index in sta_line_max:
    a_ln_max.append(a_ln[l_index][:])

for l_index in sta_line_min:
    a_ln_min.append(a_ln[l_index][:])

for l_index in sta_line_bound_min:
    a_ln_min_bound.append(a_ln[l_index][:])

for l_index in sta_line_bound_max:
    a_ln_max_bound.append(a_ln[l_index][:])

print('shape of a_ln', np.shape(a_ln_max), np.shape(a_ln_min), np.shape(a_ln_min_bound), np.shape(a_ln_max_bound))

time_0 = 0
time_1 = 0
time_2 = 0

load_status = -1

for i in range(num_samples):
    load_status = -load_status
    load_left = np.ones((1, 20), dtype=float) * 30 / 39
    load_right = np.ones((1, 19), dtype=float) * 30 / 39
    # print(load_left)
    load_random_left = np.random.uniform(0, 0.38, (1, 20))
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
    cost = cp.sum(C * x)
    constraint = [q == b * x - load.reshape(-1, ),
                  cp.sum(q) == 0,
                  e_left * u <= D * x,
                  D * x <= e_right * u,
                  u <= 1, u >= 0,
                  a_ln * q >= -line_flow_limit.reshape(-1, ),
                  line_flow_limit.reshape(-1, ) >= a_ln * q]
    try:
        prob = cp.Problem(cp.Minimize(cost), constraint)
        start_time = time.time()
        prob.solve(solver='GLPK_MI')
        time_0 = time_0 + time.time() - start_time
        print("Solution time CPLEX of original problem", time.time() - start_time)
        if x.value is None:
            # print("No solution!!!!!!")
            # print("Impossible load combination", load)
            print("Load sum", np.sum(load))
            # continue
    except KeyError:
        print("KeyError")
        # continue
    print('cost', cost.value)

    x_reduce_0 = cp.Variable(num_of_gen)
    u_reduce_0 = cp.Variable(num_of_gen, integer=True)
    q_reduce_0 = cp.Variable(num_buses)
    cost_reduce_0 = cp.sum(C * x_reduce_0)
    constraint_reduce_0 = [q_reduce_0 == b * x_reduce_0 - load.reshape(-1, ),
                  cp.sum(q_reduce_0) == 0,
                  e_left * u_reduce_0 <= D * x_reduce_0,
                  D * x_reduce_0 <= e_right * u_reduce_0,
                  u_reduce_0 <= 1, u_reduce_0 >= 0,
                  a_ln_min[0] * q_reduce_0 >= -line_flow_limit_min.reshape(-1, ),
                  line_flow_limit_max.reshape(-1, ) >= a_ln_max[0] * q_reduce_0]
    try:
        prob = cp.Problem(cp.Minimize(cost_reduce_0), constraint_reduce_0)
        start_time = time.time()
        prob.solve(solver='GLPK_MI')  # cp.CVXOPT
        time_1 = time_1 + time.time() - start_time
        print("Solution time CPLEX of reduced problem 1", time.time() - start_time)
        if x.value is None:
            # print("No solution!!!!!!")
            # print("Impossible load combination", load)
            print("Load sum", np.sum(load))
            # continue
    except KeyError:
        print("KeyError")
        # continue
    print('cost_reduce_1', cost_reduce_0.value)


    x_reduce_1 = cp.Variable(num_of_gen)
    u_reduce_1 = cp.Variable(num_of_gen, integer=True)
    q_reduce_1 = cp.Variable(num_buses)
    cost_reduce_1 = cp.sum(C * x_reduce_1)
    constraint_reduce_1 = [q_reduce_1 == b * x_reduce_1 - load.reshape(-1, ),
                           cp.sum(q_reduce_1) == 0,
                           e_left * u_reduce_1 <= D * x_reduce_1,
                           D * x_reduce_1 <= e_right * u_reduce_1,
                           u_reduce_1 <= 1, u_reduce_0 >= 0,
                           a_ln_min_bound[0] * q_reduce_1 >= -line_flow_limit_min_bound.reshape(-1, ),
                           line_flow_limit_max_bound.reshape(-1, ) >= a_ln_max_bound[0] * q_reduce_1]
    try:
        prob = cp.Problem(cp.Minimize(cost_reduce_1), constraint_reduce_1)
        start_time = time.time()
        prob.solve(solver='GLPK_MI')  # cp.CVXOPT
        time_2 = time_2 + time.time() - start_time
        print("Solution time CPLEX of reduced problem 2", time.time() - start_time)
        if x.value is None:
            # print("No solution!!!!!!")
            # print("Impossible load combination", load)
            print("Load sum", np.sum(load))
            # continue
    except KeyError:
        print("KeyError")
        # continue
    print('cost_reduce_2', cost_reduce_1.value)


print('time_0, time_1, time_2', time_0/time_0, time_1/time_0, time_2/time_0)
