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

with open('PTDF118.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    # print(rows)
    a_ln = np.array(rows, dtype=float)
    print(a_ln[185][117])

with open('GEN118.csv', 'r') as csvfile:
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

num_samples = 2000
num_of_gen = 54
num_buses = 118
num_of_lines = 186

m = np.ones((1, num_of_gen), dtype=float)
e_right = np.diag(m[0])*5
e_left = np.diag(m[0])*0
D = np.diag(m[0])

b = np.zeros((num_buses, num_of_gen), dtype=float)

for g in range(num_of_gen):
    b[g_index[g][0]-1][g] = 1

line_flow_limit = np.ones((1, num_of_lines), dtype=float) * 3.5
line_flow_limit = line_flow_limit[0]
line_flow_limit_l = np.ones((1, num_of_lines-1), dtype=float) * 3.5
line_flow_limit_l = line_flow_limit_l[0]

sta_line_min = []
sta_line_max = []

sta_line_bound_min = []
sta_line_bound_max = []
time_all = []

num_con = np.zeros((1, num_samples), dtype=float)
num_con_bound = np.zeros((1, num_samples), dtype=float)
load_status = -1
num = 0
percentage = 0.5
l_center = 120 / num_buses

for i in range(num_samples):

    load_status = -load_status
    load_left = np.ones((1, 59), dtype=float) * l_center
    load_right = np.ones((1, 59), dtype=float) * l_center
    # print(load_left)
    load_random_left = np.random.uniform(0, percentage * l_center, (1, 59))
    load_random_right = load_random_left[0][0:59]
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

    print('cost', cost.value)

    # print("Current generation", x.value)
    if num == 0:
        load_all = load.reshape(1, -1)
    else:
        load_all = np.concatenate((load_all, load.reshape(1, -1)))
    if num == 0:
        x_val = []
        u_val = []
        l_sum = []
        l_sum.append(np.sum(load))
        x_val.append(x.value)
        u_val.append(u.value)
        # line_flow_all = line_flow.value.reshape(1,-1)
        line_flow_all = np.dot(a_ln, q.value).reshape(1, -1)
        cost_all = []
        cost_all.append(prob.value)

    else:
        x_val.append(x.value)
        u_val.append(u.value)
        l_sum.append(np.sum(load))
        line_flow_all = np.vstack((line_flow_all, np.dot(a_ln, q.value).reshape(1, -1)))
        cost_all.append(prob.value)

    num += 1

l_max_status = np.zeros((line_flow_all.shape[0], num_of_lines), dtype=float)
l_min_status = np.zeros((line_flow_all.shape[0], num_of_lines), dtype=float)
print(l_max_status[0])

line_status_max = [[] for _ in range(line_flow_all.shape[0])]
line_status_min = [[] for _ in range(line_flow_all.shape[0])]

#Record the binding constraints for KNN
for j in range(line_flow_all.shape[0]):
    l_status_max = [i for i, e_max in enumerate(line_flow_all[j]) if e_max >= 3.5]
    l_status_min = [i for i, e_min in enumerate(line_flow_all[j]) if e_min <= -3.5]
    # print(l_status_max)
    for u in l_status_max:
        l_max_status[j][u] = 1
    for u in l_status_min:
        l_min_status[j][u] = 1
    # line_status_max[j].append(l_status_max)
    # line_status_min[j].append(l_status_min)
print(l_max_status[0])
    # print('real_binding_line_constraints_index', line_status_max[j][0])
    # print('real_binding_line_constraints_index', line_status_min[j][0])
print(np.shape(load_all))
print(np.shape(l_max_status))

with open('all118_max_con_for_KNN_50.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(l_max_status)

with open('all118_min_con_for_KNN_50.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(l_min_status)

with open('all118_min_load_for_KNN_50.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(load_all)









