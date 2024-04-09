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

load_range = 100  # Change the load region here so as to use different dataset (r=25~100)
with open('PTDF118.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    # print(rows)
    a_ln = np.array(rows, dtype=float)
    # print(a_ln[185][117])

with open('GEN118.csv', 'r') as csvfile:
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

with open('UC_sta_line_max_'+str(load_range)+'.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    # print(rows)
    sta_line_max = np.array(rows, dtype=int)
    # print(C)

with open('UC_sta_line_min_'+str(load_range)+'.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    # print(rows)
    sta_line_min = np.array(rows, dtype=int)
    # print(C)

with open('UC_sta_line_bound_max_'+str(load_range)+'.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    # print(rows)
    sta_line_bound_max = np.array(rows, dtype=int)
    # print(C)

with open('UC_sta_line_bound_min_'+str(load_range)+'.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    # print(rows)
    sta_line_bound_min = np.array(rows, dtype=int)
    # print(C)

with open('all118_min_load_for_KNN_'+str(load_range)+'.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    # print(rows)
    load_train = np.array(rows, dtype=float)

with open('all118_min_con_for_KNN_'+str(load_range)+'.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    # print(rows)
    l_min = np.array(rows, dtype=float)

with open('all118_max_con_for_KNN_'+str(load_range)+'.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    # print(rows)
    l_max = np.array(rows, dtype=float)


def knn(in_data, train_data, k, l_max, l_min, a_ln, num_line):
    a_ln_max_knn = []
    a_ln_min_knn = []
    l_max_knn = np.zeros((186, 1), dtype=float)
    l_min_knn = np.zeros((186, 1), dtype=float)
    train_data_size = train_data.shape[0]
    #Change size of the input data into train_data_size rows and 1 column
    distance = (np.tile(in_data, (train_data_size, 1)) - train_data) ** 2
    add_distance = distance.sum(axis=1)
    sq_distance = add_distance ** 0.5
    # Sort the Euclidean distance and return the corresponding index value
    index = sq_distance.argsort()
    # Find the labels corresponding to the first k smallest distances
    for j in range(num_line):
        for i in range(k):
            l_max_knn[j] = l_max_knn[j] + l_max[index[i]][j]
            l_min_knn[j] = l_min_knn[j] + l_min[index[i]][j]

    for j in range(num_line):
        if l_max_knn[j] >= 1:
            a_ln_max_knn.append(a_ln[j][:])
        if l_min_knn[j] >= 1:
            a_ln_min_knn.append(a_ln[j][:])
    return a_ln_max_knn, a_ln_min_knn


num_samples = 100
num_of_gen = 54
num_buses = 118
num_lines = 186

error_knn = 0
error_bound = 0
error_original = 0

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
time_3 = 0

load_status = -1
num = 0
percentage = 0.75
l_center = 120 / num_buses
k = 5

for i in range(num_samples):
    # Load generation
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

    # print(load)

    # Original UC problem
    start_time = time.time()
    x = cp.Variable(num_of_gen)
    u = cp.Variable(num_of_gen, integer=True)
    q = cp.Variable(num_buses)
    cost = cp.sum(C * x)
    constraint = [q == b * x - load.reshape(-1, ),
                  cp.sum(q) == 0,  # Generation constraint
                  e_left * u <= D * x,
                  D * x <= e_right * u,
                  u <= 1, u >= 0,
                  a_ln * q >= -line_flow_limit.reshape(-1, ),
                  line_flow_limit.reshape(-1, ) >= a_ln * q]  # line flow limits
    try:
        prob = cp.Problem(cp.Minimize(cost), constraint)
        start_time = time.time()
        prob.solve(solver='GLPK_MI')  # cp.CVXOPT
        time_0 = time_0 + time.time() - start_time
        print("Solution time CPLEX of original problem", time.time() - start_time)
        if x.value is None:
            # print("No solution!!!!!!")
            # print("Impossible load combination", load)
            print("Load sum", np.sum(load))
            continue
    except KeyError:
        print("KeyError")
        continue
    print('cost', cost.value)

    # Reduced problem derived by standard optimization-based screening
    x_reduce_0 = cp.Variable(num_of_gen)
    u_reduce_0 = cp.Variable(num_of_gen, integer=True)
    q_reduce_0 = cp.Variable(num_buses)
    cost_reduce_0 = cp.sum(C * x_reduce_0)
    constraint_reduce_0 = [q_reduce_0 == b * x_reduce_0 - load.reshape(-1, ),
                  cp.sum(q_reduce_0) == 0,  # Generation constraint
                  e_left * u_reduce_0 <= D * x_reduce_0,
                  D * x_reduce_0 <= e_right * u_reduce_0,
                  u_reduce_0 <= 1, u_reduce_0 >= 0,
                  a_ln_min[0] * q_reduce_0 >= -line_flow_limit_min.reshape(-1, ),
                  line_flow_limit_max.reshape(-1, ) >= a_ln_max[0] * q_reduce_0]  # line flow limits
    try:
        prob = cp.Problem(cp.Minimize(cost_reduce_0), constraint_reduce_0)
        start_time = time.time()
        prob.solve(solver='GLPK_MI')  # cp.CVXOPT
        time_1 = time_1 + time.time() - start_time
        print("Solution time CPLEX of reduced problem 1", time.time() - start_time)
        if x_reduce_0.value is None:
            # print("No solution!!!!!!")
            # print("Impossible load combination", load)
            print("Load sum", np.sum(load))
            # continue
    except KeyError:
        print("KeyError")
        continue
    print('cost_reduce_1', cost_reduce_0.value)
    error_original = error_original + abs(cost.value - cost_reduce_0.value)/cost.value

    # Reduced problem derived by cost-driven screening
    start_time = time.time()
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
        if x_reduce_1.value is None:
            # print("No solution!!!!!!")
            # print("Impossible load combination", load)
            print("Load sum", np.sum(load))
            continue
    except KeyError:
        print("KeyError")
        continue
    print('cost_reduce_2', cost_reduce_1.value)
    error_bound = error_bound + abs(cost.value - cost_reduce_1.value)/cost.value

    # Reduced problem derived by KNN
    start_time1 = time.time()
    a_ln_max_knn, a_ln_min_knn = knn(load, load_train, k, l_max, l_min, a_ln, num_lines) # Using KNN to screening the non-redundant constraints
    knn_time = time.time() - start_time1
    a_ln_max_knn = np.array(a_ln_max_knn)
    a_ln_min_knn = np.array(a_ln_min_knn)
    print('shape of a_ln', np.shape(a_ln_max_knn), np.shape(a_ln_min_knn))

    line_flow_limit_max_knn = np.ones((1, a_ln_max_knn.shape[0]), dtype=float) * 3.5
    line_flow_limit_min_knn = np.ones((1, a_ln_min_knn.shape[0]), dtype=float) * 3.5

    x_reduce_knn = cp.Variable(num_of_gen)
    u_reduce_knn = cp.Variable(num_of_gen, integer=True)
    q_reduce_knn = cp.Variable(num_buses)

    cost_reduce_knn = cp.sum(C * x_reduce_knn)

    constraint_reduce_knn = [q_reduce_knn == b * x_reduce_knn - load.reshape(-1, ),
                             cp.sum(q_reduce_knn) == 0,
                             e_left * u_reduce_knn <= D * x_reduce_knn,
                             D * x_reduce_knn <= e_right * u_reduce_knn,
                             u_reduce_knn <= 1, u_reduce_0 >= 0,
                             a_ln_min_knn * q_reduce_knn >= -line_flow_limit_min_knn.reshape(-1, ),
                             line_flow_limit_max_knn.reshape(-1, ) >= a_ln_max_knn * q_reduce_knn]  # line flow limits
    try:
        prob_knn = cp.Problem(cp.Minimize(cost_reduce_knn), constraint_reduce_knn)
        start_time = time.time()
        prob_knn.solve(solver='GLPK_MI')  # cp.CVXOPT
        time_3 = time_3 + time.time() - start_time + knn_time
        print("Solution time CPLEX of reduced problem 3", time.time() - start_time + knn_time)
        if x_reduce_knn.value is None:
            # print("No solution!!!!!!")
            # print("Impossible load combination", load)
            print("Load sum", np.sum(load))
            # continue
    except KeyError:
        print("KeyError")
        # continue
    print('cost_reduce_knn', cost_reduce_knn.value)

    error_knn = error_knn + abs(cost.value - cost_reduce_knn.value)/cost.value

print('time_0, time_1, time_2, time_3', time_0, time_1/time_0, time_2/time_0, time_3/time_0)
print('error_knn', error_knn)
print('error_bound', error_bound)
print('error_original', error_original)
