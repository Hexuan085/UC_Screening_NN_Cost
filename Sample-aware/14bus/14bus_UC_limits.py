import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import random
import csv
from datetime import datetime
import time
import warnings
random.seed(32)
from sklearn.metrics import mean_squared_error
from keras.layers import Dense, Dropout, Activation, Flatten
from sklearn.metrics import mean_squared_error
from keras.constraints import non_neg
import time
import h5py
import os
from keras.models import Sequential
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from keras.models import load_model
np.random.seed(32)

warnings.filterwarnings("ignore")

# Load the known parameters
with open('PTDF14.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    # print(rows)
    a_ln = np.array(rows, dtype=float)
    # print(a_ln[185][117])

with open('GEN14.csv', 'r') as csvfile:
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
C = C[0][0:5]

with open('linear_load_max.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    # print(rows)
    max_valuex = np.array(rows, dtype=float)
    # print(C)
with open('linear_load_min.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    # print(rows)
    min_valuex = np.array(rows, dtype=float)
    # print(C)

# Define the used NN model
def keras_model_dnn(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(100, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(30))
    model.add(Activation('relu'))
    model.add(Dense(output_dim))
    return model

num_samples = 100

# Topology of used network
num_of_gen = 5  # Total amount of generator
num_buses = 14  # Total amount of bus
num_of_lines = 20  # Total amount of lines

# Set the constraint coefficients
m = np.ones((1, num_of_gen), dtype=float)
e_right = np.diag(m[0])*5  # Upper bound of generation
e_left = np.diag(m[0])*0  # Lower bound of generation
D = np.diag(m[0])  # Generator coefficient matrix in bounds

b = np.zeros((num_buses, num_of_gen), dtype=float)  # Initial generator coefficient in net demand equation

for g in range(num_of_gen):  # Modify generator coefficient in net demand equation
    b[g_index[g][0]-1][g] = 1

line_flow_limit = np.ones((1, num_of_lines), dtype=float) * 3  # Upper\lower bound of line flow
line_flow_limit = line_flow_limit[0]
line_flow_limit_l = np.ones((1, num_of_lines-1), dtype=float) * 3  # Upper\lower bound of line flow
line_flow_limit_l = line_flow_limit_l[0]

# Initialize the binding status matrix
sta_line_min = []
sta_line_max = []

sta_line_bound_min = []
sta_line_bound_max = []

num_con = np.zeros((num_samples, 1), dtype=float)
num_con_bound = np.zeros((num_samples, 1), dtype=float)

load_status = -1  # load generation status
percentage = 1  # load variation range $r$
l_center = 15 / num_buses  # average nominal value

infeasible = 0

# Load the model parameters
model = keras_model_dnn(input_dim=num_buses, output_dim=1)
model.load_weights(r'model14.h5')

num = 0
for i in range(num_samples):
    # load_status = -load_status
    # load_left = np.ones((1, 7), dtype=float) * l_center
    # load_right = np.ones((1, 7), dtype=float) * l_center
    # # print(load_left)
    # load_random_left = np.random.uniform(0, percentage * l_center, (1, 7))
    # load_random_right = load_random_left[0][0:7]
    # # print(load_random_left)
    # load_left = load_left + load_random_left * load_status
    # load_right = load_right - load_random_right * load_status
    # # print(load_left[0])
    # load = list(load_left[0]) + list(load_right[0])
    # load = np.array(load)
    load = np.random.uniform(0, l_center * 2, (num_buses, 1))

    # Original UC Problem
    x = cp.Variable(num_of_gen)
    u = cp.Variable(num_of_gen, integer=True)  # Generator on/off status variables
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

    # Relaxed UC problem
    x_re = cp.Variable(num_of_gen)
    u_re = cp.Variable(num_of_gen)
    q_re = cp.Variable(num_buses)

    cost_re = cp.sum(C * x_re)

    constraint = [q_re == b * x_re - load.reshape(-1, ),
                  cp.sum(q_re) == 0,
                  e_left * u_re <= D * x_re,
                  D * x_re <= e_right * u_re,
                  u <= 1, u >= 0, # relaxed integer variables
                  a_ln * q_re >= -line_flow_limit.reshape(-1, ),
                  line_flow_limit.reshape(-1, ) >= a_ln * q_re]
    try:
        prob = cp.Problem(cp.Minimize(cost_re), constraint)
        prob.solve(solver='CPLEX')
        if x_re.value is None:
            # print("No solution!!!!!!")
            # print("Impossible load combination", load)
            print("Load sum", np.sum(load))
            continue
    except KeyError:
        print("KeyError")
        continue
    print('cost_re', cost_re.value)

    # Predict the cost for specific load vector
    load_input = (np.copy(load).reshape(1, num_buses) - min_valuex.reshape(1, num_buses)) / (max_valuex.reshape(1, num_buses) - min_valuex.reshape(1, num_buses))
    load_input = load_input.reshape(1, num_buses)
    cost_pre = model.predict(load_input)
    max_valuey = 42.4424
    min_valuey = 8.7101
    cost_pre = cost_pre * (max_valuey - min_valuey) + min_valuey
    print(cost_pre)

    if cost_pre + 0.01 * cost_pre < cost.value:
        infeasible += 1
        continue

    for m in range(num_of_lines):
        a_ln_l = np.copy(a_ln)
        a_ln_l = np.delete(a_ln_l, m, axis=0)
        # print(a_ln_l)
        # print("shape of a_ln_l", np.shape(a_ln_l))

        # Standard optimization-based screening
        x_l_max = cp.Variable(num_of_gen)
        u_l_max = cp.Variable(num_of_gen)
        q_l_max = cp.Variable(num_buses)
        load_l_max = load
        l_max = a_ln[m][:] * q_l_max
        constraint_l_max = [q_l_max == b * x_l_max - load_l_max.reshape(-1, ),
                            cp.sum(q_l_max) == 0,
                            e_left * u_l_max <= D * x_l_max,
                            D * x_l_max <= e_right * u_l_max,
                            u_l_max <= 1, u_l_max >= 0,
                            a_ln_l * q_l_max >= -line_flow_limit_l.reshape(-1, ),
                            line_flow_limit_l.reshape(-1, ) >= a_ln_l * q_l_max]
        prob_max = cp.Problem(cp.Maximize(l_max), constraint_l_max)
        prob_max.solve(solver='CPLEX')
        # print('cost_l_min', cost_l_min.value)
        if x_l_max.value is None:
            print("No solution max_l !!!!!!")
            print("Impossible load combination", load)
            print("Load sum", np.sum(load))
            num_con[i] += 1
            sta_line_max.append(m)

        # print('cost_l_max', cost_l_max.value)
        if abs(l_max.value) >= 3:
            num_con[i] += 1
            sta_line_max.append(m)

        x_l_min = cp.Variable(num_of_gen)
        u_l_min = cp.Variable(num_of_gen)
        q_l_min = cp.Variable(num_buses)
        load_l_min = load
        l_min = a_ln[m][:] * q_l_min
        constraint_l_min = [q_l_min == b * x_l_min - load_l_min.reshape(-1, ),
                            cp.sum(q_l_min) == 0,
                            e_left * u_l_min <= D * x_l_min,
                            D * x_l_min <= e_right * u_l_min,
                            u_l_min <= 1, u_l_min >= 0,
                            a_ln_l * q_l_min >= -line_flow_limit_l.reshape(-1, ),
                            line_flow_limit_l.reshape(-1, ) >= a_ln_l * q_l_min]
        prob_min = cp.Problem(cp.Minimize(l_min), constraint_l_min)
        prob_min.solve(solver='CPLEX')
        # print('cost_l_min', cost_l_min.value)
        if x_l_min.value is None:
            print("No solution min_l!!!!!!")
            print("Impossible load combination", load)
            print("Load sum", np.sum(load))
            num_con[i] += 1
            sta_line_min.append(m)
        # print('cost_l_min', cost_l_min.value)

        if abs(l_min.value) >= 3:
            num_con[i] += 1
            sta_line_min.append(m)

        # Cost-driven screening
        x_l_max_bound = cp.Variable(num_of_gen)
        u_l_max_bound = cp.Variable(num_of_gen)
        q_l_max_bound = cp.Variable(num_buses)
        load_l_max_bound = load

        # print(cost_pre)
        l_max_bound = a_ln[m][:] * q_l_max_bound
        constraint_l_max_bound = [q_l_max_bound == b * x_l_max_bound - load_l_max_bound.reshape(-1, ),
                                  cp.sum(q_l_max_bound) == 0,
                                  e_left * u_l_max_bound <= D * x_l_max_bound,
                                  D * x_l_max_bound <= e_right * u_l_max_bound,
                                  u_l_max_bound <= 1, u_l_max_bound >= 0,
                                  a_ln_l * q_l_max_bound >= -line_flow_limit_l.reshape(-1, ),
                                  line_flow_limit_l.reshape(-1, ) >= a_ln_l * q_l_max_bound,
                                  C * x_l_max_bound <= cost_pre+cost_pre*0.01,  # Cost constraint
                                  ]
        prob_max_bound = cp.Problem(cp.Maximize(l_max_bound), constraint_l_max_bound)
        prob_max_bound.solve(solver='CPLEX')
        # print('cost_l_min', cost_l_min.value)
        if x_l_max_bound.value is None:
            print("No solution max_l_bound!!!!!!")
            print("Impossible load combination", load)
            print("Load sum", np.sum(load))
            num_con_bound[i] += 1
            sta_line_bound_max.append(m)
        # print('cost_l_max', cost_l_max.value)
        elif abs(l_max_bound.value) >= 3:
            num_con_bound[i] += 1
            # print('m', m)
            # print('cost_l_max', l_max_bound.value)
            sta_line_bound_max.append(m)

        x_l_min_bound = cp.Variable(num_of_gen)
        u_l_min_bound = cp.Variable(num_of_gen)
        q_l_min_bound = cp.Variable(num_buses)
        load_l_min_bound = load
        l_min_bound = a_ln[m][:] * q_l_min_bound
        constraint_l_min_bound = [q_l_min_bound == b * x_l_min_bound - load_l_min_bound.reshape(-1, ),
                                  cp.sum(q_l_min_bound) == 0,
                                  e_left * u_l_min_bound <= D * x_l_min_bound,
                                  D * x_l_min_bound <= e_right * u_l_min_bound,
                                  u_l_min_bound <= 1, u_l_min_bound >= 0,
                                  a_ln_l * q_l_min_bound >= -line_flow_limit_l.reshape(-1, ),
                                  line_flow_limit_l.reshape(-1, ) >= a_ln_l * q_l_min_bound,
                                  C * x_l_max_bound <= cost_pre+cost_pre*0.01,  # Cost constraint
                                  ]
        prob_min_bound = cp.Problem(cp.Minimize(l_min_bound), constraint_l_min_bound)
        prob_min_bound.solve(solver='CPLEX')
        # print('cost_l_min', cost_l_min.value)
        if x_l_min_bound.value is None:
            print("No solution min_l_bound!!!!!!")
            print("Impossible load combination", load)
            print("Load sum", np.sum(load))
            num_con_bound[i] += 1
            sta_line_bound_min.append(m)

        # print('cost_l_max', cost_l_max.value)
        if abs(l_min_bound.value) >= 3:
            num_con_bound[i] += 1
            sta_line_bound_min.append(m)
            # print('m', m)
            # print('cost_l_min', l_min_bound.value)

    # print("Current generation", x.value)
    if num == 0:
        load_all = load.T
    else:
        load_all = np.concatenate((load_all, load.T))
    if num == 0:
        x_val = []
        u_val = []
        l_sum = []
        l_sum.append(np.sum(load))
        x_val.append(x.value)
        u_val.append(u.value)

        line_flow_all = np.dot(a_ln, q.value).reshape(1, -1)
        cost_all = []
        cost_all.append(prob.value)

    else:
        x_val.append(x.value)
        u_val.append(u.value)
        l_sum.append(np.sum(load))

        line_flow_all = np.vstack((line_flow_all, np.dot(a_ln, q.value).reshape(1, -1)))
        # print(np.shape(np.dot(a_ln, q.value).reshape(1, -1)))
        cost_all.append(prob.value)

    num += 1
    print(i)

# print(line_flow_all)

# Record the actual binding constraints
line_status = [[] for _ in range(line_flow_all.shape[0])]
l_record = []
for j in range(line_flow_all.shape[0]):
    l_status = [i for i, e in enumerate(line_flow_all[j]) if abs(e) >= 2.99]
    l_record.append(len(l_status))
print('real_binding_line_constraints_index', len(l_status))

# Calculate the proportion of the screened constraints
print((num_of_lines*2 - int(np.mean(num_con))))
print((num_of_lines*2 - int(np.mean(num_con_bound))))
print((num_of_lines*2 - int(np.mean(num_con)))/(num_of_lines*2))
print((num_of_lines*2 - int(np.mean(num_con_bound)))/(num_of_lines*2))
print((num_of_lines*2 - int(np.mean(l_record)))/(num_of_lines*2))
print(infeasible)

with open('UC_sta_line_min.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(sta_line_min)

with open('UC_sta_line_max.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(sta_line_max)

with open('UC_sta_line_bound_min.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(sta_line_bound_min)

with open('UC_sta_line_bound_max.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(sta_line_bound_max)










