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

from pypsa.linopt import define_variables, get_var
with open('PTDF14.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    # print(rows)
    a_ln = np.array(rows, dtype=float)
    print(a_ln[19][13])

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
num_samples = 1
num_of_gen = 5
num_buses = 14
num_of_lines = 20

m = np.ones((1, num_of_gen), dtype=float)
e_right = np.diag(m[0])*5
e_left = np.diag(m[0])*0
D = np.diag(m[0])

b = np.zeros((num_buses, num_of_gen), dtype=float)

for g in range(num_of_gen):
    b[g_index[g][0]-1][g] = 1

line_flow_limit = np.ones((1, num_of_lines), dtype=float) * 3
line_flow_limit = line_flow_limit[0]
line_flow_limit_l = np.ones((1, num_of_lines-1), dtype=float) * 3
line_flow_limit_l = line_flow_limit_l[0]

sta_line_min = []
sta_line_max = []

sta_line_bound_min = []
sta_line_bound_max = []
time_all = []

#Solver for collecting solutions
sta_line_bound_min = []
sta_line_bound_max = []
time_all = []
#Solver for collecting solutions
num = 0
num_con = np.zeros((1, num_samples), dtype=float)
num_con_bound = np.zeros((1, num_samples), dtype=float)
load_status = -1
percentage = 1
l_center = 15 / num_buses
for i in range(num_samples):

    load_status = -load_status
    load_left = np.ones((1, 7), dtype=float) * l_center
    load_right = np.ones((1, 7), dtype=float) * l_center
    # print(load_left)
    load_random_left = np.random.uniform(0, percentage * l_center, (1, 7))
    load_random_right = load_random_left[0][0:7]
    # print(load_random_left)
    load_left = load_left + load_random_left * load_status
    load_right = load_right - load_random_right * load_status
    # print(load_left[0])
    load = list(load_left[0]) + list(load_right[0])
    load = np.array(load)

    x = cp.Variable(num_of_gen)
    u = cp.Variable(num_of_gen, integer=True)
    q = cp.Variable(num_buses)
    # line_flow = cp.Variable(186)
    # load_fac = np.random.uniform(0.2, 1.8, (num_buses, 1)) #Uniform distribution now, change the scale here
    # with open('linear_load_test.csv', 'r') as csvfile:
    #     reader = csv.reader(csvfile)
    #     rows = [row for row in reader]
    #     load = np.array(rows, dtype=float)
    # print(load)
    #Scale 1: 0.2-1.8
    #Scale 2: 0.5-1.5
    #Scale 3: 0.8-1.2
    #Largest scale: 0.1-2.0

    # with tf.Session() as sess:
    #     saver = tf.train.import_meta_graph('my_test_model.meta')
    #     saver.restore(sess, tf.train.latest_checkpoint('./'))
    #     graph = tf.get_default_graph()
    #     load_input = graph.get_tensor_by_name("load:0")
    #     feed_dict = {load_input: load}
    #     cost_pre = graph.get_operation_by_name("predictions")

    start_time = time.time()
    #cost = cp.sum_squares(A * x)
    # A_tilda = np.dot(A.T, K_mat)
    cost = cp.sum(C * x)

    constraint = [q == b * x - load.reshape(-1,),
                  cp.sum(q) == 0,  #Generation constraint
                  e_left * u <= D * x,
                  D * x <= e_right * u,
                  u <= 1, u >= 0,
                  a_ln * q >= -line_flow_limit.reshape(-1, ),
                  line_flow_limit.reshape(-1, ) >= a_ln * q] #line flow limits
                  #cycle_mat * line_flow == cycle_vec.reshape(-1, )  # Cycle constraint
    #Formulation with cycle
    '''constraint = [b * x + A.T * line_flow == load.reshape(-1,),
                  cycle_mat * line_flow == cycle_vec.reshape(-1,), #Cycle constraint
                  D * x <= e, #Generation constraint
                  F * line_flow <= line_flow_limit.reshape(-1,)] #line flow limits'''
    try:
        prob = cp.Problem(cp.Minimize(cost), constraint)
        prob.solve(solver='GLPK_MI')  # cp.CVXOPT
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
    print("Solution time CPLEX", time.time() - start_time)

    # easier to solve
    start_time = time.time()
    # cost = cp.sum_squares(A * x)
    # A_tilda = np.dot(A.T, K_mat)
    x_re = cp.Variable(num_of_gen)
    u_re = cp.Variable(num_of_gen)
    q_re = cp.Variable(num_buses)
    # line_flow = cp.Variable(num_of_lines)

    cost_re = cp.sum(C * x_re)

    constraint = [q_re == b * x_re - load.reshape(-1, ),
                  cp.sum(q_re) == 0,  # Generation constraint
                  e_left * u_re <= D * x_re,
                  D * x_re <= e_right * u_re,
                  u <= 1, u >= 0,
                  a_ln * q_re >= -line_flow_limit.reshape(-1, ),
                  line_flow_limit.reshape(-1, ) >= a_ln * q_re]  # line flow limits
    # cycle_mat * line_flow == cycle_vec.reshape(-1, )  # Cycle constraint
    # Formulation with cycle
    '''constraint = [b * x + A.T * line_flow == load.reshape(-1,),
                  cycle_mat * line_flow == cycle_vec.reshape(-1,), #Cycle constraint
                  D * x <= e, #Generation constraint
                  F * line_flow <= line_flow_limit.reshape(-1,)] #line flow limits'''
    try:
        prob = cp.Problem(cp.Minimize(cost_re), constraint)
        prob.solve(solver='CPLEX')  # cp.CVXOPT
        if x_re.value is None:
            # print("No solution!!!!!!")
            # print("Impossible load combination", load)
            print("Load sum", np.sum(load))
            continue
    except KeyError:
        print("KeyError")
        continue
    print('cost_re', cost_re.value)

    for m in range(num_of_lines):
        a_ln_l = np.copy(a_ln)
        a_ln_l = np.delete(a_ln_l, m, axis=0)
        # print(a_ln_l)
        # print("shape of a_ln_l", np.shape(a_ln_l))
        x_l_max = cp.Variable(num_of_gen)
        u_l_max = cp.Variable(num_of_gen)
        q_l_max = cp.Variable(num_buses)
        load_l_max = cp.Variable(num_buses)
        l_max = a_ln[m][:] * q_l_max
        constraint_l_max = [q_l_max == b * x_l_max - load_l_max,
                            cp.sum(q_l_max) == 0,  # Generation constraint
                            e_left * u_l_max <= D * x_l_max,
                            D * x_l_max <= e_right * u_l_max,
                            u_l_max <= 1, u_l_max >= 0,
                            load_l_max >= l_center - percentage * l_center,
                            load_l_max <= l_center + percentage * l_center,
                            cp.sum(load_l_max) == 15,
                            a_ln_l * q_l_max >= -line_flow_limit_l.reshape(-1, ),
                            line_flow_limit_l.reshape(-1, ) >= a_ln_l * q_l_max]  # line flow limits

        # prob_min = cp.Problem(cp.Minimize(cost_l_min), constraint)
        # prob_min.solve(solver='CPLEX')  # cp.CVXOPT
        prob_max = cp.Problem(cp.Maximize(l_max), constraint_l_max)
        prob_max.solve(solver='CPLEX')
        #print('cost_l_min', cost_l_min.value)
        if x_l_max.value is None:
            print("No solution max!!!!!!")
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
        load_l_min = cp.Variable(num_buses)
        l_min = a_ln[m][:] * q_l_min
        constraint_l_min = [q_l_min == b * x_l_min - load_l_min,
                            cp.sum(q_l_min) == 0,  # Generation constraint
                            e_left * u_l_min <= D * x_l_min,
                            D * x_l_min <= e_right * u_l_min,
                            u_l_min <= 1, u_l_min >= 0,
                            load_l_min >= l_center - percentage * l_center,
                            load_l_min <= l_center + percentage * l_center,
                            cp.sum(load_l_min) == 15,
                            a_ln_l * q_l_min >= -line_flow_limit_l.reshape(-1, ),
                            line_flow_limit_l.reshape(-1, ) >= a_ln_l * q_l_min]  # line flow limits

        # prob_min = cp.Problem(cp.Minimize(cost_l_min), constraint)
        # prob_min.solve(solver='CPLEX')  # cp.CVXOPT
        prob_min = cp.Problem(cp.Minimize(l_min), constraint_l_min)
        prob_min.solve(solver='CPLEX')
        # print('cost_l_min', cost_l_min.value)
        if x_l_min.value is None:
            print("No solution max!!!!!!")
            print("Impossible load combination", load)
            print("Load sum", np.sum(load))
            num_con[i] += 1
            sta_line_min.append(m)
        # print('cost_l_min', cost_l_min.value)

        if abs(l_min.value) >= 3:
            num_con[i] += 1
            sta_line_min.append(m)


        x_l_max_bound = cp.Variable(num_of_gen)
        u_l_max_bound = cp.Variable(num_of_gen)
        q_l_max_bound = cp.Variable(num_buses)
        load_l_max_bound = cp.Variable(num_buses)
        l_max_bound = a_ln[m][:] * q_l_max_bound
        constraint_l_max_bound = [q_l_max_bound == b * x_l_max_bound - load_l_max_bound,
                                  cp.sum(q_l_max_bound) == 0,  # Generation constraint
                                  e_left * u_l_max_bound <= D * x_l_max_bound,
                                  D * x_l_max_bound <= e_right * u_l_max_bound,
                                  u_l_max_bound <= 1, u_l_max_bound >= 0,
                                  load_l_max_bound >= l_center - percentage * l_center,
                                  load_l_max_bound <= l_center + percentage * l_center,
                                  cp.sum(load_l_max_bound) == 15,
                                  a_ln_l * q_l_max_bound >= -line_flow_limit_l.reshape(-1, ),
                                  line_flow_limit_l.reshape(-1, ) >= a_ln_l * q_l_max_bound,
                                  C * x_l_max_bound <= 22.432411,
                                  ]  # line flow limits

        # prob_min = cp.Problem(cp.Minimize(cost_l_min), constraint)
        # prob_min.solve(solver='CPLEX')  # cp.CVXOPT
        prob_max_bound = cp.Problem(cp.Maximize(l_max_bound), constraint_l_max_bound)
        prob_max_bound.solve(solver='CPLEX')
        # print('cost_l_min', cost_l_min.value)
        if x_l_max_bound.value is None:
            print("No solution max!!!!!!")
            print("Impossible load combination", load)
            print("Load sum", np.sum(load))
            num_con_bound[i] += 1
            sta_line_bound_max.append(m)
        # print('cost_l_max', cost_l_max.value)
        elif abs(l_max_bound.value) >= 3:
            num_con_bound[i] += 1
            print('m', m)
            print('cost_l_max', l_max_bound.value)
            sta_line_bound_max.append(m)

        x_l_min_bound = cp.Variable(num_of_gen)
        u_l_min_bound = cp.Variable(num_of_gen)
        q_l_min_bound = cp.Variable(num_buses)
        load_l_min_bound = cp.Variable(num_buses)
        l_min_bound = a_ln[m][:] * q_l_min_bound
        constraint_l_min_bound = [q_l_min_bound == b * x_l_min_bound - load_l_min_bound,
                                  cp.sum(q_l_min_bound) == 0,  # Generation constraint
                                  e_left * u_l_min_bound <= D * x_l_min_bound,
                                  D * x_l_min_bound <= e_right * u_l_min_bound,
                                  u_l_min_bound <= 1, u_l_min_bound >= 0,
                                  load_l_min_bound >= l_center - percentage * l_center,
                                  load_l_min_bound <= l_center + percentage * l_center,
                                  cp.sum(load_l_min_bound) == 15,
                                  a_ln_l * q_l_min_bound >= -line_flow_limit_l.reshape(-1, ),
                                  line_flow_limit_l.reshape(-1, ) >= a_ln_l * q_l_min_bound,
                                  C * x_l_min_bound <= 22.432411,
                                  ]  # line flow limits

        # prob_min = cp.Problem(cp.Minimize(cost_l_min), constraint)
        # prob_min.solve(solver='CPLEX')  # cp.CVXOPT
        prob_min_bound = cp.Problem(cp.Minimize(l_min_bound), constraint_l_min_bound)
        prob_min_bound.solve(solver='CPLEX')
        # print('cost_l_min', cost_l_min.value)
        if x_l_min_bound.value is None:
            print("No solution max!!!!!!")
            print("Impossible load combination", load)
            print("Load sum", np.sum(load))
            num_con_bound[i] += 1
            sta_line_bound_min.append(m)

        # print('cost_l_max', cost_l_max.value)
        if abs(l_min_bound.value) >= 3:
            num_con_bound[i] += 1
            sta_line_bound_min.append(m)
            print('m', m)
            print('cost_l_min', l_min_bound.value)

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
        # line_flow_all = line_flow.value.reshape(1,-1)
        line_flow_all = np.dot(a_ln, q.value).reshape(1, -1)
        cost_all = []
        cost_all.append(prob.value)
        # constraint_all = np.concatenate((constraint[0].dual_value.reshape(-1,),
        #                                     constraint[1].dual_value.reshape(-1,),
        #                                     constraint[2].dual_value.reshape(-1,)))
    else:
        x_val.append(x.value)
        u_val.append(u.value)
        l_sum.append(np.sum(load))
        # line_flow_all = np.concatenate((line_flow_all, line_flow.value.reshape(1, -1)), axis=0)
        line_flow_all = np.vstack((line_flow_all, np.dot(a_ln, q.value).reshape(1, -1)))
        # print(np.shape(np.dot(a_ln, q.value).reshape(1, -1)))
        cost_all.append(prob.value)
        # constraint_all = np.concatenate((constraint_all, np.concatenate((constraint[0].dual_value.reshape(-1,),
        #                                     constraint[1].dual_value.reshape(-1,),
        #                                     constraint[2].dual_value.reshape(-1,)))))
    num += 1
    # dual_val[i][0]=constraint[0].dual_value
    # dual_val[i,1:]=constraint[1].dual_value
print(line_flow_all)
line_status = [[] for _ in range(num_samples)]
for j in range(num_samples):
    l_status = [i for i, e in enumerate(line_flow_all[j]) if abs(e) > 2.99]
    line_status[j].append(l_status)
print('real_binding_line_constraints_index', line_status[0][0])
print(num_con)
print(num_con_bound)
# print("shape of line flow", np.shape(line_flow_all))
# print("line flow", line_flow_all)
# print("shape of constraint", np.shape(constraint_all))
time_all = np.array(time_all).reshape(-1, 1)

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
# with open('linear_constraint14_s3.csv', 'w',newline='') as f:
#     writer = csv.writer(f)
#     constraint_all = np.array(constraint_all, dtype=float).reshape(-1, num_buses + num_of_gen * 2 + num_of_lines * 2)
#     constraint_all = np.round(constraint_all, 4)
#     writer.writerows(constraint_all)

# cost_all = np.array(cost_all, dtype=float).reshape(-1, 1)
# print(np.shape(load_all))
# print(np.shape(x_val))
# print(np.shape(line_flow_all))
# print(np.shape(cost_all))
# data_all = np.concatenate((load_all, x_val, line_flow_all, cost_all), axis=1)
# data_all = np.round(data_all, 4)
# with open('linear_data_all118_LP_RE.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerows(data_all)










