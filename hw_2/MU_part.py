# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import matplotlib.pyplot as plt


# %%
# All the functions related to section b

bin = lambda x : ''.join(reversed( [str((x >> i) & 1) for i in range(jobs)] ) )
def calc_cost_per_state(problem_table, state):
    in_jobs = bin(state)
    in_jobs=in_jobs[::-1]
    cost = 0
    for idx, job in enumerate(in_jobs):
        if int(job):
            cost += problem_table[1, idx]
    return cost

def calc_p(problem_table, policy):
    P = np.zeros([states, states])
    for row in range(states):
        for col in range(states):
            curr_state = row
            next_state = col
            if curr_state > 0:
                possible_next_state = curr_state - 2 ** (int(policy[curr_state]) - 1)
            if curr_state == 0:
                if next_state == 0:
                    P[row,col] = 1
                else:
                    P[row,col] = 0
                continue
            if next_state == possible_next_state:
                P[row,col] = problem_table[0, int(policy[curr_state]) - 1]
            elif next_state == curr_state:
                P[row,col] = 1 - problem_table[0, int(policy[curr_state]) - 1]
            else:
                P[row,col] = 0
    return P  

def get_v(p, r):
    p[0][0] = 0
    return np.dot(np.linalg.inv(np.identity(len(r))-p), r)

def get_v_iteration(p, r):
    v = np.ones([len(r)])
    v_new = np.zeros([len(r)])
    while not (v == v_new).all():
        v = v_new.copy()
        v_new = r + p @ v
    return v


# %%
# policies defintions

def get_policy_c(table, state):
    in_jobs = bin(state)
    in_jobs=in_jobs[::-1]
    max_c = -1
    max_idx = -1
    for idx, job in enumerate(in_jobs):
        if int(job):
            cost = problem_table[1, idx]
            if cost > max_c:
                max_c = cost
                max_idx = idx
    return max_idx+1

def get_policy_uc(table, state):
    in_jobs = bin(state)
    in_jobs=in_jobs[::-1]
    max_c = -1
    max_idx = -1
    for idx, job in enumerate(in_jobs):
        if int(job):
            cost = problem_table[1, idx] * problem_table[0, idx]
            if cost > max_c:
                max_c = cost
                max_idx = idx
    return max_idx+1

def get_policy_rand(table, state):
    in_jobs = bin(state)
    in_jobs=in_jobs[::-1]
    while True:
        rand_job = np.random.randint(len(in_jobs))
        if int(in_jobs[rand_job]):
            break
    return rand_job+1

def get_full_policy_uc(table):
    num_states = 2 ** (len(table[0]))
    policy = np.zeros(num_states)
    for state in range(num_states):
        policy[state] = get_policy_uc(table, state)
    return policy

def get_full_policy_c(table):
    num_states = 2 ** (len(table[0]))
    policy = np.zeros(num_states)
    for state in range(num_states):
        policy[state] = get_policy_c(table, state)
    return policy


def get_full_policy_rand(table):
    num_states = 2 ** (len(table[0]))
    policy = np.zeros(num_states)
    for state in range(num_states):
        if state == 0:
            policy[state] = 0
            continue
        policy[state] = get_policy_rand(table, state)
    return policy


# %%
# calc policy iteration related

def is_action_valid(state, action):
    in_jobs = bin(state)
    in_jobs=in_jobs[::-1]
    return bool(int(in_jobs[action-1]))

def get_temp_p(state, action):
    p = np.zeros([2 ** jobs])
    possible_next_state = state - 2 ** (action - 1)
    p[possible_next_state] = problem_table[0, action - 1]
    p[state] = 1 - problem_table[0, action - 1]
    return p

def get_new_policy(r, v):
    num_states = 32
    policy = np.zeros([num_states])
    for state in range(num_states):
        min_action_val = 10000000
        min_action = -1
        for a in range(1,jobs+1):
            if is_action_valid(state=state, action=a):
                val = r[state]+get_temp_p(state=state, action=a) @ v
                if val < min_action_val:
                    min_action_val = val
                    min_action = a
        policy[state] = min_action
    return policy

def policy_iteration(r, table):
    num_states = 2 ** len(table[0])
    new_policy = np.zeros([num_states])
    init_state_value = []
    for state in range(num_states):
        new_policy[state] = get_policy_c(table=table, state=state)
    policy = np.zeros([num_states])
    while not (new_policy == policy).all():
        #print("*"*20)
        #print(new_policy)
        policy = new_policy.copy()
        p = calc_p(problem_table=table, policy=policy)
        v = get_v_iteration(p=p, r=r)
        # tried to extract values throughout iteration
        init_state_value.append(v[-1])
        #print(v)
        new_policy = get_new_policy(r=r, v=v)
    return policy, init_state_value
        


# %%
jobs = 5
states = 2 ** jobs
cost = np.zeros([states])
problem_table = np.array([
    [0.6, 0.5, 0.3, 0.7, 0.1],
    [1,     4,   6,   2,   9]
])

for state in range(states):
    cost[state] = calc_cost_per_state(problem_table=problem_table, state=state)


# %%
# section B
policy = get_full_policy_uc(table=problem_table)
p = calc_p(problem_table, policy)
get_v_iteration(p=p, r=cost)
get_v(p=p, r=cost)


# %%
# section C
policy = get_full_policy_c(table=problem_table)
p = calc_p(problem_table, policy)
get_v_iteration(p=p, r=cost)
V_c = get_v(p=p, r=cost)

plt.plot(V_c)
plt.grid()
plt.ylabel('value')
plt.xlabel('state')
plt.title('pi_c values')
plt.show()


# %%
# section d + e
policy_uc, s0_values = policy_iteration(r=cost, table=problem_table)
print("Policy Iteration: \n", policy_uc)
print("uc by definition: \n", get_full_policy_uc(table=problem_table))

policy = get_full_policy_uc(table=problem_table)
p = calc_p(problem_table, policy)
V_uc = get_v(p=p, r=cost)


plt.plot(V_c, label='V_c')
plt.plot(V_uc, label='V_uc')
plt.title('compare V_c and V_uc')
plt.ylabel('value')
plt.xlabel('state')
plt.grid()
plt.legend()

plt.show()

# print full state value
plt.plot([1, 2],s0_values, label='s0 value')
plt.title('s0 values throughout policy iteration')
plt.ylabel('value')
plt.xlabel('iteration')
plt.xticks([1,2])
# plt.grid()
plt.legend()
plt.show()


# %%
#section f Simulator
print('section f - simulator')
def single_step_sim(state, action, table):
    cost = calc_cost_per_state(problem_table=table, state=state)
    miu=problem_table[0][int(action-1)]
    finished = np.random.binomial(1, miu)
    if bool(finished):
        new_state = state - 2 ** (action-1)
    else:
        new_state = state
    return cost, new_state
    
def get_next_state(state, action, miu):
    finished = np.random.binomial(1, miu)
    #print("is done:",finished)
    if bool(finished):
        #print("done")
        new_state = state - 2 ** (action-1)
    else:
        #print("not done")
        new_state = state
    #print("new_state", new_state)
    return new_state

def simulation(policy, table, random=False):
    if random:
        curr_state = int(np.random.randint(32))
    else:
        curr_state = 2 ** len(table[0]) -1
    total_cost = 0
    trajectories = []
    while not curr_state == 0:
        curr_cost = calc_cost_per_state(problem_table=table, state=curr_state)
        curr_action = policy[curr_state]
        #print("="*10)
        #print("state:",bin(curr_state)[::-1])
        #print("action:", curr_action)
        new_state = get_next_state(state=curr_state, action=curr_action, miu=problem_table[0][int(curr_action-1)])
        total_cost += curr_cost
        new_sample = (int(curr_state), int(curr_action), int(curr_cost), int(new_state))
        curr_state = int(new_state)
        #print("cost:", total_cost)
        trajectories.append(new_sample)
        
    return total_cost, trajectories


# %%
# section g
print('section g')
def update_TD_zero(estimated_v, alpha, gamma, sample):
    s_t, a_t, r_t, s_tag = sample
    v_s_t = estimated_v[s_t]
    v_s_tag = estimated_v[s_tag]
    delta_t = r_t + gamma * v_s_tag - v_s_t
    updated_v = estimated_v.copy()
    updated_v[s_t] = estimated_v[s_t] + alpha * delta_t
    return updated_v

N = 10000

for count,title in enumerate(['TD_zero multiple alphas', 'All jobs in system state (s0) - TD_zero multiple alphas']):
    # plot values
    inf_norm = []
    final_state_norm = []

    v = np.zeros([32])
    for i in range(N):
        cost, trajectory = simulation(policy=get_full_policy_c(table=problem_table), table=problem_table, random=True)
        for sample in trajectory:
            alpha = 0.01
            v = update_TD_zero(estimated_v=v, alpha=alpha, gamma=1, sample=sample)

        # added tracers for plot
        inf_norm.append(np.linalg.norm((V_c - v),ord=np.inf))
        final_state_norm.append(np.linalg.norm(V_c[-1] - v[-1]))

    if count == 0:
        plt.plot(inf_norm, color='b', label='TD_zero (alpha=0.01)')
    else:
        plt.plot(final_state_norm, color='b', label='TD_zero (alpha=0.01)')

    # plot values
    inf_norm = []
    final_state_norm = []

    print("*"*20)
    print("TD(0) alpha=0.01 for policy_c:\n", np.round(v,3))

    v = np.zeros([32])
    n = np.zeros([32])
    for i in range(N):
        cost, trajectory = simulation(policy=get_full_policy_c(table=problem_table), table=problem_table, random=True)
        for sample in trajectory:
            s_t, _, _, _ = sample
            n[s_t] += 1
            alpha = 1/n[s_t]
            v = update_TD_zero(estimated_v=v, alpha=alpha, gamma=1, sample=sample)

        # added tracers for plot
        inf_norm.append(np.linalg.norm((V_c - v),ord=np.inf))
        final_state_norm.append(np.linalg.norm(V_c[-1] - v[-1]))
    if count == 0:
        plt.plot(inf_norm, color='r', label='TD_zero (alpha=1/n)')
    else:
        plt.plot(final_state_norm, color='r', label='TD_zero (alpha=1/n)')
    

    # plot values
    inf_norm = []
    final_state_norm = []

    print("*"*20)
    print("TD(0) alpha=1/n for policy_c:\n", np.round(v,3))



    v = np.zeros([32])
    n = np.zeros([32])
    for i in range(N):
        cost, trajectory = simulation(policy=get_full_policy_c(table=problem_table), table=problem_table, random=True)
        for sample in trajectory:
            s_t, _, _, _ = sample
            n[s_t] += 1
            alpha = 10/(100 + n[s_t])
            v = update_TD_zero(estimated_v=v, alpha=alpha, gamma=1, sample=sample)

        # added tracers for plot
        inf_norm.append(np.linalg.norm((V_c - v),ord=np.inf))
        final_state_norm.append(np.linalg.norm(V_c[-1] - v[-1]))

    if count == 0:
        plt.plot(inf_norm, color='g', label='TD_zero (alpha=10/(100+n))')
    else:
        plt.plot(final_state_norm, color='g', label='TD_zero (alpha=10/(100+n))')

    plt.title(title)
    plt.ylabel('norm')
    plt.xlabel('iteration')
    plt.legend()
    plt.show()

print("*"*20)
print("TD(0) alpha=10/(100+n) for policy_c:\n", np.round(v,3))
        
print("*"*20)
print("V of policy_c:\n", np.round(V_c,3))


# %%
# section h
print('section h')
def update_eligibilty_traces(eligibilty, sample, lamda, gamma):
    s_t, a_t, r_t, s_tag = sample
    eligibilty = eligibilty * lamda * gamma
    eligibilty[s_t] = eligibilty[s_t] + 1
    return eligibilty
    
def calc_delta_t(sample, v, gamma):
    s_t, a_t, r_t, s_tag = sample
    delta = r_t + gamma*v[s_tag] - v[s_t]
    return delta
    
def update_v_TD_lambda(estimated_v, eligibility ,alpha, delta_t):
    return estimated_v + alpha * eligibility * delta_t

# v = np.zeros([32])
# e = np.zeros([32])
# n = np.zeros([32])
# N = 10000
# plot values
# lamda = 0
# inf_norm = []
# final_state_norm = []
colors = iter(['b', 'r', 'g', 'm', 'y'])


# # %%
# for lam in [0, 0.01, 0.5, 0.9]:
#     inf_norm = []
#     final_state_norm = []
#     for i in range(20):
#         v = np.zeros([32])
#         e = np.zeros([32])
#         n = np.zeros([32])
#         N = 10000
#         inf_norm_i = []
#         final_state_norm_i = []
#         for i in range(N):
#             cost, trajectory = simulation(policy=get_full_policy_c(table=problem_table), table=problem_table, random=True)
#             for sample in trajectory:
#                 s_t, _, _, _ = sample
#                 n[s_t] += 1
#                 alpha = 10/(100 + n[s_t])
#                 delta = calc_delta_t(sample=sample, v=v, gamma=1.0)
#                 e = update_eligibilty_traces(eligibilty=e, sample=sample, lamda=lam, gamma=1.0)
#                 v = update_v_TD_lambda(estimated_v=v, eligibility=e, alpha=alpha, delta_t=delta)
#             # added tracers for plot
#             inf_norm_i.append(np.linalg.norm((V_c - v),ord=np.inf))
#             final_state_norm_i.append(np.linalg.norm(V_c[-1] - v[-1]))
#         inf_norm.append(inf_norm_i)
#         final_state_norm.append(final_state_norm_i)

#     mean_inf_norm = np.mean(inf_norm, axis=0)
#     mean_final_state_norm = np.mean(final_state_norm, axis=0)

#     # print infinte norm
#     plt.plot(mean_final_state_norm, color=next(colors), label=f'lambda = {lam}')

# # plt.title(f'||V^pi_c - V_TD(lambda)||_inf, alpha=10/(100+n)')
# # plt.ylabel('infinite norm')

# # print final state norm
# # plt.plot(final_state_norm)
# plt.title(f'final state: |V^pi_c(s0) - V_TD(lambda)(s0)|, alpha=10/(100+n)')
# plt.ylabel('norm')
# plt.xlabel('iteration')
# plt.legend()
# plt.show()


# print("*"*20)
# print("TD(lambda) alpha=10/(100+n) for policy_c:\n", np.round(v,3))
        
# print("*"*20)
# print("V of policy_c:\n", np.round(V_c,3))



# %%
# section i
print('section i')
def get_action_min(state, Q, epsilon):     # 1-5
    if state == 0:
        return 0
    greedy = np.random.binomial(1, 1-epsilon)
    if greedy:
        #print("greedy", end='-')
        min_a = np.inf
        min_idx = -1
        for a in range(5):
            if is_action_valid(state=state, action=a+1):
                if Q[state, a] < min_a:
                    min_a = Q[state, a]
                    min_idx = a+1
        #print("choosen action:", max_idx)
        return min_idx
    
    #print("random", end='-')
    rand_act = int(np.random.randint(5))+1
    while not is_action_valid(state=state, action=rand_act):
        rand_act = int(np.random.randint(5))+1
    #print("choosen action:", rand_act)
    return rand_act



Q = np.zeros([32,5])
n = np.zeros([32,5])
sample_rate = 1
value_pi_Q_error_lists = []
full_state_value_pi_Q_error_lists = []

labels = iter(['epsilon 0.1', 'epsilon 0.01'])

jobs = 5
states = 2 ** jobs
cost = np.zeros([states])
problem_table = np.array([
    [0.6, 0.5, 0.3, 0.7, 0.1],
    [1,     4,   6,   2,   9]
])

for state in range(states):
    cost[state] = calc_cost_per_state(problem_table=problem_table, state=state)

N = 100_000
epsilon = 0.1

for epsilon in [0.1, 0.01]:
    Q = np.zeros([32,5])
    n = np.zeros([32,5])
    value_pi_Q_error_list = []
    full_state_value_pi_Q_error_list = []
    for i in range(N):
        #print("*"*10)
        state_t = 31
        action_t = get_action_min(state=state_t, Q=Q, epsilon=epsilon)
        while not state_t == 0:
            cost_t, state_tag = single_step_sim(state=state_t, action=action_t, table=problem_table)
            #print("state:", state_t, ", action:", action_t, ", next_state:", state_tag)
            
            f_epsilon = epsilon
            action_tag = get_action_min(state=state_tag, Q=Q, epsilon=f_epsilon)
            reward = cost_t
            
            delta = reward + 1 * Q[state_tag, action_tag-1] - Q[state_t, action_t-1]
            
            n[state_t, action_t-1] = n[state_t, action_t-1] + 1
            alpha = 10/(100+n[state_t, action_t-1])
            # alpha = 0.1
            # alpha = 1/n[state_t, action_t-1]
            
            Q[state_t, action_t-1] = Q[state_t, action_t-1] + alpha * delta
            
            state_t = state_tag
            action_t = action_tag

        if (i+1) % sample_rate == 0:
            p = calc_p(problem_table, np.argmin(Q+1000*(Q==0), axis=1)+1)
            v_pi_Q = get_v(p, cost)
            value_pi_Q_error_list.append(np.linalg.norm((V_uc - v_pi_Q),ord=np.inf))
            full_state_value_pi_Q_error_list.append(np.linalg.norm(V_uc[-1] - v_pi_Q[-1]))



    value_pi_Q_error_lists.append(value_pi_Q_error_list)
    full_state_value_pi_Q_error_lists.append(full_state_value_pi_Q_error_list)
    
plt.plot(value_pi_Q_error_list[0], color='b', label='epsilon 0.1')
plt.plot(value_pi_Q_error_list[1], color='r', label='epsilon 0.01')
# print final state norm

plt.title('||V* - V^{pi_Q}||')
plt.ylabel('infinite norm')
plt.xlabel('iteration')
plt.legend()
plt.show()


plt.plot(full_state_value_pi_Q_error_lists[0], color='b', label='epsilon 0.1')
plt.plot(full_state_value_pi_Q_error_lists[1], color='r', label='epsilon 0.01')
# print final state norm

plt.title('|V*(s0) - argmin Q(s0,a)|')
plt.ylabel('norm')
plt.xlabel('iteration')
plt.legend()
plt.show()
Q = Q - 1000
print(np.round(Q, 3))
 


# %%
final_Q = Q + 1000*(Q==0)
final_Q[0,0] = 0
print((np.argmin(final_Q, axis=1)+1)[1:] == get_full_policy_uc(table=problem_table)[1:])
print(31 - ((np.argmin(final_Q, axis=1)+1)[1:] == get_full_policy_uc(table=problem_table)[1:]).sum())

p = calc_p(problem_table, (np.argmin(final_Q, axis=1)+1))
v_pi_Q = get_v(p, cost)
print(v_pi_Q)



# %%
print(np.min(final_Q, axis=1))


# %%
print(np.max(V_uc - np.min(final_Q, axis=1)))


# %%
np.random.randint(2)


# %%



