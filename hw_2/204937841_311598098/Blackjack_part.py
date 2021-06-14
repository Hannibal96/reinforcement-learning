# %%
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
from mpl_toolkits.mplot3d import Axes3D


# %%
P_dealer = (1/13) * np.matrix([
    #2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 Bust  
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 2
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0], # 3
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 0, 0, 0, 0, 0, 0, 0], # 4
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 0, 0, 0, 0, 0, 0], # 5
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 0, 0, 0, 0, 0], # 6
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 0, 0, 0, 0], # 7
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 0, 0, 0], # 8
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 0, 0], # 9
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 0], # 10
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1], # 11
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 5], # 12
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 6], # 13
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 7], # 14
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 8], # 15
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 9], # 16
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13,0, 0, 0, 0, 0], # 17
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13,0, 0, 0, 0], # 18
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13,0, 0, 0], # 19
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,13, 0, 0], # 20
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,13, 0], # 21
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 13], # Bust
])
f_P_dealer = P_dealer ** 10
              
#print(f_P_dealer)


# %%
P_player_hit = (1/13) * np.matrix([
    #4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 Bust  
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 0, 0, 0, 0, 0, 0, 0], # 4
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 0, 0, 0, 0, 0, 0], # 5
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 0, 0, 0, 0, 0], # 6
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 0, 0, 0, 0], # 7
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 0, 0, 0], # 8
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 0, 0], # 9
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 0], # 10
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1], # 11
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 5], # 12
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 6], # 13
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 7], # 14
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 8], # 15
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 9], # 16
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,10], # 17
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,11], # 18
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,12], # 19
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,13], # 20
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,13], # 21
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,13], # Bust
])
P_player_stick = np.matrix([
    #4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 Stick  
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 4
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 5
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 6
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 7
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 8
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 9
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 10
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 11
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 12
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 13
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 14
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 15
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 16
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 17
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 18
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 19
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 20
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 21
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # Stick
])


# %%
R_player_hit = (-1/13) * np.matrix([   
    [0], # 4
    [0], # 5
    [0], # 6
    [0], # 7
    [0], # 8
    [0], # 9
    [0], # 10
    [1], # 11
    [5], # 12
    [6], # 13
    [7], # 14
    [8], # 15
    [9], # 16
    [10], # 17
    [11], # 18
    [12], # 19
    [13], # 20
    [13], # 21
    [0], # Bust
])


# %%
def calc_player_stick_reward(f_P_dealer):
    R_player_stick = np.zeros([19, 10])
    for player_sum in range(0,19): # 4-21 + bust
        for dealer_sum in range(0,10): # 2-11
            delaer_bust = f_P_dealer[dealer_sum, 20]
            draw = f_P_dealer[dealer_sum, player_sum+2]
            player_win = f_P_dealer[dealer_sum, 0:player_sum+2].sum() # win by points
            player_lose = 1-draw-delaer_bust-player_win # lose by points
            R_player_stick[player_sum, dealer_sum] = 0*draw + 1*(delaer_bust+player_win) -1 * (player_lose)
            if player_sum == 18:
                R_player_stick[player_sum, dealer_sum] = 0
    return R_player_stick

R_player_stick = calc_player_stick_reward(f_P_dealer)
print(R_player_stick)


# %%
def conver_value(card):
    if card == 1:
        return 11
    if card == 11 or card == 12 or card == 13:
        return 10
    return card

def simulator(player_sum_in, dealer_sum_in, hit=None):
    total_reward = 0
    iteration = 100000
    for i in range(iteration):
        dealer_sum = dealer_sum_in
        player_sum = player_sum_in
        
        if hit:
            player_sum += conver_value(np.random.randint(1,14))
        if player_sum > 21:
            total_reward += -1
            continue
            
        while dealer_sum < 17:
            dealer_sum += conver_value(np.random.randint(1,14))
        curr_reward = 0
        if dealer_sum > 21:
            curr_reward = 1
        elif dealer_sum < player_sum:
            curr_reward = 1
        elif dealer_sum > player_sum:
            curr_reward = -1
        total_reward += curr_reward
        #print(dealer_sum, curr_reward)
    return (float(total_reward)/iteration)


# %%
simulator(player_sum_in=13, dealer_sum_in=11, hit=True)


# %%
def calc_future_val_hit(P_player_hit_, curr_sum, V_, dealer_sum_):
    sum = 0
    for next_player_sum in range(0, 19):
        sum += P_player_hit[curr_sum, next_player_sum] * V_[next_player_sum, dealer_sum_]
    return sum

def value_iteration(R_player_hit_, R_player_stick_, P_player_hit_, P_player_stick_):
    V = np.ones([19, 10])
    V_new = np.zeros([19, 10])
    while not (V == V_new).all():
        V = V_new.copy()
        for player_sum in range(0, 19):
            for dealer_sum in range(0, 10):
                temp_v_hit = R_player_hit[player_sum] + calc_future_val_hit(P_player_hit_=P_player_hit, curr_sum=player_sum, V_=V, dealer_sum_=dealer_sum)
                temp_v_stick = R_player_stick[player_sum, dealer_sum] + V[18, dealer_sum]
                V_new[player_sum, dealer_sum] = max(temp_v_hit, temp_v_stick)
        #print(V)
        #print("*"*50)
    return V


# %%
optimal_v = value_iteration(R_player_hit_=R_player_hit, R_player_stick_=R_player_stick, P_player_hit_=P_player_hit, P_player_stick_=P_player_stick)
print(np.round(optimal_v,3))


# %%
def get_policy(R_player_hit_, R_player_stick_, P_player_hit_, P_player_stick_, optimal_V):
    policy = np.zeros([19, 10])
    for player_sum in range(0, 19):
        for dealer_sum in range(0, 10):
            temp_v_hit = R_player_hit[player_sum] + calc_future_val_hit(P_player_hit_=P_player_hit, curr_sum=player_sum, V_=optimal_V, dealer_sum_=dealer_sum)
            temp_v_stick = R_player_stick[player_sum, dealer_sum] + optimal_V[18, dealer_sum]
            if temp_v_hit > temp_v_stick:
                policy[player_sum, dealer_sum] = 1
            else:
                policy[player_sum, dealer_sum] = -1
    return policy

optimal_policy = get_policy(R_player_hit_=R_player_hit, R_player_stick_=R_player_stick, 
                            P_player_hit_=P_player_hit, P_player_stick_=P_player_stick,optimal_V=optimal_v)
print(optimal_policy)

# %%
for i in range(2,12):
    stick_val = simulator(player_sum_in=13, dealer_sum_in=i)
    hit_val = simulator(player_sum_in=13, dealer_sum_in=i, hit=True)
    #print("stick:", stick_val)
    #print("hit:", hit_val)
    if hit_val > stick_val:
        print("hit better")
    else:
        print("stick better")


# %%
def plot_policy(value, policy): 

    player = np.arange(4,23)
    dealer = np.arange(2,12)
    X, Y = np.meshgrid(dealer, player)
    grid = np.array([X, Y])

    fig = plt.figure(figsize = (16,8))
    fig.suptitle(f'Blackjack Optimals')

    #Surface plot
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(X, Y, value, alpha=1, cmap = 'Spectral', edgecolor = 'none')
    ax.set_title(f'Optimal Value function [player sum vs. dealer card]')


    ax.view_init(45, 340)
    ax.set_xlabel('dealer card')
    ax.set_ylabel('player sum')
    ax.set_xticks(np.arange(2,12))
    ax.set_yticks(np.arange(4,22))

    #Colormesh plot
    ax = fig.add_subplot(1, 2, 2)
    c = ax.pcolormesh(X,Y,policy, cmap = 'binary_r')
    ax.set_xlabel('dealer card')
    ax.set_ylabel('player sum')
    ax.set_xticks(np.arange(2,12))
    ax.set_yticks(np.arange(4,22))
    cbar = plt.colorbar(c, orientation="horizontal")
    cbar.set_ticks([-1, 1])
    cbar.set_ticklabels(["Stick", "Hit"])
    ax.set_title(f'Player Policy [player sum vs. dealer card]')

    plt.show()


plot_policy(optimal_v, optimal_policy)


