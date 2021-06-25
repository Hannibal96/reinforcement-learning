from lspi import *
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from data_transformer import *
from data_collector import *


# ## Section 2.2 ## #

num_samples = 150
delta = 2.0

p_axis = np.linspace(-1.2-delta, 0.6, num_samples)     # [-1.2, 0.6]
v_axis = np.linspace(-0.7-delta, 0.7, num_samples)     # [-0.7, 0.7]

states = None
for p in p_axis:
    for v in v_axis:
        a = np.array([p, v])
        a = np.expand_dims(a, axis=0)
        if states is None:
            states = a.copy()
            continue
        states = np.concatenate((states, a))

RBF = RadialBasisFunctionExtractor([12, 10])
encoded_states = RBF.encode_states_with_radial_basis_functions(states=states)

def Encode(p_axis, v_axis, RBF):
    Z_1 = np.zeros((len(p_axis), len(v_axis)))
    Z_2 = np.zeros((len(p_axis), len(v_axis)))
    for idx_p, p in enumerate(p_axis):
        for idx_v, v in enumerate(v_axis):
            state = np.array([p, v])
            state = np.expand_dims(state, axis = 0)
            features = RBF.encode_states_with_radial_basis_functions(states=state)
            Z_1[idx_p, idx_v] = features[0, 0]
            Z_2[idx_p, idx_v] = features[0, 1]

    return Z_1, Z_2


fig = plt.figure()
ax = plt.axes(projection='3d')

p_grid, v_grid = np.meshgrid(p_axis, v_axis)

Z_1, Z_2 = Encode(p_axis, v_axis, RBF)

ax = plt.axes(projection='3d')
ax.plot_surface(p_grid, v_grid, Z_1, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_title('RFB Feature #1')
plt.show()

ax = plt.axes(projection='3d')
ax.plot_surface(p_grid, v_grid, Z_2, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_title('RFB Feature #2')
plt.show()

# ## Section 3.2 ## #

env = MountainCarWithResetEnv()
dc = DataCollector(env_with_reset=env)
random_data = dc.collect_data(number_of_samples=100000)

print("State mean: " + str(random_data[0].mean(axis=0)) )
print("State std: " + str(random_data[0].std(axis=0)) )


# ## Section c ## #
LP = LinearPolicy(number_of_state_features=2, number_of_actions=3, include_bias=True)







