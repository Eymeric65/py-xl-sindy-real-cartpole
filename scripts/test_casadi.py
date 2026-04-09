import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Configuration (Frequency Setup)
# ---------------------------------------------------------
T_horizon = 2.0         # Swing up time (s)
freq_ctrl = 100         # Control Frequency (Hz)
freq_sim  = 500         # Physics/RK4 Frequency (Hz)

# Calculated parameters
ratio = int(freq_sim / freq_ctrl)  # 500 / 100 = 5 sub-steps
N = int(T_horizon * freq_ctrl)     # Total control intervals (200)
dt_sim = 1.0 / freq_sim            # Integration step (0.002s)

print(f"Optimization: {N} control steps.")
print(f"Integration: {ratio} RK4 steps per control (Total {N*ratio} physics steps).")

# ---------------------------------------------------------
# 2. CasADi Setup
# ---------------------------------------------------------
opti = ca.Opti()

# Variables
X = opti.variable(4, N+1) # State trajectory (sampled at 100Hz)
U = opti.variable(1, N)   # Control trajectory (sampled at 100Hz)

# ---------------------------------------------------------
# 3. Dynamics (Lagrangian Cartpole)
# ---------------------------------------------------------
x_sym = ca.SX.sym('x', 2) 
dx_sym = ca.SX.sym('dx', 2)
u_sym = ca.SX.sym('u', 1)

input_vec = ca.vertcat(x_sym, dx_sym)


# Physics Constants
mc, mp, l, g = 1.0, 0.1, 0.5, 9.81

# Unpack
pos, theta, dpos, dtheta = x_sym[0], dx_sym[0], x_sym[1], dx_sym[1]

# Equations of Motion
s = ca.sin(theta)
c = ca.cos(theta)
den = mc + mp * s**2

ddpos = (u_sym + mp * s * (l * dtheta**2 + g * c)) / den
ddtheta = (-u_sym * c - mp * l * dtheta**2 * c * s - (mc + mp) * g * s) / (l * den)

rhs = ca.vertcat(dpos, dtheta, ddpos, ddtheta)
f_dynamics = ca.Function('f', [input_vec, u_sym], [rhs])

# ---------------------------------------------------------
# 4. The "Sub-stepped" Integrator
# ---------------------------------------------------------

# A. Define ONE step of RK4 (at 500Hz)
k1 = f_dynamics(input_vec,            u_sym)
k2 = f_dynamics(input_vec + dt_sim/2*k1, u_sym)
k3 = f_dynamics(input_vec + dt_sim/2*k2, u_sym)
k4 = f_dynamics(input_vec + dt_sim*k3,   u_sym)
x_next_step = input_vec + (dt_sim/6) * (k1 + 2*k2 + 2*k3 + k4)

# Create a function for a single 500Hz step
F_rk4_single = ca.Function('F_rk4_single', [input_vec, u_sym], [x_next_step])

# B. Define the COMPOUND step (Control Interval)
# We chain 'ratio' (5) steps together using the SAME control 'u'
x_temp = input_vec
for _ in range(ratio):
    x_temp = F_rk4_single(x_temp, u_sym)

# This function takes us from t -> t + 0.01s (using 5 internal steps)
F_control_interval = ca.Function('F_interval', [input_vec, u_sym], [x_temp])

# ---------------------------------------------------------
# 5. Constraints
# ---------------------------------------------------------
# A. Dynamics Constraints
for k in range(N):
    # X[k+1] must match the result of integrating X[k] for 0.01s
    opti.subject_to(X[:, k+1] == F_control_interval(X[:, k], U[:, k]))

# B. Boundary Conditions
opti.subject_to(X[:, 0] == [0, 0, 0, 0])          # Start Down
opti.subject_to(X[:, -1] == [0, np.pi, 0, 0])     # End Up

# C. Voltage/Force Limits
u_max = 10.0
opti.subject_to(opti.bounded(-u_max, U, u_max))

# D. State Limits (Track length)
opti.subject_to(opti.bounded(-2.0, X[0, :], 2.0))

# ---------------------------------------------------------
# 6. Solve
# ---------------------------------------------------------
opti.minimize(ca.sumsqr(U)) # Minimize energy

p_opts = {"expand": True} 
s_opts = {"max_iter": 2000, "tol": 1e-4}
opti.solver('ipopt', p_opts, s_opts)

try:
    sol = opti.solve()
    
    # ---------------------------------------------------------
    # 7. Visualization
    # ---------------------------------------------------------
    t_grid = np.linspace(0, T_horizon, N+1)
    x_opt = sol.value(X)
    u_opt = sol.value(U)
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    axs[0].plot(t_grid, x_opt[0,:])
    axs[0].set_ylabel('Position (m)')
    axs[0].set_title(f'Swing Up: Ctrl {freq_ctrl}Hz | Phys {freq_sim}Hz')
    axs[0].grid(True)
    
    axs[1].plot(t_grid, x_opt[1,:])
    axs[1].axhline(np.pi, color='r', linestyle='--')
    axs[1].set_ylabel('Angle (rad)')
    axs[1].grid(True)
    
    # Duplicate last control for step plot
    u_plot = np.append(u_opt, u_opt[-1])
    axs[2].step(t_grid, u_plot, where='post')
    axs[2].axhline(u_max, color='r', linestyle='--', alpha=0.5)
    axs[2].axhline(-u_max, color='r', linestyle='--', alpha=0.5)
    axs[2].set_ylabel('Control (V)')
    axs[2].set_xlabel('Time (s)')
    axs[2].grid(True)
    
    plt.show()
    
except RuntimeError:
    print("Solver failed. Relax constraints or increase Horizon.")