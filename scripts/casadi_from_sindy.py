import sympy as sp
import xlsindy
import numpy as np
import casadi as ca
import jax.numpy as jnp


def sympy2casadi(sympy_expr, sympy_var, casadi_var):
    """Convert sympy expression to CasADi using lambdify with custom module mapping"""
    assert casadi_var.is_vector()
    if casadi_var.shape[1] > 1:
        casadi_var = casadi_var.T
    casadi_var_list = ca.vertsplit(casadi_var)
    
    from sympy.utilities.lambdify import lambdify
    
    mapping = {
        'ImmutableDenseMatrix': ca.blockcat,
        'MutableDenseMatrix': ca.blockcat,
        'Abs': ca.fabs,
        'cos': ca.cos,
        'sin': ca.sin,
    }
    f = lambdify(sympy_var, sympy_expr, modules=[mapping, ca])
    return f(*casadi_var_list)

def casadi_acceleration_function(solution, catalog_repartition, symbols_matrix):

    num_coords = symbols_matrix.shape[1]

    expanded_catalog = catalog_repartition.expand_catalog()

    # Filter out terms where solution is less than 1e-5 in absolute value

    mask = np.any(np.abs(solution) >= 1e-5, axis=1)
    solution = solution[mask]
    expanded_catalog = expanded_catalog[mask]

    print("Masked solution:")
    print(solution)

    dynamic_equations = solution.T @ expanded_catalog
    dynamic_equations = dynamic_equations.flatten()

    # Helper function to convert sympy expressions to CasADi

    valid = True

    for i in range(num_coords):
        if str(symbols_matrix[3, i]) not in str(dynamic_equations[i]):
            valid = False

    if valid:
        system_matrix, force_vector = (
            np.empty((num_coords, num_coords), dtype=object),
            np.empty((num_coords, 1), dtype=object),
        )

        for i in range(num_coords):
            equation = dynamic_equations[i]
            for j in range(num_coords):
                equation = equation.collect(symbols_matrix[3, j])
                term = equation.coeff(symbols_matrix[3, j])
                system_matrix[i, j] = -term
                equation -= term * symbols_matrix[3, j]

            force_vector[i, 0] = equation

        # Create CasADi symbolic variables from symbols_matrix
        # Flatten all symbols from the matrix for easier handling
        sympy_symbols_flat = []
        casadi_symbols_flat = []
        
        for i in range(symbols_matrix.shape[0]):
            for j in range(symbols_matrix.shape[1]):
                sym = symbols_matrix[i, j]
                sympy_symbols_flat.append(sym)
                ca_sym = ca.SX.sym(str(sym))
                casadi_symbols_flat.append(ca_sym)
        
        # Add external force symbols (F1, F2, etc.)
        external_force_symbols = []
        external_force_casadi = []
        for j in range(num_coords):
            f_sym = sp.Symbol(f'F_{j+1}')
            sympy_symbols_flat.append(f_sym)
            external_force_symbols.append(f_sym)
            
            ca_f_sym = ca.SX.sym(f'F_{j+1}')
            casadi_symbols_flat.append(ca_f_sym)
            external_force_casadi.append(ca_f_sym)
    
    # Create vertical concatenation of CasADi symbols
        casadi_state = ca.vertcat(*casadi_symbols_flat)
    
        # Convert system_matrix and force_vector to CasADi
        system_matrix_ca = ca.SX.zeros(num_coords, num_coords)
        force_vector_ca = ca.SX.zeros(num_coords, 1)
        
        for i in range(num_coords):
            for j in range(num_coords):
                system_matrix_ca[i, j] = sympy2casadi(
                    system_matrix[i, j], sympy_symbols_flat, casadi_state
                )
            force_vector_ca[i] = sympy2casadi(
                force_vector[i, 0], sympy_symbols_flat, casadi_state
            )
        
        # Solve for accelerations: system_matrix * acc = force_vector
        acc_ca = ca.solve(system_matrix_ca, force_vector_ca)
        
        # Create CasADi Function with external forces as inputs
        # Input: flattened state vector [time, q, q_dot, q_ddot, F1, F2, ...]
        acc_func = ca.Function('acceleration_solver', [casadi_state], [acc_ca])

        return acc_func

## Catalog initialization

time_sym = sp.symbols("t")

num_coordinates = 2

symbols_matrix = xlsindy.symbolic_util.generate_symbolic_matrix(
    num_coordinates, time_sym
)

# Hard-coded Lagrangian catalog for double pendulum
# L = 1/2 (I1 + m1*d1^2 + m2*l1^2) θ̇1^2 + 1/2 (I2 + m2*d2^2) θ̇2^2 
#     + m2*l1*d2 θ̇1 θ̇2 cos(θ1 - θ2) + (m1*d1 + m2*l1)*g cos(θ1) + m2*d2*g cos(θ2)

theta1 = symbols_matrix[1, 0]  # θ1
theta2 = symbols_matrix[1, 1]  # θ2
theta1_d = symbols_matrix[2, 0]  # θ̇1
theta2_d = symbols_matrix[2, 1]  # θ̇2

lagrange_catalog = np.array([ # Wworking ? 
    theta1_d**2,                                    # Term 1: θ̇1^2
    theta2_d**2,                                    # Term 2: θ̇2^2
    theta1_d * theta2_d * sp.cos(theta1 - theta2),   # Term 3: θ̇1 θ̇2 cos(θ1 - θ2)
    sp.cos(theta1),                               # Term 4: cos(θ1)
    sp.cos(theta2),                               # Term 5: cos(θ2)
])

lagrange_catalog = np.array([ # Wworking ? 
    theta1_d**2,                                    # Term 1: θ̇1^2
    theta2_d**2,                                    # Term 2: θ̇2^2
    # theta1_d * theta2_d * sp.cos(theta1),
    # theta1_d * theta2_d * sp.sin(theta1),
    theta1_d * theta2_d * sp.cos(theta2),
    theta1_d * theta2_d * sp.sin(theta2),
    # theta1_d * theta2_d * sp.cos(theta1) * sp.sin(theta2),
    # theta1_d * theta2_d * sp.sin(theta1) * sp.cos(theta2),
    # theta1_d * theta2_d * sp.cos(theta1) * sp.cos(theta2),
    # theta1_d * theta2_d * sp.sin(theta1) * sp.sin(theta2),
    #sp.cos(theta1),                               # Term 4: cos(θ1)
    sp.cos(theta2),                               # Term 5: cos(θ2)
    #sp.sin(theta1),                               # Term 6: sin(θ1)
    sp.sin(theta2),                               # Term 7: sin(θ2)
    #sp.cos(theta1) * sp.sin(theta2),
    #sp.sin(theta1) * sp.cos(theta2),
    #sp.cos(theta1) * sp.cos(theta2),
    #sp.sin(theta1) * sp.sin(theta2),
    theta1_d * sp.cos(theta2),
    theta1_d * sp.sin(theta2),
    theta2_d * sp.cos(theta2),
    theta2_d * sp.sin(theta2),
])

friction_function = np.array(
    [[symbols_matrix[2, x] for x in range(num_coordinates)]]
)


friction_catalog = (
    friction_function.flatten()
)  # Contain only \dot{q}_1 \dot{q}_2

expand_matrix = np.ones((len(friction_catalog), num_coordinates), dtype=int)

catalog_repartition = xlsindy.catalog.CatalogRepartition(
    [
        xlsindy.catalog_base.ExternalForces(
            [[1], [2]], symbols_matrix
        ),
        xlsindy.catalog_base.Lagrange(
            lagrange_catalog, symbols_matrix, time_sym
        ),
        xlsindy.catalog_base.Classical(
            friction_catalog, expand_matrix
        ),
    ]
)


## End of catalog initialization

solution_dict = np.load("regression_solution.pkl", allow_pickle=True)

solution = solution_dict["solution"]
print("Retrieved solution:")
print(solution)

## create a casadi function from the retrieved solution



# ---------------------------------------------------------
# Trajectory Prediction Setup
# ---------------------------------------------------------
num_coords = symbols_matrix.shape[1]

# Define state for dynamics (only positions and velocities)
# State: [q0, q1, ..., qn, q0_dot, q1_dot, ..., qn_dot]
state_dim = 2 * num_coords
x_dyn = ca.SX.sym('x', state_dim)
u_dyn = ca.SX.sym('u', num_coords)  # Control inputs (one per coordinate)

# Extract positions and velocities from state
q = x_dyn[:num_coords]
q_dot = x_dyn[num_coords:]

# Build the full state for acc_func (needs all derivatives from symbols_matrix)
# Construct the vector that acc_func expects including external forces
state_for_acc = []
for i in range(symbols_matrix.shape[0]):
    for j in range(symbols_matrix.shape[1]):
        if i == 0:  # time
            state_for_acc.append(0)
        elif i == 1:  # positions
            state_for_acc.append(q[j])
        elif i == 2:  # velocities
            state_for_acc.append(q_dot[j])
        elif i == 3:  # accelerations - placeholder
            state_for_acc.append(0)

# Add external forces/controls at the end
for j in range(num_coords):
    state_for_acc.append(u_dyn[j])

state_for_acc_vec = ca.vertcat(*state_for_acc)

# Compute accelerations
q_ddot = acc_func(state_for_acc_vec)

# Full dynamics: dx/dt = [q_dot, q_ddot]
x_dot = ca.vertcat(q_dot, q_ddot)

# Create dynamics function with control input
f_dynamics = ca.Function('f_dynamics', [x_dyn, u_dyn], [x_dot])

print("\n" + "="*60)
print("CasADi Functions Created Successfully:")
print("="*60)
print(f"  - acc_func: Computes accelerations from full state")
print(f"  - f_dynamics: Computes state derivatives [q_dot, q_ddot]")
print(f"  - State dimension: {state_dim} (positions and velocities)")
print("="*60)

# ---------------------------------------------------------
# Trajectory Optimization Setup
# ---------------------------------------------------------

# Configuration (Frequency Setup)
T_horizon = 10.0         # Optimization time horizon (s)
freq_ctrl = 100         # Control Frequency (Hz)
freq_sim  = 500         # Physics/RK4 Frequency (Hz)

# Calculated parameters
ratio = int(freq_sim / freq_ctrl)  # sub-steps per control interval
N = int(T_horizon * freq_ctrl)     # Total control intervals
dt_sim = 1.0 / freq_sim            # Integration step

print(f"\nOptimization Setup:")
print(f"  - Horizon: {T_horizon}s")
print(f"  - Control steps: {N} at {freq_ctrl}Hz")
print(f"  - Integration: {ratio} RK4 steps per control at {freq_sim}Hz")
print(f"  - Total physics steps: {N*ratio}")

# ---------------------------------------------------------
# RK4 Integrator
# ---------------------------------------------------------

# A. Define ONE step of RK4 (at high frequency)
# Use the fixed time step dt_sim directly
k1 = f_dynamics(x_dyn, u_dyn)
k2 = f_dynamics(x_dyn + dt_sim/2*k1, u_dyn)
k3 = f_dynamics(x_dyn + dt_sim/2*k2, u_dyn)
k4 = f_dynamics(x_dyn + dt_sim*k3, u_dyn)
x_next_step = x_dyn + (dt_sim/6) * (k1 + 2*k2 + 2*k3 + k4)

# Create a function for a single high-freq step
F_rk4_single = ca.Function('F_rk4_single', [x_dyn, u_dyn], [x_next_step], 
                            ['x', 'u'], ['x_next'])

# B. Define the COMPOUND step (Control Interval)
# Chain 'ratio' steps together with the SAME control input
x_temp = x_dyn
for _ in range(ratio):
    x_temp = F_rk4_single(x_temp, u_dyn)

# This function takes us from t -> t + (1/freq_ctrl)s using multiple internal steps
F_control_interval = ca.Function('F_interval', [x_dyn, u_dyn], [x_temp])

# ---------------------------------------------------------
# Trajectory Optimization Problem
# ---------------------------------------------------------

opti = ca.Opti()

# Variables
X = opti.variable(state_dim, N+1)  # State trajectory (sampled at control freq)
U = opti.variable(num_coords, N)   # Control trajectory (sampled at control freq)

# ---------------------------------------------------------
# Constraints
# ---------------------------------------------------------

# A. Dynamics Constraints
for k in range(N):
    # X[k+1] must match the result of integrating X[k] with control U[k]
    opti.subject_to(X[:, k+1] == F_control_interval(X[:, k], U[:, k]))

# B. Boundary Conditions
# Start: pendulum hanging down
opti.subject_to(X[:, 0] == ca.vertcat(*([0.0]*state_dim)))  # All zeros

# End: pendulum pointing up (depends on your system configuration)
# For double pendulum: target both angles at pi (pointing up)
target_state = [np.pi] * num_coords + [0.0] * num_coords
opti.subject_to(X[:, -1] == ca.vertcat(*target_state))

# C. Control Limits
u_max = 15.0  # Maximum control force/torque
opti.subject_to(opti.bounded(-u_max, U, u_max))

# D. State Limits (angle and velocity bounds)
# Uncomment and adjust as needed:
# opti.subject_to(opti.bounded(-10.0, X[num_coords:, :], 10.0))  # velocity limits

# ---------------------------------------------------------
# Objective
# ---------------------------------------------------------

# Minimize control effort (sum of squared control inputs)
cost = ca.sumsqr(U)

opti.minimize(cost)

# ---------------------------------------------------------
# Solve
# ---------------------------------------------------------

p_opts = {"expand": True}
s_opts = {"max_iter": 2000, "tol": 1e-4, "print_level": 5}
opti.solver('ipopt', p_opts, s_opts)

print("\n" + "="*60)
print("Starting Trajectory Optimization...")
print("="*60)

try:
    sol = opti.solve()
    
    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    import matplotlib.pyplot as plt
    
    t_grid = np.linspace(0, T_horizon, N+1)
    x_opt = sol.value(X)
    u_opt = sol.value(U)
    
    # Create subplots for each state variable + controls
    n_plots = state_dim + num_coords
    fig, axs = plt.subplots(n_plots, 1, figsize=(10, 3*n_plots), sharex=True)
    
    if n_plots == 1:
        axs = [axs]
    
    # Plot positions
    for i in range(num_coords):
        axs[i].plot(t_grid, x_opt[i, :])
        axs[i].set_ylabel(f'q{i} (rad)')
        axs[i].axhline(np.pi, color='r', linestyle='--', alpha=0.5, label='target')
        axs[i].grid(True)
        axs[i].legend()
    
    # Plot velocities
    for i in range(num_coords):
        axs[num_coords + i].plot(t_grid, x_opt[num_coords + i, :])
        axs[num_coords + i].set_ylabel(f'q{i}_dot (rad/s)')
        axs[num_coords + i].grid(True)
    
    # Plot controls
    for i in range(num_coords):
        # Duplicate last control for step plot
        u_plot = np.append(u_opt[i, :], u_opt[i, -1])
        axs[state_dim + i].step(t_grid, u_plot, where='post')
        axs[state_dim + i].axhline(u_max, color='r', linestyle='--', alpha=0.5)
        axs[state_dim + i].axhline(-u_max, color='r', linestyle='--', alpha=0.5)
        axs[state_dim + i].set_ylabel(f'u{i} (N/Nm)')
        axs[state_dim + i].grid(True)
    
    axs[-1].set_xlabel('Time (s)')
    axs[0].set_title(f'Swing Up Trajectory: Ctrl {freq_ctrl}Hz | Phys {freq_sim}Hz')
    
    plt.tight_layout()
    plt.savefig('trajectory_optimization.png', dpi=150)
    print("\n" + "="*60)
    print("Optimization Successful!")
    print("="*60)
    print(f"  - Plot saved to: trajectory_optimization.png")
    print(f"  - Final cost: {sol.value(cost):.4f}")
    print("="*60)
    plt.show()
    
except RuntimeError as e:
    print("\n" + "="*60)
    print("Solver failed!")
    print("="*60)
    print(f"Error: {e}")
    print("\nSuggestions:")
    print("  - Relax boundary constraints")
    print("  - Increase T_horizon")
    print("  - Adjust initial guess")
    print("  - Check if dynamics are physically feasible")
    print("="*60)

