import sympy as sp
import xlsindy
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

from typing import Tuple, Callable

DEBUG=True

IDEAL_DYNAMICS = False  # If True, uses the known cartpole dynamics instead of the SINDy-derived one. Useful for debugging the rest of the pipeline.

def generate_acceleration_function_casadi(
    regression_solution: np.ndarray,
    catalog_repartition, 
    symbol_matrix: np.ndarray,
) -> Tuple[ca.Function, bool]:
    """
    Generates a CasADi Function for acceleration: acc = f(u, q, dq)
    
    Returns:
        ca.Function: A CasADi function named 'acc_solver'
                     Input:  [u; q; dq] (Vertical concatenation, size 3*N)
                     Output: [ddq]      (Size N)
        bool: Valid generation
    """
    num_coords = symbol_matrix.shape[1]

    # 1. Expand and Filter (Your original logic)
    mask = np.any(np.abs(regression_solution) >= 1e-5, axis=1)
    regression_solution = regression_solution[mask]
    expanded_catalog = catalog_repartition.expand_catalog()
    expanded_catalog = expanded_catalog[mask]

    dynamic_equations = regression_solution.T @ expanded_catalog

    dynamic_equations = dynamic_equations.flatten()

    # Print the discovered dynamic equations
    print("\n" + "="*70)
    print("DISCOVERED DYNAMIC EQUATIONS (from SINDy)")
    print("="*70)
    for i in range(num_coords):
        print(f"\nEquation {i+1}:")
        sp.pprint(dynamic_equations[i], use_unicode=True)
    print("\n" + "="*70 + "\n")

    valid = True

    # Validation check
    for i in range(num_coords):
        if str(symbol_matrix[3, i]) not in str(dynamic_equations[i]):
            valid = False

    if not valid:
        return None, False

    # 2. Algebraic Extraction of A and b (A * acc = b)
    # Using lists of lists for compatibility with SymPy Matrix
    system_matrix_list = [[0] * num_coords for _ in range(num_coords)]
    force_vector_list = [0] * num_coords

    for i in range(num_coords):
        equation = dynamic_equations[i]
        for j in range(num_coords):
            # symbol_matrix[3, j] is the acceleration qdd_j(t)
            acc_term = symbol_matrix[3, j]
            
            equation = equation.collect(acc_term)
            term = equation.coeff(acc_term)
            
            # A[i,j]
            system_matrix_list[i][j] = -term 
            
            # Remove acceleration term to get the remainder
            equation -= term * acc_term

        # b[i]
        force_vector_list[i] = equation

    # Convert to SymPy Matrices
    # This is crucial because 'lambdify' handles SymPy Matrices better than numpy arrays
    A_sym = sp.Matrix(system_matrix_list)
    b_sym = sp.Matrix(force_vector_list)

    # -------------------------------------------------------
    # 3. The Bridge: SymPy -> CasADi
    # -------------------------------------------------------
    
    # A. Define CasADi symbolic variables
    # We create the variables that will be the inputs to the final function
    u_ca  = ca.SX.sym('u', num_coords)
    q_ca  = ca.SX.sym('q', num_coords)
    dq_ca = ca.SX.sym('dq', num_coords)
    
    # B. define the SymPy arguments
    # These must match the order inside A_sym and b_sym
    # Flatten rows 0, 1, 2 of symbol_matrix: [F0, F1..., q0, q1..., qd0, qd1...]
    sympy_args = (
        list(symbol_matrix[0, :]) + 
        list(symbol_matrix[1, :]) + 
        list(symbol_matrix[2, :])
    )
    
    # C. Prepare CasADi arguments for the lambdified function
    # lambdify will generate a function expecting N scalars.
    # We split our CasADi vectors into a list of scalar SX symbols.
    casadi_args_list = (
        ca.vertsplit(u_ca) + 
        ca.vertsplit(q_ca) + 
        ca.vertsplit(dq_ca)
    )
    
    # D. Setup the Module Mapping
    # This tells SymPy how to translate Matrix types to CasADi types
    mapping = {
        'ImmutableDenseMatrix': ca.blockcat,
        'MutableDenseMatrix': ca.blockcat,
        'Abs': ca.fabs,
        'sin': ca.sin,
        'cos': ca.cos,
        'sqrt': ca.sqrt,
        'exp': ca.exp,
    }
    
    # E. Lambdify A and b
    # Note: explicit 'modules=[mapping, "casadi"]' ensures it uses standard CasADi math
    # AND your matrix mapping.
    print("Converting Mass Matrix (A) to CasADi...")
    func_A = sp.lambdify(sympy_args, A_sym, modules=[mapping, ca])
    
    print("Converting Force Vector (b) to CasADi...")
    func_b = sp.lambdify(sympy_args, b_sym, modules=[mapping, ca])
    
    # F. Evaluate to get the CasADi Expression Graph
    # We call the python functions with our CasADi symbolic scalars.
    # This builds the computational graph.
    A_expr = func_A(*casadi_args_list)
    b_expr = func_b(*casadi_args_list)
    
    # -------------------------------------------------------
    # 4. Symbolic Solve & Function Creation
    # -------------------------------------------------------
    
    print("Building Symbolic Solver (A \\ b)...")
    # CasADi symbolic solve (uses LU factorization under the hood)
    acc_expr = ca.solve(A_expr, b_expr)
    
    # Create the final function
    # Inputs: u, q, dq
    # Output: ddq
    casadi_func = ca.Function('acc_solver', [u_ca, q_ca, dq_ca], [acc_expr], 
                              ['u', 'q', 'dq'], ['ddq'])
    
    return casadi_func, valid

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

accel_function,valid = generate_acceleration_function_casadi(
    regression_solution=solution,
    catalog_repartition=catalog_repartition,
    symbol_matrix=symbols_matrix,
        )

if not valid:
    print("The generated function is NOT valid. Check the equations.")
    exit()
## create a casadi function from the retrieved solution
# ---------------------------------------------------------
# 1. Configuration (Frequency Setup)
# ---------------------------------------------------------
T_horizon = 3.0         # Swing up time (s)
freq_ctrl = 100         # Control Frequency (Hz)
freq_sim  = 500         # Physics/RK4 Frequency (Hz)

# Calculated parameters
ratio = int(freq_sim / freq_ctrl)  # 500 / 100 = 5 sub-steps
N = int(T_horizon * freq_ctrl)     # Total control intervals (200)
dt_sim = 1.0 / freq_sim            # Integration step (0.002s)

print(f"Optimization: {N} control steps.")
print(f"Integration: {ratio} RK4 steps per control (Total {N*ratio} physics steps).")



# ---------------------------------------------------------
# 2. Dynamics (Lagrangian Cartpole)
# ---------------------------------------------------------
x_sym = ca.SX.sym('x', 2) 
dx_sym = ca.SX.sym('dx', 2)
u_sym = ca.SX.sym('u', 1)

ddx_sym = accel_function(u_sym, x_sym, dx_sym)

input_vec = ca.vertcat(x_sym, dx_sym)
output_vec = ca.vertcat(dx_sym, ddx_sym)

# print("input and output shapes:")
# print(input_vec.shape)
# print(output_vec.shape)

# Physics Constants
mc, mp, l, g = 1.0, 0.1, 0.5, 9.81

# Unpack (CORRECTED)
pos = x_sym[0]      # position
theta = x_sym[1]    # angle
dpos = dx_sym[0]    # velocity
dtheta = dx_sym[1]  # angular velocity

# Equations of Motion
s = ca.sin(theta)
c = ca.cos(theta)
den = mc + mp * s**2

ddpos = (u_sym + mp * s * (l * dtheta**2 + g * c)) / den
ddtheta = (-u_sym * c - mp * l * dtheta**2 * c * s - (mc + mp) * g * s) / (l * den)

rhs = ca.vertcat(dpos, dtheta, ddpos, ddtheta)

if IDEAL_DYNAMICS:
    f_dynamics = ca.Function('f', [input_vec, u_sym], [rhs])
else:
    f_dynamics = ca.Function('f', [input_vec, u_sym], [output_vec])

# ---------------------------------------------------------
# 3. The "Sub-stepped" Integrator
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

if DEBUG:
    # Test the integrator with a 2-second forward simulation
    print("\n" + "="*60)
    print("TESTING FORWARD SIMULATION (2 seconds)")
    print("="*60)
    
    # Simulation parameters
    T_test = 20.0  # 2 seconds
    dt_test = 1.0 / freq_ctrl  # 0.01s
    n_steps = int(T_test / dt_test)  # 200 steps
    
    # Initialize storage
    time_vec = np.linspace(0, T_test, n_steps + 1)
    state_history = np.zeros((4, n_steps + 1))
    control_history = np.zeros(n_steps)
    
    # Initial condition: pendulum slightly displaced
    state_history[:, 0] = [0.0, np.pi-0.2, 0.0, 0.0]  # [pos, theta, dpos, dtheta]
    
    # Control input: sinusoidal force
    for i in range(n_steps):
        control_history[i] = 0.0 * np.sin(2 * np.pi * 1.0 * time_vec[i])  # 1 Hz sine wave, amplitude 2N
    
    # Forward simulation
    print(f"Simulating {n_steps} steps at {freq_ctrl} Hz...")
    for k in range(n_steps):
        current_state = ca.DM(state_history[:, k])
        current_control = ca.DM([control_history[k]])
        
        # Integrate one step
        next_state = F_control_interval(current_state, current_control)
        state_history[:, k+1] = np.array(next_state).flatten()
    
    print("Simulation complete!")
    print("="*60 + "\n")
    
    # Plot results
    fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    
    axs[0].plot(time_vec, state_history[0, :], 'b-', linewidth=2)
    axs[0].set_ylabel('Position (m)')
    axs[0].set_title('Forward Simulation Test (2 seconds)')
    axs[0].grid(True)
    
    axs[1].plot(time_vec, state_history[1, :], 'r-', linewidth=2)
    axs[1].set_ylabel('Angle (rad)')
    axs[1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axs[1].grid(True)
    
    axs[2].plot(time_vec, state_history[2, :], 'b-', linewidth=2)
    axs[2].set_ylabel('Velocity (m/s)')
    axs[2].grid(True)
    
    axs[3].plot(time_vec, state_history[3, :], 'r-', linewidth=2)
    axs[3].set_ylabel('Angular Vel (rad/s)')
    axs[3].grid(True)
    
    control_plot = np.append(control_history, control_history[-1])
    axs[4].step(time_vec, control_plot, 'g-', linewidth=2, where='post')
    axs[4].set_ylabel('Control (N)')
    axs[4].set_xlabel('Time (s)')
    axs[4].grid(True)
    
    plt.tight_layout()
    plt.savefig('forward_simulation_test.png', dpi=150)
    print("Plot saved as 'forward_simulation_test.png'")
    plt.show()
    
    exit()  # Exit after debug test



# ---------------------------------------------------------
# 4. CasADi Setup
# ---------------------------------------------------------
opti = ca.Opti()

# Variables
X = opti.variable(4, N+1) # State trajectory (sampled at 100Hz)
U = opti.variable(1, N)   # Control trajectory (sampled at 100Hz)

# ---------------------------------------------------------
# 5. Constraints
# ---------------------------------------------------------
# A. Dynamics Constraints
for k in range(N):
    # X[k+1] must match the result of integrating X[k] for 0.01s
    opti.subject_to(X[:, k+1] == F_control_interval(X[:, k], U[:, k]))

# B. Boundary Conditions
opti.subject_to(X[:, 0] == [0, 0, 0, 0])          # Start Down
opti.subject_to(X[:, -1] == [0, 0, 0, 0])     # End Up

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