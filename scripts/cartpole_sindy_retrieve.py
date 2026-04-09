import xlsindy
import sympy as sp
import numpy as np
import pandas as pd
import glob
import os
import time
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
from .generate_trajectory import generate_theoretical_trajectory
from xlsindy.optimization import lasso_regression

from sklearn.linear_model import LassoCV, Lasso

def lasso_regression_tuned(
    whole_exp_matrix: np.ndarray,
    mask: np.ndarray,
    max_iterations: int = 10**4,
    tolerance: float = 1e-5,
    eps: float = 5e-2,
) -> np.ndarray:
    """
    Performs Lasso regression to select sparse features with proper normalization.

    Parameters:
        exp_matrix (np.ndarray): Experimental matrix.
        mask (int): the forces column
        max_iterations (int): Maximum number of iterations.
        tolerance (float): Convergence tolerance.
        eps (float): Regularization parameter.

    Returns:
        np.ndarray: Coefficients of the fitted model. shape (-1,)
    """

    exp_matrix, forces_vector = xlsindy.optimization.amputate_experiment_matrix(whole_exp_matrix, mask)

    # Normalize features (columnwise standardization)
    # Save mean and std for unnormalization
    X_mean = np.mean(exp_matrix, axis=0)
    X_std = np.std(exp_matrix, axis=0)
    
    # Avoid division by zero for constant columns
    X_std[X_std == 0] = 1.0
    
    # Normalize: X_norm = (X - mean) / std
    exp_matrix_normalized = (exp_matrix - X_mean) / X_std
    
    print(f"Feature normalization: mean range [{X_mean.min():.3e}, {X_mean.max():.3e}], std range [{X_std.min():.3e}, {X_std.max():.3e}]")

    y = forces_vector[:, 0]
    
    # Fit LassoCV on normalized data
    model_cv = LassoCV(
        cv=5, random_state=0, max_iter=max_iterations, eps=eps, tol=tolerance, verbose=2
    )
    model_cv.fit(exp_matrix_normalized, y)
    best_alpha = model_cv.alpha_
    
    print(f"Best alpha selected by CV: {best_alpha:.6e}")

    # Fit final Lasso model on normalized data
    lasso_model = Lasso(
        alpha=best_alpha,
        max_iter=max_iterations,
        tol=tolerance,
    )
    lasso_model.fit(exp_matrix_normalized, y)

    # Get normalized coefficients
    coef_normalized = lasso_model.coef_
    
    # Unnormalize coefficients: β_original = β_normalized / std
    # This is because X_norm = X / std, so β must be scaled inversely
    coef_unnormalized = coef_normalized / X_std
    
    result_solution = np.reshape(coef_unnormalized, (-1, 1))

    result_solution = xlsindy.optimization.populate_solution(result_solution, mask)

    return result_solution

## The transformation for cartpole (identity for now)
def mujoco_transform(pos, vel, acc):
    return pos, vel, acc

def inverse_mujoco_transform(pos, vel, acc):
    if acc is not None:
        return pos, vel, acc
    else:
        return pos, vel, None


data_ratio = 1

## Create the catalog (Mandatory part)
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

## Import the data from the cartpole recorder

# Search for specific terms in the catalog
theta1_d = symbols_matrix[2, 0]  # θ̇1
theta2_d = symbols_matrix[2, 1]  # θ̇2

# Find indices of θ̇1² and θ̇2² in the lagrange catalog
theta1_d_squared_idx = None
theta2_d_squared_idx = None

for idx, term in enumerate(lagrange_catalog):
    if term.equals(theta1_d**2):
        theta1_d_squared_idx = idx
        print(f"Found θ̇1² at index {idx}")
    if term.equals(theta2_d**2):
        theta2_d_squared_idx = idx
        print(f"Found θ̇2² at index {idx}")

if theta1_d_squared_idx is None or theta2_d_squared_idx is None:
    raise ValueError("Could not find θ̇1² and θ̇2² terms in catalog")

#kinetic_energy_indices = np.array([theta1_d_squared_idx, theta2_d_squared_idx]) 
kinetic_energy_indices = np.array([theta2_d_squared_idx]) 


# Load all cartpole data CSV files
data_files = glob.glob("cartpole_data_*.csv")
if not data_files:
    raise FileNotFoundError("No cartpole data files found matching 'cartpole_data_*.csv'")

print(f"Found {len(data_files)} data file(s):")
for file in sorted(data_files):
    print(f"  - {file}")

# Load and concatenate all CSV files with proper time offset
df_list = []
time_offset = 0.0

truncate = 10

for file in sorted(data_files):
    df_temp = pd.read_csv(file)

    df_temp = df_temp[truncate:-truncate]
    
    # Add time offset to make time continuous
    if time_offset > 0:
        df_temp['time'] = df_temp['time'] + time_offset
    
    # Update offset for next file (add a small gap between files)
    time_offset = df_temp['time'].max() + 0.01  # 10ms gap between sessions
    
    df_list.append(df_temp)

df = pd.concat(df_list, ignore_index=True)
print(f"Loaded total of {len(df)} data points from {len(data_files)} file(s)")
print(f"Total time span: {df['time'].min():.2f}s to {df['time'].max():.2f}s")

# Cartpole coordinates: [cart_x, angle]
modules_name = ["Cart", "Pole"]
n_coordinates = len(modules_name)

# Extract time
train_time = df['time'].values
m_time = len(train_time)

# Initialize arrays with shape (m_time, n_coordinates)
train_position = np.zeros((m_time, n_coordinates))
train_velocity = np.zeros((m_time, n_coordinates))
train_forces = np.zeros((m_time, n_coordinates))

# Fill position arrays from cartpole CSV columns
train_position[:, 0] = df['cart_x'].values      # Cart position
train_position[:, 1] = df['angle'].values       # Pole angle (raw, will unwrap)

# Unwrap the angle to handle -π to π jumps
train_position[:, 1] = np.unwrap(train_position[:, 1])
print(f"Angle unwrapped: range from {train_position[:, 1].min():.2f} to {train_position[:, 1].max():.2f} rad")

# Fill forces
train_forces[:, 0] = df['force'].values         # Force applied to cart
train_forces[:, 1] = 0                          # No direct force on angle

# Compute velocity using numerical gradient
train_velocity = np.zeros((m_time, n_coordinates))
for i in range(n_coordinates):
    train_velocity[:, i] = np.gradient(train_position[:, i], train_time)

# # Apply low-pass Butterworth filter to velocity
# dt_mean = np.mean(np.diff(train_time))
# fs = 1.0 / dt_mean  # Sampling frequency
# cutoff_vel = 10.0  # Cutoff frequency in Hz for velocity
# order = 4  # Filter order

# nyquist = fs / 2.0
# normal_cutoff_vel = cutoff_vel / nyquist
# b_vel, a_vel = signal.butter(order, normal_cutoff_vel, btype='low', analog=False)

# train_velocity_filtered = np.zeros_like(train_velocity)
# for i in range(n_coordinates):
#     train_velocity_filtered[:, i] = signal.filtfilt(b_vel, a_vel, train_velocity[:, i])

# train_velocity = train_velocity_filtered
# print(f"Velocity filtered with {order}th order Butterworth filter (cutoff: {cutoff_vel} Hz)")

# Compute acceleration from filtered velocity
train_acceleration = np.zeros((m_time, n_coordinates))
for i in range(n_coordinates):
    train_acceleration[:, i] = np.gradient(train_velocity[:, i], train_time)

# # Apply low-pass Butterworth filter to acceleration
# cutoff_acc = 5.0  # Cutoff frequency in Hz for acceleration
# normal_cutoff_acc = cutoff_acc / nyquist
# b_acc, a_acc = signal.butter(order, normal_cutoff_acc, btype='low', analog=False)

# train_acceleration_filtered = np.zeros_like(train_acceleration)
# for i in range(n_coordinates):
#     train_acceleration_filtered[:, i] = signal.filtfilt(b_acc, a_acc, train_acceleration[:, i])

# train_acceleration = train_acceleration_filtered
# print(f"Acceleration filtered with {order}th order Butterworth filter (cutoff: {cutoff_acc} Hz)")

print(f"Data shape:")
print(f"  position: {train_position.shape}")
print(f"  velocity: {train_velocity.shape}")
print(f"  acceleration: {train_acceleration.shape}")
print(f"  forces: {train_forces.shape}")
print(f"  time points: {m_time}")
print(f"  coordinates: {n_coordinates}")

## Apply mujoco transformation
train_position, train_velocity, train_acceleration = mujoco_transform(
    train_position, train_velocity, train_acceleration
)

## sampling to data ratio
catalog_size = catalog_repartition.catalog_length


# Sample uniformly n samples from the imported arrays
n_samples = int(catalog_size * data_ratio)
total_samples = train_position.shape[0]



if n_samples < total_samples:

    # Evenly spaced sampling (deterministic, uniform distribution)
    sample_indices = np.linspace(truncate, total_samples - 1 - truncate, n_samples, dtype=int)
    
    # Apply sampling to all arrays
    t_position = train_position[sample_indices]
    t_velocity = train_velocity[sample_indices]
    t_acceleration = train_acceleration[sample_indices]
    t_forces = train_forces[sample_indices]
    t_train_time = train_time[sample_indices]
    
    print(f"Sampled {n_samples} points uniformly from {total_samples} total samples")
else:
    print(f"Using all {total_samples} samples (requested {n_samples})")

pre_knowledge_mask = np.zeros((catalog_repartition.catalog_length,))


pre_knowledge_indices = np.array([0,1]) + catalog_repartition.starting_index_by_type("ExternalForces")
pre_knowledge_mask[pre_knowledge_indices] =1.0
print(f"Pre-knowledge indices: {pre_knowledge_indices} (forces terms)")

#pre_knowledge_indices = kinetic_energy_indices + catalog_repartition.starting_index_by_type("Lagrange")
#pre_knowledge_mask[pre_knowledge_indices] = 1.0
#print(f"Pre-knowledge indices: {pre_knowledge_indices} (θ̇1² and θ̇2² terms)")

print(f"Catalog size: {catalog_size}")
print(f"pre_knowledge_indices: {pre_knowledge_indices}")

print(sample_indices)

start_time = time.perf_counter()

solution, exp_matrix = xlsindy.simulation.regression_mixed(
    theta_values=t_position,
    velocity_values=t_velocity,
    acceleration_values=t_acceleration,
    time_symbol=time_sym,
    symbol_matrix=symbols_matrix,
    catalog_repartition=catalog_repartition,
    external_force=t_forces,
    pre_knowledge_mask=pre_knowledge_mask,
    regression_function=lasso_regression_tuned,
    l1_lambda=1e-8,
    deg_tol=70,
    weight_distribution_threshold=0.1
)

regression_time = time.perf_counter() - start_time

print(f"Regression completed in {regression_time:.2f} seconds")


#threshold = 1e-2  # Adjust threshold value as needed
#solution = np.where(np.abs(solution) < threshold, 0, solution)

##--------------------------------

final_equation=[]
catalog_label = catalog_repartition.label()
print("Identified model terms:")
for idx, coeff in enumerate(solution.flatten()):
    if coeff != 0:
        print(f"  Coefficient: {coeff:.6f} | Term: {catalog_label[idx]}")
        final_equation.append((coeff, catalog_label[idx]))

##--------------------------------
for term in final_equation:
    print(f"{term[0]:+.6f} * {term[1]}")

# Save solution and catalog information
import pickle

save_data = {
    'solution': solution,
    'catalog_labels': catalog_label,
    'final_equation': final_equation,
    'lagrange_catalog': lagrange_catalog,
    'friction_catalog': friction_catalog,
    'num_coordinates': num_coordinates,
    'regression_time': regression_time,
    'data_ratio': data_ratio,
    'catalog_size': catalog_size,
    'n_samples': n_samples
}

save_filename = 'regression_solution.pkl'
with open(save_filename, 'wb') as f:
    pickle.dump(save_data, f)

print(f"\n✓ Solution and catalog saved to: {save_filename}")

# Also save as numpy array for easy loading
np.save('regression_solution_array.npy', solution)
print(f"✓ Solution array saved to: regression_solution_array.npy")

model_acceleration_func, valid_model = (
    xlsindy.dynamics_modeling.generate_acceleration_function(
        solution, 
        catalog_repartition,
        symbols_matrix,
        time_sym,
        lambdify_module="numpy",
    )
)

if valid_model:
    print("✓ Regression model generated successfully")
    
    # Create force interpolation function factory
    def create_force_function(t_start):
        """Returns a force function that interpolates experimental forces starting at t_start
        
        Args:
            t_start: The time in experimental data where the prediction starts
            
        Returns:
            A function that takes time t (relative to prediction start) and returns
            interpolated forces from experimental data at time (t + t_start)
        """
        # Create interpolators for each force component
        # Use bounds_error=False and fill_value=0 to return 0 outside data range
        force_interp_cart = interp1d(train_time, train_forces[:, 0], 
                                      kind='linear', bounds_error=False, fill_value=0.0)
        force_interp_pole = interp1d(train_time, train_forces[:, 1], 
                                      kind='linear', bounds_error=False, fill_value=0.0)
        
        def force_function(t):
            """Returns interpolated forces at time t (relative to prediction start)"""
            # Translate time: t=0 in prediction corresponds to t_start in experimental data
            actual_time = t + t_start
            
            if np.isscalar(actual_time):
                return np.array([force_interp_cart(actual_time), force_interp_pole(actual_time)])
            else:
                return np.column_stack([force_interp_cart(actual_time), force_interp_pole(actual_time)])
        
        return force_function
    
    # Load experimental data for comparison
    print("\n" + "="*60)
    print("Creating interactive visualization...")
    print("="*60)
    
    # Use full processed dataset for visualization (already unwrapped and filtered)
    exp_time = train_time
    exp_theta1 = train_position[:, 0]  # Cart position
    exp_theta2 = train_position[:, 1]  # Unwrapped pole angle
    exp_omega1 = train_velocity[:, 0]  # Cart velocity (filtered)
    exp_omega2 = train_velocity[:, 1]  # Pole angular velocity (filtered)
    exp_alpha1 = train_acceleration[:, 0]  # Cart acceleration (filtered)
    exp_alpha2 = train_acceleration[:, 1]  # Pole angular acceleration (filtered)
    
    # Create interactive plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Interactive Regression Model Forecast - Click on any plot to generate prediction', 
                 fontsize=14, fontweight='bold')
    
    # Store plot objects
    predicted_lines = []
    click_markers = []
    
    # Plot experimental data
    axes[0, 0].plot(exp_time, exp_theta1, 'b-', alpha=0.7, linewidth=1.5, label='Experimental')
    axes[0, 0].set_ylabel('Theta1 (rad)', fontweight='bold')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_title('Link 1 Angle')
    axes[0, 0].grid(True, alpha=0.3)
    pred_line1, = axes[0, 0].plot([], [], 'r--', linewidth=2, alpha=0.8, label='Predicted')
    click_marker1 = axes[0, 0].plot([], [], 'go', markersize=10, label='Initial Condition')[0]
    predicted_lines.append(pred_line1)
    click_markers.append(click_marker1)
    axes[0, 0].legend()
    
    axes[0, 1].plot(exp_time, exp_theta2, 'g-', alpha=0.7, linewidth=1.5, label='Experimental')
    axes[0, 1].set_ylabel('Theta2 (rad)', fontweight='bold')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_title('Link 2 Angle')
    axes[0, 1].grid(True, alpha=0.3)
    pred_line2, = axes[0, 1].plot([], [], 'r--', linewidth=2, alpha=0.8, label='Predicted')
    click_marker2 = axes[0, 1].plot([], [], 'go', markersize=10, label='Initial Condition')[0]
    predicted_lines.append(pred_line2)
    click_markers.append(click_marker2)
    axes[0, 1].legend()
    
    axes[1, 0].plot(exp_time, exp_omega1, 'b-', alpha=0.7, linewidth=1.5, label='Experimental')
    axes[1, 0].set_ylabel('Omega1 (rad/s)', fontweight='bold')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_title('Link 1 Angular Velocity')
    axes[1, 0].grid(True, alpha=0.3)
    pred_line3, = axes[1, 0].plot([], [], 'r--', linewidth=2, alpha=0.8, label='Predicted')
    click_marker3 = axes[1, 0].plot([], [], 'go', markersize=10, label='Initial Condition')[0]
    predicted_lines.append(pred_line3)
    click_markers.append(click_marker3)
    axes[1, 0].legend()
    
    axes[1, 1].plot(exp_time, exp_omega2, 'g-', alpha=0.7, linewidth=1.5, label='Experimental')
    axes[1, 1].set_ylabel('Omega2 (rad/s)', fontweight='bold')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_title('Link 2 Angular Velocity')
    axes[1, 1].grid(True, alpha=0.3)
    pred_line4, = axes[1, 1].plot([], [], 'r--', linewidth=2, alpha=0.8, label='Predicted')
    click_marker4 = axes[1, 1].plot([], [], 'go', markersize=10, label='Initial Condition')[0]
    predicted_lines.append(pred_line4)
    click_markers.append(click_marker4)
    axes[1, 1].legend()
    
    # Info text
    info_text = fig.text(0.5, 0.01, 'Click anywhere on the plots to select initial condition and generate 5s prediction', 
                         ha='center', fontsize=11, style='italic',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def onclick(event):
        """Handle click event to generate trajectory"""
        if event.inaxes is None:
            return
        
        clicked_time = event.xdata
        if clicked_time is None:
            return
        
        # Find closest time point in data to clicked position
        clicked_idx = np.argmin(np.abs(exp_time - clicked_time))
        
        # Find nearest point where both accelerations are nearly null (< 10)
        # Calculate total acceleration magnitude for each point
        accel_magnitude = np.sqrt(exp_alpha1**2 + exp_alpha2**2)
        
        # Find indices where both accelerations are small
        null_accel_mask = (np.abs(exp_alpha1) < 10) & (np.abs(exp_alpha2) < 10)
        
        if np.any(null_accel_mask):
            # Find valid indices
            valid_indices = np.where(null_accel_mask)[0]
            
            # Find the closest valid index to the clicked point
            distances = np.abs(valid_indices - clicked_idx)
            nearest_valid = valid_indices[np.argmin(distances)]
            idx = nearest_valid
            
            print(f"\n{'='*60}")
            print(f"Clicked at t={exp_time[clicked_idx]:.3f}s")
            print(f"Offsetting to nearest low-acceleration point: t={exp_time[idx]:.3f}s")
            print(f"  α1={exp_alpha1[idx]:.2f} rad/s², α2={exp_alpha2[idx]:.2f} rad/s²")
        else:
            # If no points with nearly null acceleration, use clicked point
            idx = clicked_idx
            print(f"\n{'='*60}")
            print(f"No low-acceleration points found, using clicked point")
            print(f"  α1={exp_alpha1[idx]:.2f} rad/s², α2={exp_alpha2[idx]:.2f} rad/s²")
        
        t0 = exp_time[idx]
        
        # Get initial conditions
        theta_init = np.array([exp_theta1[idx], exp_theta2[idx]])
        omega_init = np.array([exp_omega1[idx], exp_omega2[idx]])
        
        print(f"Starting prediction from t={t0:.3f}s")
        print(f"  Initial angles: θ1={np.degrees(theta_init[0]):.1f}°, θ2={np.degrees(theta_init[1]):.1f}°")
        print(f"  Initial velocities: ω1={np.degrees(omega_init[0]):.1f}°/s, ω2={np.degrees(omega_init[1]):.1f}°/s")
        
        # Generate 5 second forecast
        prediction_duration = 2.0
        
        # Create force function with experimental forces starting at t0
        force_func = create_force_function(t0)
        
        # Generate trajectory using regression model
        model_dynamics_system = xlsindy.dynamics_modeling.dynamics_function(
            model_acceleration_func, 
            force_func
        )
        
        initial_conditions = np.column_stack([theta_init, omega_init])
        
        try:
            pred_time_array, phase_values = xlsindy.dynamics_modeling.run_rk45_integration(
                model_dynamics_system, 
                initial_conditions,  # Shape: (2, 2) [positions, velocities]
                prediction_duration, 
                max_step=0.005
            )
        except Exception as e:
            info_text.set_text(f'Integration failed: {str(e)}')
            fig.canvas.draw_idle()
            print(f"  Integration error: {e}")
            return
        
        pred_theta = phase_values[:, ::2]  # Even columns are positions
        pred_omega = phase_values[:, 1::2]  # Odd columns are velocities
        
        # Shift time to match actual time axis
        pred_time_shifted = pred_time_array + t0
        
        # Update prediction lines
        predicted_lines[0].set_data(pred_time_shifted, pred_theta[:, 0])
        predicted_lines[1].set_data(pred_time_shifted, pred_theta[:, 1])
        predicted_lines[2].set_data(pred_time_shifted, pred_omega[:, 0])
        predicted_lines[3].set_data(pred_time_shifted, pred_omega[:, 1])
        
        # Update click markers
        click_markers[0].set_data([t0], [theta_init[0]])
        click_markers[1].set_data([t0], [theta_init[1]])
        click_markers[2].set_data([t0], [omega_init[0]])
        click_markers[3].set_data([t0], [omega_init[1]])
        
        # Calculate error at end of prediction if within experimental range
        if pred_time_shifted[-1] <= exp_time[-1]:
            end_idx = np.argmin(np.abs(exp_time - pred_time_shifted[-1]))
            theta1_error = np.degrees(np.abs(pred_theta[-1, 0] - exp_theta1[end_idx]))
            theta2_error = np.degrees(np.abs(pred_theta[-1, 1] - exp_theta2[end_idx]))
            
            info_text.set_text(
                f'Prediction from t={t0:.3f}s to t={pred_time_shifted[-1]:.3f}s (5s forecast) | '
                f'End errors: θ1={theta1_error:.1f}°, θ2={theta2_error:.1f}°'
            )
            print(f"  Prediction duration: 5.0s")
            print(f"  End errors: θ1={theta1_error:.1f}°, θ2={theta2_error:.1f}°")
        else:
            info_text.set_text(f'Prediction from t={t0:.3f}s (5s forecast, extends beyond data)')
            print(f"  5s prediction extends beyond experimental data range")
        
        print(f"{'='*60}")
        
        fig.canvas.draw_idle()
    
    # Connect click event
    fig.canvas.mpl_connect('button_press_event', onclick)
    
    plt.tight_layout()
    
    print("\n" + "="*60)
    print("INSTRUCTIONS:")
    print("  - Click anywhere on the plots to select an initial condition")
    print("  - The regression model will predict 5 seconds forward from that point")
    print("  - Red dashed lines show the model prediction")
    print("  - Green dots mark the selected initial condition")
    print("  - Compare predicted vs experimental trajectories")
    print("="*60)
    
    plt.show()
    
else:
    print("Model validation failed - cannot generate forecasts")
    
    # Show training data only
    fig, axes = plt.subplots(4, num_coordinates, figsize=(12, 16))
    fig.suptitle('Training Data (Regression Failed)', fontsize=16)
    
    data_labels = ['Position (qpos)', 'Velocity (qvel)', 'Acceleration (qacc)', 'Forces']
    train_data = [train_position, train_velocity, train_acceleration, train_forces]
    
    for row_idx, (label, train_arr) in enumerate(zip(data_labels, train_data)):
        for col_idx, module in enumerate(modules_name):
            ax = axes[row_idx, col_idx]
            
            # Plot training data
            ax.plot(train_time, train_arr[:, col_idx], 'b-', label='Training', alpha=0.7, linewidth=1.5)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(label)
            ax.set_title(f'{module} q_{col_idx} - {label}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pendulum_training_data.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'pendulum_training_data.png'")
    plt.show()
