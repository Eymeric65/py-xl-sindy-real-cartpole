"""
Some util used in generate data, align data and so one.
Not really something that should go in xlsindy.
"""
import contextlib
import logging 
import numpy as np

from typing import List

import xlsindy
from tqdm import tqdm

import sympy as sp

logger = logging.getLogger(__name__)

def generate_forces_function(
    component_count: int,
    scale_vector: np.ndarray,
    random_seed: List[int],
    time_end: float,
):
        # First try to generate forces function        
    # forces_function = xlsindy.dynamics_modeling.optimized_force_generator(
    #     component_count=num_coordinates,
    #     scale_vector=forces_scale_vector,
    #     time_end=max_time,
    #     period=forces_period,
    #     period_shift=forces_period_shift,
    #     augmentations=10, # base is 40
    #     random_seed=[random_seed,i],
    # )

    forces_function = xlsindy.dynamics_modeling.sinusoidal_force_generator(
        component_count=component_count,
        scale_vector=scale_vector,
        time_end=time_end,
        num_frequencies=5,
        freq_range=(0.01,1.0),
        random_seed=random_seed,
    )

    return forces_function


def generate_theoretical_trajectory(
    num_coordinates: int,
    initial_position: np.ndarray,
    initial_condition_randomness: np.ndarray,
    random_seed: List[int],
    batch_number: int,
    max_time: float,
    solution_vector: np.ndarray,
    solution_catalog: xlsindy.catalog.CatalogRepartition,
    time_symb: sp.Symbol,
    symbols_matrix: np.ndarray,
    forces_scale_vector: np.ndarray,
):
    """
    [INFO] maybe I should but this function inside the main library.
    Generate a theoretical trajectory using theoretical background.

    Args:

    Returns:
    """
    
    simulation_time_g = np.empty((0,1))
    simulation_qpos_g = np.empty((0,num_coordinates))
    simulation_qvel_g = np.empty((0,num_coordinates))
    simulation_qacc_g = np.empty((0,num_coordinates))
    force_vector_g = np.empty((0,num_coordinates))
    batch_starting_times = []

    if len(initial_position)==0:
        initial_position = np.zeros((num_coordinates,2))

    rng = np.random.default_rng(random_seed)

    model_acceleration_func, valid_model = xlsindy.dynamics_modeling.generate_acceleration_function(
    solution_vector,
    solution_catalog,
    symbols_matrix,
    time_symb,
    lambdify_module="numpy"
    )
    
    for i in tqdm(range(batch_number),desc="Generating batches", unit="batch"):

        # Record batch starting time
        batch_start_time = 0.0 if len(simulation_time_g) == 0 else np.max(simulation_time_g)
        batch_starting_times.append(float(batch_start_time))

        # Initial condition
        initial_condition = np.array(initial_position).reshape(num_coordinates,2)

        if len(initial_condition_randomness) == 1:
            initial_condition += rng.normal(
                loc=0, scale=initial_condition_randomness, size=initial_condition.shape
            )
        else:
            initial_condition += rng.normal(
                loc=0, scale=np.reshape(initial_condition_randomness,initial_condition.shape)
            )

        forces_function = generate_forces_function(
            component_count=num_coordinates,
            scale_vector=forces_scale_vector,
            time_end=max_time,
            random_seed=[random_seed,i],
        )

        model_dynamics_system = xlsindy.dynamics_modeling.dynamics_function(model_acceleration_func,forces_function) 
        logger.info("theoretical initialized")
        try:
            simulation_time_m, phase_values = xlsindy.dynamics_modeling.run_rk45_integration(model_dynamics_system, initial_condition, max_time, max_step=0.005)
        except Exception as e:
            logger.error(f"An error occurred on the RK45 integration: {e}")
        logger.info("theoretical simulation done")

        simulation_qpos_m = phase_values[:, ::2]
        simulation_qvel_m = phase_values[:, 1::2]

        simulation_qacc_m = np.gradient(simulation_qvel_m, simulation_time_m, axis=0, edge_order=1)

        force_vector_m = forces_function(simulation_time_m.T).T

        if len(simulation_qvel_g) >0:
            simulation_time_m += np.max(simulation_time_g)

        # Concatenate the data
        simulation_time_g = np.concatenate((simulation_time_g, simulation_time_m.reshape(-1, 1)), axis=0)
        simulation_qpos_g = np.concatenate((simulation_qpos_g, simulation_qpos_m), axis=0)
        simulation_qvel_g = np.concatenate((simulation_qvel_g, simulation_qvel_m), axis=0)
        simulation_qacc_g = np.concatenate((simulation_qacc_g, simulation_qacc_m), axis=0)
        force_vector_g = np.concatenate((force_vector_g, force_vector_m), axis=0)

    return simulation_time_g, simulation_qpos_g, simulation_qvel_g, simulation_qacc_g, force_vector_g, batch_starting_times