#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目2：打靶法与scipy.solve_bvp求解边值问题 - 学生代码模板

本项目要求实现打靶法和scipy.solve_bvp两种方法来求解二阶线性常微分方程边值问题：
u''(x) = -π(u(x)+1)/4
边界条件：u(0) = 1, u(1) = 1

学生姓名：[赖株涛]
学号：[20231050070]
完成日期：[2025/6/4]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp, solve_bvp
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')


def ode_system_shooting(t, y):
    """
    Define the ODE system for shooting method.

    Convert the second-order ODE u'' = -π(u+1)/4 into a first-order system:
    y1 = u, y2 = u'
    y1' = y2
    y2' = -π(y1+1)/4

    Args:
        t (float): Independent variable (time/position)
        y (array): State vector [y1, y2] where y1=u, y2=u'

    Returns:
        list: Derivatives [y1', y2']
    """
    return [y[1], -np.pi*(y[0]+1)/4]


def boundary_conditions_scipy(ya, yb):
    """
    Define boundary conditions for scipy.solve_bvp.

    Boundary conditions: u(0) = 1, u(1) = 1
    ya[0] should equal 1, yb[0] should equal 1

    Args:
        ya (array): Values at left boundary [u(0), u'(0)]
        yb (array): Values at right boundary [u(1), u'(1)]

    Returns:
        array: Boundary condition residuals
    """
    return np.array([ya[0] - 1, yb[0] - 1])


def ode_system_scipy(x, y):
    """
    Define the ODE system for scipy.solve_bvp.

    Note: scipy.solve_bvp uses (x, y) parameter order, different from odeint

    Args:
        x (float): Independent variable
        y (array): State vector [y1, y2]

    Returns:
        array: Derivatives as column vector
    """
    return np.vstack((y[1], -np.pi*(y[0]+1)/4))


def solve_bvp_shooting_method(x_span, boundary_conditions, n_points=100,
                              max_iterations=10, tolerance=1e-6):
    """
    Solve boundary value problem using shooting method.

    Algorithm:
    1. Guess initial slope m1
    2. Solve IVP with initial conditions [u(0), m1]
    3. Check if u(1) matches boundary condition
    4. If not, adjust slope using secant method and repeat

    Args:
        x_span (tuple): Domain (x_start, x_end)
        boundary_conditions (tuple): (u_left, u_right)
        n_points (int): Number of discretization points
        max_iterations (int): Maximum iterations for shooting
        tolerance (float): Convergence tolerance

    Returns:
        tuple: (x_array, y_array) solution arrays
    """
    # Validate input parameters
    if len(x_span) != 2 or x_span[0] >= x_span[1]:
        raise ValueError(
            "x_span must be a tuple of two numbers "
            "where x_span[0] < x_span[1]")
    if len(boundary_conditions) != 2:
        raise ValueError("boundary_conditions must be a tuple of two numbers")

    # Extract boundary conditions and setup domain
    u_left, u_right = boundary_conditions
    x_array = np.linspace(x_span[0], x_span[1], n_points)

    # Initial guesses for slope
    m1 = 0.0
    m2 = 1.0

    for _ in range(max_iterations):
        # Solve IVP with m1
        sol1 = solve_ivp(ode_system_shooting, x_span,
                         [u_left, m1], t_eval=x_array)
        u1 = sol1.y[0, -1]

        # Solve IVP with m2
        sol2 = solve_ivp(ode_system_shooting, x_span,
                         [u_left, m2], t_eval=x_array)
        u2 = sol2.y[0, -1]

        # Check if u(1) matches boundary condition
        if np.abs(u2 - u_right) < tolerance:
            return x_array, sol2.y[0]

        # Secant method to adjust slope
        m_new = m2 - (u2 - u_right) * (m2 - m1) / (u2 - u1)
        m1 = m2
        m2 = m_new

    raise ValueError("Shooting method did not converge within max_iterations")


def solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points=50):
    """
    Solve boundary value problem using scipy.solve_bvp.

    Args:
        x_span (tuple): Domain (x_start, x_end)
        boundary_conditions (tuple): (u_left, u_right)
        n_points (int): Number of initial mesh points

    Returns:
        tuple: (x_array, y_array) solution arrays
    """
    # Setup initial mesh and guess
    x = np.linspace(x_span[0], x_span[1], n_points)
    y_guess = np.zeros((2, x.size))

    # Call scipy.solve_bvp
    sol = solve_bvp(ode_system_scipy, boundary_conditions_scipy, x, y_guess)

    # Extract and return solution
    if sol.success:
        x_array = sol.x
        y_array = sol.y[0]
        return x_array, y_array
    else:
        raise ValueError("scipy.solve_bvp did not converge")


def compare_methods_and_plot(x_span=(0, 1), boundary_conditions=(1, 1),
                             n_points=100):
    """
    Compare shooting method and scipy.solve_bvp, generate comparison plot.

    Args:
        x_span (tuple): Domain for the problem
        boundary_conditions (tuple): Boundary values (left, right)
        n_points (int): Number of points for plotting

    Returns:
        dict: Dictionary containing solutions and analysis
    """
    # Solve using both methods
    x_shooting, y_shooting = solve_bvp_shooting_method(
        x_span, boundary_conditions, n_points)
    x_scipy, y_scipy = solve_bvp_scipy_wrapper(
        x_span, boundary_conditions, n_points)

    # Create comparison plot with English labels
    plt.figure(figsize=(10, 6))
    plt.plot(x_shooting, y_shooting, label='Shooting Method')
    plt.plot(x_scipy, y_scipy, label='scipy.solve_bvp')
    plt.title('Comparison of Shooting Method and scipy.solve_bvp')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate and display differences
    y_common = np.interp(x_shooting, x_scipy, y_scipy)
    difference = np.abs(y_shooting - y_common)
    max_difference = np.max(difference)
    print(f"Maximum difference between the two methods: {max_difference}")

    # Return analysis results
    return {
        'shooting': (x_shooting, y_shooting),
        'scipy': (x_scipy, y_scipy),
        'max_difference': max_difference
    }


# Test functions for development and debugging
def test_ode_system():
    """
    Test the ODE system implementation.
    """
    print("Testing ODE system...")
    try:
        # Test point
        t_test = 0.5
        y_test = np.array([1.0, 0.5])

        # Test shooting method ODE system
        dydt = ode_system_shooting(t_test, y_test)
        print(f"ODE system (shooting): dydt = {dydt}")

        # Test scipy ODE system
        dydt_scipy = ode_system_scipy(t_test, y_test)
        print(f"ODE system (scipy): dydt = {dydt_scipy}")

    except NotImplementedError:
        print("ODE system functions not yet implemented.")


def test_boundary_conditions():
    """
    Test the boundary conditions implementation.
    """
    print("Testing boundary conditions...")
    try:
        ya = np.array([1.0, 0.5])  # Left boundary
        yb = np.array([1.0, -0.3])  # Right boundary

        bc_residual = boundary_conditions_scipy(ya, yb)
        print(f"Boundary condition residuals: {bc_residual}")

    except NotImplementedError:
        print("Boundary conditions function not yet implemented.")


if __name__ == "__main__":
    print("项目2：打靶法与scipy.solve_bvp求解边值问题")
    print("=" * 50)

    # Run basic tests
    test_ode_system()
    test_boundary_conditions()

    # Try to run comparison (will fail until functions are implemented)
    try:
        print("\nTesting method comparison...")
        results = compare_methods_and_plot()
        print("Method comparison completed successfully!")
    except NotImplementedError as e:
        print(f"Method comparison not yet implemented: {e}")

    print("\n请实现所有标记为 TODO 的函数以完成项目。")
