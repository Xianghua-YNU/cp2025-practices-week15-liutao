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
# 忽略警告信息
warnings.filterwarnings('ignore')


def ode_system_shooting(t, y):
    """
    定义打靶法所需的常微分方程系统。

    将二阶常微分方程 u'' = -π(u+1)/4 转换为一阶系统：
    y1 = u, y2 = u'
    y1' = y2
    y2' = -π(y1+1)/4

    参数:
        t (float): 自变量（时间/位置）
        y (array): 状态向量 [y1, y2]，其中 y1=u, y2=u'

    返回:
        list: 导数 [y1', y2']
    """
    return [y[1], -np.pi*(y[0]+1)/4]


def boundary_conditions_scipy(ya, yb):
    """
    定义 scipy.solve_bvp 所需的边界条件。

    边界条件：u(0) = 1, u(1) = 1
    ya[0] 应等于 1，yb[0] 应等于 1

    参数:
        ya (array): 左边界的值 [u(0), u'(0)]
        yb (array): 右边界的值 [u(1), u'(1)]

    返回:
        array: 边界条件残差
    """
    return np.array([ya[0] - 1, yb[0] - 1])


def ode_system_scipy(x, y):
    """
    定义 scipy.solve_bvp 所需的常微分方程系统。

    注意：scipy.solve_bvp 使用 (x, y) 参数顺序，与 odeint 不同

    参数:
        x (float): 自变量
        y (array): 状态向量 [y1, y2]

    返回:
        array: 作为列向量的导数
    """
    return np.vstack((y[1], -np.pi*(y[0]+1)/4))


def solve_bvp_shooting_method(x_span, boundary_conditions, n_points=100,
                              max_iterations=10, tolerance=1e-6):
    """
    使用打靶法求解边值问题。

    算法步骤:
    1. 猜测初始斜率 m1
    2. 使用初始条件 [u(0), m1] 求解初值问题
    3. 检查 u(1) 是否满足边界条件
    4. 如果不满足，使用割线法调整斜率并重复

    参数:
        x_span (tuple): 定义域 (x_start, x_end)
        boundary_conditions (tuple): (u_left, u_right)
        n_points (int): 离散化点数
        max_iterations (int): 打靶法的最大迭代次数
        tolerance (float): 收敛容差

    返回:
        tuple: (x_array, y_array) 解数组
    """
    # 验证输入参数
    if len(x_span) != 2 or x_span[0] >= x_span[1]:
        raise ValueError(
            "x_span 必须是包含两个数的元组，且 x_span[0] < x_span[1]")
    if len(boundary_conditions) != 2:
        raise ValueError("boundary_conditions 必须是包含两个数的元组")

    # 提取边界条件并设置定义域
    u_left, u_right = boundary_conditions
    x_array = np.linspace(x_span[0], x_span[1], n_points)

    # 初始斜率猜测
    m1 = 0.0
    m2 = 1.0

    for _ in range(max_iterations):
        # 使用 m1 求解初值问题
        sol1 = solve_ivp(ode_system_shooting, x_span,
                         [u_left, m1], t_eval=x_array)
        u1 = sol1.y[0, -1]

        # 使用 m2 求解初值问题
        sol2 = solve_ivp(ode_system_shooting, x_span,
                         [u_left, m2], t_eval=x_array)
        u2 = sol2.y[0, -1]

        # 检查 u(1) 是否满足边界条件
        if np.abs(u2 - u_right) < tolerance:
            return x_array, sol2.y[0]

        # 使用割线法调整斜率
        m_new = m2 - (u2 - u_right) * (m2 - m1) / (u2 - u1)
        m1 = m2
        m2 = m_new

    raise ValueError("打靶法在最大迭代次数内未收敛")


def solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points=50):
    """
    使用 scipy.solve_bvp 求解边值问题。

    参数:
        x_span (tuple): 定义域 (x_start, x_end)
        boundary_conditions (tuple): (u_left, u_right)
        n_points (int): 初始网格点数

    返回:
        tuple: (x_array, y_array) 解数组
    """
    # 设置初始网格和猜测值
    x = np.linspace(x_span[0], x_span[1], n_points)
    y_guess = np.zeros((2, x.size))

    # 调用 scipy.solve_bvp
    sol = solve_bvp(ode_system_scipy, boundary_conditions_scipy, x, y_guess)

    # 提取并返回解
    if sol.success:
        x_array = sol.x
        y_array = sol.y[0]
        return x_array, y_array
    else:
        raise ValueError("scipy.solve_bvp 未收敛")


def compare_methods_and_plot(x_span=(0, 1), boundary_conditions=(1, 1),
                             n_points=100):
    """
    比较打靶法和 scipy.solve_bvp，并生成比较图。

    参数:
        x_span (tuple): 问题的定义域
        boundary_conditions (tuple): 边界值 (左, 右)
        n_points (int): 绘图点数

    返回:
        dict: 包含解和分析结果的字典
    """
    # 使用两种方法求解
    x_shooting, y_shooting = solve_bvp_shooting_method(
        x_span, boundary_conditions, n_points)
    x_scipy, y_scipy = solve_bvp_scipy_wrapper(
        x_span, boundary_conditions, n_points)

    # 创建比较图，使用英文标签
    plt.figure(figsize=(10, 6))
    plt.plot(x_shooting, y_shooting, label='Shooting Method')
    plt.plot(x_scipy, y_scipy, label='scipy.solve_bvp')
    plt.title('Comparison of Shooting Method and scipy.solve_bvp')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 计算并显示差异
    y_common = np.interp(x_shooting, x_scipy, y_scipy)
    difference = np.abs(y_shooting - y_common)
    max_difference = np.max(difference)
    print(f"两种方法之间的最大差异: {max_difference}")

    # 返回分析结果
    return {
        'shooting': (x_shooting, y_shooting),
        'scipy': (x_scipy, y_scipy),
        'max_difference': max_difference
    }

# 开发和调试用的测试函数


def test_ode_system():
    """
    测试常微分方程系统的实现。
    """
    print("Testing ODE system...")
    try:
        # 测试点
        t_test = 0.5
        y_test = np.array([1.0, 0.5])

        # 测试打靶法的常微分方程系统
        dydt = ode_system_shooting(t_test, y_test)
        print(f"ODE system (shooting): dydt = {dydt}")

        # 测试 scipy 的常微分方程系统
        dydt_scipy = ode_system_scipy(t_test, y_test)
        print(f"ODE system (scipy): dydt = {dydt_scipy}")

    except NotImplementedError:
        print("ODE system functions not yet implemented.")


def test_boundary_conditions():
    """
    测试边界条件的实现。
    """
    print("Testing boundary conditions...")
    try:
        ya = np.array([1.0, 0.5])  # 左边界
        yb = np.array([1.0, -0.3])  # 右边界

        bc_residual = boundary_conditions_scipy(ya, yb)
        print(f"Boundary condition residuals: {bc_residual}")

    except NotImplementedError:
        print("Boundary conditions function not yet implemented.")


if __name__ == "__main__":
    print("项目2：打靶法与scipy.solve_bvp求解边值问题")
    print("=" * 50)

    # 运行基本测试
    test_ode_system()
    test_boundary_conditions()

    # 尝试运行比较（在函数实现之前会失败）
    try:
        print("\nTesting method comparison...")
        results = compare_methods_and_plot()
        print("Method comparison completed successfully!")
    except NotImplementedError as e:
        print(f"Method comparison not yet implemented: {e}")

    print("\n请实现所有标记为 TODO 的函数以完成项目。")
