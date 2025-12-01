import cvxpy as cp
import numpy as np

# 1. Define System Matrices
A1 = np.array([[-0.5, 0.3, 0.4],
               [ 1.0, 0.0, 0.0],
               [ 0.0, 1.0, 0.0]])

A2 = np.array([[-0.7, 0.1, -0.2],
               [ 1.0, 0.0,  0.0],
               [ 0.0, 1.0,  0.0]])

A3 = np.array([[ 0.6, -0.7, 0.2],
               [ 1.0,  0.0, 0.0],
               [ 0.0,  1.0, 0.0]])

B = np.array([[1.0],
              [0.0],
              [0.0]])

beta = 1.0
n = 3  # State dimension
m = 1  # Input dimension

# 2. Define Optimization Variables
# Q corresponds to P^-1 (Must be Symmetric and Positive Definite)
Q = cp.Variable((n, n), symmetric=True)

# Y corresponds to K * P^-1 (Rectangular matrix)
Y = cp.Variable((m, n))

# 3. Define Constraints
constraints = []

# Constraint A: Q must be Positive Definite (Q > 0)
# We enforce Q >= epsilon*I to ensure it is strictly positive
constraints.append(Q >> np.eye(n) * 1e-5)

# Constraint B: Stability LMIs for all subsystems
# Inequality: A*Q + Q*A.T + B*Y + Y.T*B.T + beta*Q <= 0
system_matrices = [A1, A2, A3]

for A in system_matrices:
    lmi = A @ Q + Q @ A.T + B @ Y + Y.T @ B.T + beta * Q
    constraints.append(lmi << 0)  # Negative Semi-Definite

# 4. Solve the Problem
# We just need to find ANY valid solution (Feasibility problem)
prob = cp.Problem(cp.Minimize(0), constraints)

try:
    prob.solve()
except cp.SolverError:
    print("Solver failed!")

# 5. Recover K and P
if prob.status == 'optimal':
    Q_val = Q.value
    Y_val = Y.value
    
    # Recover P (inverse of Q)
    P_val = np.linalg.inv(Q_val)
    
    # Recover K (Y * P)
    K_val = Y_val @ P_val
    
    print("-" * 40)
    print("Optimization Status: Optimal")
    print("-" * 40)
    print("Calculated Feedback Gain K:")
    print(K_val)
    print("\nLyapunov Matrix P:")
    print(P_val)
    
    print("\n--- Verification (Real part should be <= -0.5) ---")
    for i, A in enumerate(system_matrices):
        eigvals = np.linalg.eigvals(A + B @ K_val)
        max_real = np.max(np.real(eigvals))
        print(f"System {i+1} Max Real Part: {max_real:.4f}")
        
else:
    print("Problem is infeasible or unbounded.")