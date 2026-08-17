import time
import json
import numpy as np
import sympy as sp
from scipy.spatial.transform import Rotation as R_scipy

from pnp_poly_solvers import *

class PnPSolver:
    def __init__(self, camera_matrix, p3d=None, weights=None):
        self.K = camera_matrix
        self.invK = np.linalg.inv(camera_matrix)
        self.q_vars = sp.symbols('q0 q1 q2 q3')
        self.valid_mask = None
        self.p3d_valid = None
        self.weights_valid = None
        self.weights = None
        self.set_p3d(p3d, weights)


    # converts quaternion to rotation matrix
    @staticmethod
    def quat_to_R(q):
        n = np.linalg.norm(q)
        if n < 1e-8:
            return np.eye(3)
        # scipy uses [x, y, z, w]
        return R_scipy.from_quat([q[1]/n, q[2]/n, q[3]/n, q[0]/n]).as_matrix()

    # calculates the rays from the camera center to the 2D points in the image captured by the camera
    def _get_rays(self, p2d):
        rays = [self.invK @ np.array([uv[0], uv[1], 1]) for uv in p2d]
        rays = [r / np.linalg.norm(r) for r in rays]
        return rays

    # sets the 3D points with weights to support coreset
    def set_p3d(self, p3d, weights):
        if p3d is None: return
        if weights is None: weights = np.ones(len(p3d))

        self.p3d = p3d
        self.weights = weights
        self.valid_mask = weights > 1e-9
        self.p3d_valid = p3d[self.valid_mask]
        self.weights_valid = weights[self.valid_mask]

    # builds the Q matrix used in the polynomial formulation of the PnP problem
    def _build_Q(self, p2d):
        if self.p3d is None: return
        rays = self._get_rays(p2d)
        n_valid = len(self.p3d_valid)

        p3d_mean = np.average(self.p3d_valid, axis=0, weights=self.weights_valid)
        p3d_centered = self.p3d_valid - p3d_mean
        dist_avg = np.mean(np.linalg.norm(p3d_centered, axis=1))
        scale_3d = np.sqrt(3) / dist_avg
        p3d_norm = p3d_centered * scale_3d

        W = np.zeros((3,3))
        for i in range(n_valid): W += self.weights_valid[i] * (np.eye(3) - np.outer(rays[i], rays[i]))
        Winv = np.linalg.inv(W)

        sum_VK = np.zeros((3, 9))
        for i in range(n_valid):
            V_i = np.eye(3) - np.outer(rays[i], rays[i])
            p = p3d_norm[i]
            K_i = np.zeros((3, 9)); K_i[0, 0:3] = p; K_i[1, 3:6] = p; K_i[2, 6:9] = p
            sum_VK += self.weights_valid[i] * (V_i @ K_i)

        T_mat = - Winv @ sum_VK

        Q = np.zeros((9, 9))
        for i in range(n_valid):
            V_i = np.eye(3) - np.outer(rays[i], rays[i])
            p = p3d_norm[i]
            K_i = np.zeros((3, 9)); K_i[0, 0:3] = p; K_i[1, 3:6] = p; K_i[2, 6:9] = p
            A_i = V_i @ (K_i + T_mat)
            Q += self.weights_valid[i] * (A_i.T @ A_i)
        return Q

    # builds the equations for the polynomial formulation of the PnP problem
    def _build_equations(self, p2d):
        p2d_valid = p2d[self.valid_mask]
        Q = self._build_Q(p2d_valid)

        q0,q1,q2,q3 = sp.symbols('q0 q1 q2 q3')
        R11 = q0**2 + q1**2 - q2**2 - q3**2; R12 = 2*(q1*q2 - q0*q3); R13 = 2*(q1*q3 + q0*q2)
        R21 = 2*(q1*q2 + q0*q3); R22 = q0**2 - q1**2 + q2**2 - q3**2; R23 = 2*(q2*q3 - q0*q1)
        R31 = 2*(q1*q3 - q0*q2); R32 = 2*(q2*q3 + q0*q1); R33 = q0**2 - q1**2 - q2**2 + q3**2
        r_sym = sp.Matrix([R11, R12, R13, R21, R22, R23, R31, R32, R33])

        # def to_rat(x): return sp.Integer(int(round(float(x) * 1)))
        def to_rat(x): return sp.nsimplify(round(float(x), 2), rational=True)
        # def to_rat(x): return sp.nsimplify(x, tolerance=1e-8, rational=True)
        Q_rat = sp.Matrix(9, 9, [to_rat(x) for x in Q.flatten()])
        Sq = (r_sym.T * Q_rat * r_sym)[0]

        grad = [sp.diff(Sq, x) for x in [q0,q1,q2,q3]]
        eqs = [
            sp.expand(q0*grad[1] - q1*grad[0]),
            sp.expand(q1*grad[2] - q2*grad[1]),
            sp.expand(q2*grad[3] - q3*grad[2]),
            q0**2 + q1**2 + q2**2 + q3**2 - 1
        ]
        return eqs

    # extracts the quaternion that minimizes the cost
    def _extract_best_solution(self, quaternions, p2d):
        best_err = np.inf
        best_R, best_t = None, None
        rays_full = self._get_rays(p2d)
        rays = self._get_rays(p2d[self.valid_mask])

        W_sum = np.zeros((3, 3))
        for i in range(len(rays)):
            v = rays[i].reshape(3, 1)
            W_i = self.weights_valid[i] * (np.eye(3) - v @ v.T)
            W_sum += W_i
        Winv_full = np.linalg.inv(W_sum)

        for q in quaternions:
            q_norm = np.linalg.norm(q)
            q = q / q_norm
            R_est = self.quat_to_R(q)

            rhs = np.zeros((3, 1))
            for i in range(len(self.p3d_valid)):
                v = rays[i].reshape(3, 1)
                W_i = self.weights_valid[i] * (np.eye(3) - v @ v.T)
                # Apply rotation to the 3D point
                rotated_p = (R_est @ self.p3d_valid[i].reshape(3, 1))
                rhs += W_i @ rotated_p

            # The optimal translation
            t_est = -(Winv_full @ rhs).flatten()

            current_err = self.calc_err(R_est, t_est, rays_full)

            if current_err < best_err:
                best_err = current_err
                best_R, best_t = R_est, t_est

        return best_R, best_t, best_err

    # calculates the cost for a given R and t
    def calc_err(self, R, t, rays):
        err = 0
        for i in range(len(self.p3d)):
            Pc = R @ self.p3d[i] + t
            v = rays[i]
            proj = v * np.dot(v, Pc)
            err += self.weights[i] * np.sum((Pc - proj)**2)
        return err

    # solves the PnP problem
    def solve(self, p2d, p3d=None, weights=None):
        if self.p3d is None:
            if p3d is None:
                print('Error: no 3D points given')
                return
            self.set_p3d(p3d, weights)

        eqs = self._build_equations(p2d)

        # print(f" Running polynomial solver...")
        t_start_sympy = time.perf_counter()
        # quaternions = solve_with_sympy_fast(eqs)
        # quaternions = solve_with_sympy_poly_optimized(eqs)
        quaternions = solve_with_msolve(eqs) # the poly solver that we choose
        t_end_sympy = time.perf_counter()
        sympy_duration = t_end_sympy - t_start_sympy
        # print(f"Polynomial solver finished in {sympy_duration:.4f}s")

        best_R, best_t, best_err = self._extract_best_solution(quaternions, p2d)
        return best_R, best_t, best_err

    def solve2(self, p2d):
        omega = self._build_Q(p2d)
        _, U = np.linalg.eigh(omega)

        best_r = None
        min_err = float('inf')

        for i in range(3):
            r0 = U[:, i] * np.sqrt(3)

            r_refined = self._run_sqp(r0)


            err = r_refined.T @ omega @ r_refined
            if err < min_err:
                min_err = err
                best_r = r_refined

        R_est = best_r.reshape(3, 3)
        rays = self._get_rays(p2d)
        sum_Wi = np.zeros((3, 3))
        sum_Wi_R_p = np.zeros(3)

        for i in range(len(self.p3d)):
            v = rays[i]

            Vi = np.eye(3) - np.outer(v, v)
            Wi = self.weights[i] * Vi

            sum_Wi += Wi

            sum_Wi_R_p += Wi @ (R_est @ self.p3d[i])

        t_est = -np.linalg.inv(sum_Wi) @ sum_Wi_R_p
        err = self.calc_err(R_est, t_est, rays)
        return R_est, t_est, err

    def _run_sqp(self, r):


        for _ in range(15):

            U, _, Vt = np.linalg.svd(r.reshape(3, 3))
            r_hat = (U @ Vt).flatten()


            if np.linalg.norm(r - r_hat) < 1e-10:
                break


            r = r_hat
        return r

    def solve3(self, p2d):
        omega = self._build_Q(p2d)
        M = self._map_to_quaternion_matrix(omega)
        print(M)

        q0, q1, q2, q3 = sp.symbols('q0 q1 q2 q3', real=True)
        q = sp.Matrix([q0, q1, q2, q3])

        M_sp = sp.Matrix(M).applyfunc(lambda x: sp.nsimplify(x, tolerance=1e-4, rational=True))
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        best_q = eigenvectors[:, 0]

        print(f"eigenvectors: {eigenvectors}")

        cost_func = (q.T * M_sp * q)[0]
        grad = [sp.diff(cost_func, var) for var in [q0, q1, q2, q3]]


        polys = [
            sp.expand(q0 * grad[1] - q1 * grad[0]),
            sp.expand(q1 * grad[2] - q2 * grad[1]),
            sp.expand(q2 * grad[3] - q3 * grad[2]),
            q0**2 + q1**2 + q2**2 + q3**2 - 1
        ]

        print(f"Sending {len(polys)} polynomial equations to sp.solve_poly_system...")


        solutions = sp.solve_poly_system(polys, [q0, q1, q2, q3])
        numeric_solutions = []
        for sol in solutions:


            numeric_sol = [float(val.evalf()) if hasattr(val, 'evalf') else float(val) for val in sol]
            numeric_solutions.append(numeric_sol)

        print(numeric_solutions)
        return self._extract_best_solution(numeric_solutions, p2d)

    def _map_to_quaternion_matrix(self, Q):
        M = np.zeros((4, 4))


        M[0, 0] = Q[0, 0] + Q[4, 4] + Q[8, 8]
        M[1, 1] = Q[0, 0] - Q[4, 4] - Q[8, 8]
        M[2, 2] = -Q[0, 0] + Q[4, 4] - Q[8, 8]
        M[3, 3] = -Q[0, 0] - Q[4, 4] + Q[8, 8]



        M[0, 1] = Q[7, 5] - Q[5, 7]
        M[0, 2] = Q[2, 6] - Q[6, 2]
        M[0, 3] = Q[3, 1] - Q[1, 3]

        M[1, 2] = Q[1, 3] + Q[3, 1]
        M[1, 3] = Q[6, 2] + Q[2, 6]
        M[2, 3] = Q[5, 7] + Q[7, 5]


        M = M + M.T - np.diag(np.diag(M))
        return M

# ==========================================
# Benchmark
# ==========================================

# test functions that run on test_data.json generated by build_test.py

def run_benchmark():
    print(" --- Python PnP Solver Benchmark --- ")
    K_orig = np.array([[647.8, 0, 335.9], [0, 645.9, 226.0], [0, 0, 1.0]])

    try:
        with open('test_data.json', 'r') as f:
            test_units = json.load(f)
    except FileNotFoundError:
        print("Error: 'test_data.json' not found. Please run 'build_test.py' first.")
        return

    for i, unit in enumerate(test_units):
        print(f"\n --- Processing Unit {i+1}/{len(test_units)} ---")
        p3d = np.array(unit['p3d'])
        p2d = np.array(unit['p2d'])
        R_true = np.array(unit['R'])
        t_true = np.array(unit['t'])

        start_time = time.perf_counter()
        pnp_solver = PnPSolver(K_orig, p3d)
        R_res, t_res, err = pnp_solver.solve(p2d)
        end_time = time.perf_counter()
        print(f"Solver took: {end_time - start_time:.6f} seconds")

        print(f"\n --- RESULTS UNIT {i+1} --- ")
        print(f"Estimated R:\n{np.round(R_res, 4)}")
        print(f"True R:\n{np.round(R_true, 4)}")
        print(f"Estimated t: {np.round(t_res, 4)}")
        print(f"True t: {np.round(t_true, 4)}")
        print(f"Cost: {err:.6f}")

def run_fast_benchmark(n_limit=50):
    print(f" --- Python PnP Solver Fast Benchmark (Average over {n_limit} units) --- ")
    K_orig = np.array([[647.8, 0, 335.9], [0, 645.9, 226.0], [0, 0, 1.0]])

    try:
        with open('test_data.json', 'r') as f:
            test_units = json.load(f)
    except FileNotFoundError:
        print("Error: 'test_data.json' not found. Please run 'build_test.py' first.")
        return

    test_units = test_units[:n_limit]
    n_units = len(test_units)
    total_time = 0
    total_cost = 0

    for unit in test_units:
        p3d = np.array(unit['p3d'])
        p2d = np.array(unit['p2d'])

        start_t = time.perf_counter()
        pnp_solver = PnPSolver(K_orig, p3d)
        _, _, err = pnp_solver.solve(p2d)
        total_time += (time.perf_counter() - start_t)
        total_cost += err
        print('unit number: ' + str(test_units.index(unit) + 1))

    print("\n --- FINAL AVERAGES --- ")
    print(f"Average Solver Time: {total_time / n_units:.6f} seconds")
    print(f"Average Solver Cost: {total_cost / n_units:.6f}")

if __name__ == "__main__":
    run_benchmark()