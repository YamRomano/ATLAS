import numpy as np
import sympy as sp
from sympy.polys.polytools import groebner
import subprocess
import ast
import re
import sys
import platform


# ====================================
# Msolve Solver
# ====================================

MSOLVE_PIPE = "/mnt/c/Users/user/Desktop/PnP_Project/OptimalPnP-main/msolve_pipe_wsl.py"

# runs msolve in a universal way that works on both Windows and Linux, using WSL if on Windows. Communicates via stdin/stdout pipes.
def run_msolve_pipeline(msolve_input):
    # different command for Windows (using WSL) vs Linux
    is_windows = platform.system() == "Windows"
    cmd = ["python3", MSOLVE_PIPE]
    if is_windows:
        cmd = ["wsl"] + cmd

    # the pipe
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = process.communicate(input=msolve_input)

    if process.returncode != 0:
        print(f"Error: {stderr}", file=sys.stderr)
    return stdout

# solves using the msolve pipeline
def solve_with_msolve(equations):
    polys = [str(sp.expand(sp.together(e).as_numer_denom()[0])).replace("**","^") for e in equations]

    # constructing the msolve input format
    clean_polys = [str(p).replace(" ", "").strip() for p in polys]
    polys_block = ",\n".join(clean_polys)
    vars_line = "q0,q1,q2,q3"
    field_line = "0"
    msolve_input = f"{vars_line}\n{field_line}\n{polys_block}\n"

    out_txt = run_msolve_pipeline(msolve_input)

    # processing the output to extract the RUR parametrization
    start_idx = out_txt.find('[')
    end_idx = out_txt.rfind(']') + 1

    if start_idx == -1:
        print("  [ERROR] Could not find RUR in msolve output!")
        return []

    rur_str = out_txt[start_idx:end_idx]
    safe_str = re.sub(r'(-?\s*\d+\s*/\s*2\^\d+)', r'"\1"', rur_str)
    raw_list = ast.literal_eval(safe_str)

    # data fix for new version of msolve that adds an extra nesting level
    raw_list = raw_list[1][1]
    for i in range(len(raw_list)):
        raw_list[i] = [raw_list[i][j][0] for j in range(len(raw_list[0]))]

    def to_float(item):
        if isinstance(item, list):
            return [to_float(x) for x in item]

        elif isinstance(item, str):
            if '/' in item:
                num_str, denom_str = item.split('/')
                num = float(num_str.replace(" ", ""))
                exp = int(denom_str.split('^')[1].strip())
                return num / (2.0**exp)
            return float(item)

        else:
            return float(item)

    quaternions = to_float(raw_list)

    return quaternions

# ====================================
# SymPy Solvers
# ====================================

def solve_with_sympy(equations):
    q0, q1, q2, q3 = sp.symbols('q0 q1 q2 q3')

    try:
        initial_guess = [1, 0, 0, 0]
        sol = sp.nsolve(equations, (q0, q1, q2, q3), initial_guess)
        return [[float(v) for v in sol]]
    except Exception as e:
        print(f"SymPy Solver failed: {e}")
        return []

def solve_with_sympy_exact(equations):
    q0, q1, q2, q3 = sp.symbols('q0 q1 q2 q3')
    try:
        solutions = sp.solve(equations, (q0, q1, q2, q3), dict=True)

        results = []
        for s in solutions:
            res = [float(s[q0]), float(s[q1]), float(s[q2]), float(s[q3])]
            results.append(res)
        return results
    except Exception as e:
        print(f"SymPy Exact Solver failed: {e}")
        return []

def solve_with_sympy_poly(equations):
    q0, q1, q2, q3 = sp.symbols('q0 q1 q2 q3')
    try:
        solutions = sp.solve_poly_system(equations, q0, q1, q2, q3)
        return [[float(v) for v in sol] for sol in solutions]
    except Exception as e:
        print(f"SymPy Poly Solver failed: {e}")
        return []

def solve_with_sympy_poly_optimized(equations):
    vars = sp.symbols('q0 q1 q2 q3')
    try:
        print("Converting equations to Polynomial objects...")
        poly_eqs = [sp.Poly(e, *vars, domain='QQ') for e in equations]
        print("Computing Groebner Basis (This is the heavy part)...")
        basis = groebner(poly_eqs, *vars, order='lex')
        print("Finished Groebner Basis computation.")
        solutions = sp.solve_poly_system(basis, *vars)

        return [[float(v) for v in sol] for sol in solutions]
    except Exception as e:
        print(f"Failed: {e}")
        return []

def solve_with_sympy_fast(equations):
    q0, q1, q2, q3 = sp.symbols('q0 q1 q2 q3')

    guesses = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.5, 0.5, 0.5, 0.5]
    ]

    results = []
    for guess in guesses:
        try:
            sol = sp.nsolve(equations, (q0, q1, q2, q3), guess, tol=1e-6)
            q_sol = [float(v) for v in sol]
            if not any(np.allclose(q_sol, r, atol=1e-3) for r in results):
                results.append(q_sol)
        except:
            continue

    return results



# solving with sympy solver and with tracing
_MSOLVE_TRACER_CACHE = {}

def solve_with_sympy_traced(eqs):
    vars_sym = sp.symbols('q0 q1 q2 q3')
    num_eqs = len(eqs)
    trace_key = (num_eqs, len(vars_sym))

    try:
        gb = sp.groebner(eqs, *vars_sym, order='grlex')

        if trace_key not in _MSOLVE_TRACER_CACHE:
            poly_gb = [sp.Poly(g, *vars_sym) for g in gb]
            leading_monomials = [p.LM for p in poly_gb]

            basis_monomials = []
            max_deg = sum(p.total_degree() for p in poly_gb)
            for deg in range(max_deg + 1):
                for mono in sp.itermonomials(vars_sym, deg):
                    m_poly = sp.Poly(mono, *vars_sym)
                    m_exp = m_poly.degree_list()
                    is_divisible = False
                    for lm in leading_monomials:
                        lm_exp = sp.Poly(lm, *vars_sym).degree_list()
                        if all(m_exp[i] >= lm_exp[i] for i in range(len(vars_sym))):
                            is_divisible = True
                            break
                    if not is_divisible:
                        basis_monomials.append(mono)
            _MSOLVE_TRACER_CACHE[trace_key] = basis_monomials

        basis_monomials = _MSOLVE_TRACER_CACHE[trace_key]

        matrix_data = []
        for m in basis_monomials:
            rem_poly = sp.Poly(m * vars_sym[3], *vars_sym).rem(gb)
            coeffs_dict = rem_poly.as_dict()

            row = []
            for b in basis_monomials:
                b_mono = sp.Poly(b, *vars_sym).monoms()[0]
                val = coeffs_dict.get(b_mono, 0)
                row.append(float(val.evalf()) if hasattr(val, 'evalf') else float(val))
            matrix_data.append(row)

        M = np.array(matrix_data).T

        vals_q3 = np.linalg.eigvals(M)

        solutions = []
        for val in vals_q3:
            if np.isreal(val):
                try:
                    sol = sp.nsolve(eqs, vars_sym, [1.0, 0.0, 0.0, float(val.real)])
                    solutions.append([float(x) for x in sol])
                except:
                    continue

        return solutions

    except Exception as e:
        print(f"Traced solver error: {e}")
        return []