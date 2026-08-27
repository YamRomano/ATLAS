#include <complex.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef PNP_USE_LAPACK
extern void zgeev_(
    const char *jobvl,
    const char *jobvr,
    const int *n,
    double complex *a,
    const int *lda,
    double complex *w,
    double complex *vl,
    const int *ldvl,
    double complex *vr,
    const int *ldvr,
    double complex *work,
    const int *lwork,
    double *rwork,
    int *info
);

extern void zgesv_(
    const int *n,
    const int *nrhs,
    double complex *a,
    const int *lda,
    int *ipiv,
    double complex *b,
    const int *ldb,
    int *info
);
#endif

static double wall_seconds(void) {
#ifdef CLOCK_MONOTONIC
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
#else
    return (double)clock() / (double)CLOCKS_PER_SEC;
#endif
}

static double complex cpowi_int(double complex z, int n) {
    double complex out = 1.0 + 0.0 * I;
    for (int i = 0; i < n; ++i) {
        out *= z;
    }
    return out;
}

static int solve3(double complex A[3][3], double complex b[3], double complex x[3]) {
    double complex M[3][4];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            M[i][j] = A[i][j];
        }
        M[i][3] = b[i];
    }

    for (int k = 0; k < 3; ++k) {
        int piv = k;
        double best = cabs(M[k][k]);
        for (int i = k + 1; i < 3; ++i) {
            double v = cabs(M[i][k]);
            if (v > best) {
                best = v;
                piv = i;
            }
        }
        if (best < 1e-24 || !isfinite(best)) {
            return 0;
        }
        if (piv != k) {
            for (int j = k; j < 4; ++j) {
                double complex tmp = M[k][j];
                M[k][j] = M[piv][j];
                M[piv][j] = tmp;
            }
        }
        double complex diag = M[k][k];
        for (int j = k; j < 4; ++j) {
            M[k][j] /= diag;
        }
        for (int i = 0; i < 3; ++i) {
            if (i == k) {
                continue;
            }
            double complex f = M[i][k];
            for (int j = k; j < 4; ++j) {
                M[i][j] -= f * M[k][j];
            }
        }
    }

    for (int i = 0; i < 3; ++i) {
        x[i] = M[i][3];
    }
    return 1;
}

static void eval_system(
    double complex x,
    double complex y,
    double complex z,
    int n_terms,
    const int *term_eq,
    const int *exp_a,
    const int *exp_b,
    const int *exp_c,
    const double *coeff_re,
    const double *coeff_im,
    double complex f[3],
    double complex J[3][3]
) {
    for (int i = 0; i < 3; ++i) {
        f[i] = 0.0 + 0.0 * I;
        for (int j = 0; j < 3; ++j) {
            J[i][j] = 0.0 + 0.0 * I;
        }
    }

    double complex xp[8], yp[8], zp[8];
    xp[0] = yp[0] = zp[0] = 1.0 + 0.0 * I;
    for (int i = 1; i < 8; ++i) {
        xp[i] = xp[i - 1] * x;
        yp[i] = yp[i - 1] * y;
        zp[i] = zp[i - 1] * z;
    }

    for (int t = 0; t < n_terms; ++t) {
        int eq = term_eq[t];
        int a = exp_a[t];
        int b = exp_b[t];
        int c = exp_c[t];
        if (eq < 0 || eq >= 3 || a < 0 || b < 0 || c < 0 || a >= 8 || b >= 8 || c >= 8) {
            continue;
        }
        double complex coeff = coeff_re[t] + coeff_im[t] * I;
        double complex mon = xp[a] * yp[b] * zp[c];
        f[eq] += coeff * mon;
        if (a > 0) {
            J[eq][0] += coeff * (double)a * xp[a - 1] * yp[b] * zp[c];
        }
        if (b > 0) {
            J[eq][1] += coeff * (double)b * xp[a] * yp[b - 1] * zp[c];
        }
        if (c > 0) {
            J[eq][2] += coeff * (double)c * xp[a] * yp[b] * zp[c - 1];
        }
    }
}

static double relative_residual(double complex f[3], const double norms[3]) {
    double r = 0.0;
    for (int i = 0; i < 3; ++i) {
        double denom = norms[i] > 1.0 ? norms[i] : 1.0;
        double v = cabs(f[i]) / denom;
        if (v > r) {
            r = v;
        }
    }
    return r;
}

int pnp_refine_roots(
    int n_seeds,
    const double *seed_re,
    const double *seed_im,
    int n_terms,
    const int *term_eq,
    const int *exp_a,
    const int *exp_b,
    const int *exp_c,
    const double *coeff_re,
    const double *coeff_im,
    const double *norms,
    int max_iter,
    double target_rel_tol,
    double residual_tol,
    double max_abs_root,
    int target_ok_roots,
    double *out_re,
    double *out_im,
    double *out_residual,
    int *out_ok,
    int *out_iters
) {
    int ok_count = 0;
    const double dampings[6] = {1.0, 0.5, 0.25, 0.1, 0.05, 0.01};

    for (int s = 0; s < n_seeds; ++s) {
        double complex r[3];
        double complex best[3];
        for (int j = 0; j < 3; ++j) {
            r[j] = seed_re[3 * s + j] + seed_im[3 * s + j] * I;
            best[j] = r[j];
        }

        double best_rel = INFINITY;
        int best_iter = 0;
        int converged = 0;

        for (int it = 1; it <= max_iter; ++it) {
            double complex f[3];
            double complex J[3][3];
            eval_system(r[0], r[1], r[2], n_terms, term_eq, exp_a, exp_b, exp_c, coeff_re, coeff_im, f, J);
            double rel = relative_residual(f, norms);
            if (isfinite(rel) && rel < best_rel) {
                best_rel = rel;
                best_iter = it;
                best[0] = r[0];
                best[1] = r[1];
                best[2] = r[2];
            }
            if (rel <= target_rel_tol) {
                converged = 1;
                break;
            }

            double complex rhs[3] = {-f[0], -f[1], -f[2]};
            double complex step[3];
            if (!solve3(J, rhs, step)) {
                break;
            }

            int improved = 0;
            for (int d = 0; d < 6; ++d) {
                double complex cand[3] = {
                    r[0] + dampings[d] * step[0],
                    r[1] + dampings[d] * step[1],
                    r[2] + dampings[d] * step[2],
                };
                double complex fc[3];
                double complex Jc[3][3];
                eval_system(cand[0], cand[1], cand[2], n_terms, term_eq, exp_a, exp_b, exp_c, coeff_re, coeff_im, fc, Jc);
                double cand_rel = relative_residual(fc, norms);
                if (isfinite(cand_rel) && cand_rel < rel) {
                    r[0] = cand[0];
                    r[1] = cand[1];
                    r[2] = cand[2];
                    improved = 1;
                    break;
                }
            }
            if (!improved) {
                r[0] += step[0];
                r[1] += step[1];
                r[2] += step[2];
            }
        }

        double max_abs = fmax(cabs(best[0]), fmax(cabs(best[1]), cabs(best[2])));
        int ok = (best_rel <= residual_tol) && (max_abs <= max_abs_root);
        if (ok) {
            ok_count += 1;
            if (target_ok_roots > 0 && ok_count >= target_ok_roots) {
                for (int j = 0; j < 3; ++j) {
                    out_re[3 * s + j] = creal(best[j]);
                    out_im[3 * s + j] = cimag(best[j]);
                }
                out_residual[s] = best_rel;
                out_ok[s] = ok;
                out_iters[s] = converged ? best_iter : max_iter;
                return ok_count;
            }
        }

        for (int j = 0; j < 3; ++j) {
            out_re[3 * s + j] = creal(best[j]);
            out_im[3 * s + j] = cimag(best[j]);
        }
        out_residual[s] = best_rel;
        out_ok[s] = ok;
        out_iters[s] = converged ? best_iter : max_iter;
    }

    return ok_count;
}

int pnp_extract_roots_from_actions(
    int qdim,
    const double *ax,
    const double *ay,
    const double *az,
    int n_weights,
    const double *weights,
    int n_terms,
    const int *term_eq,
    const int *exp_a,
    const int *exp_b,
    const int *exp_c,
    const double *coeff_re,
    const double *coeff_im,
    const double *norms,
    int max_iter,
    double target_rel_tol,
    double residual_tol,
    double max_abs_root,
    int target_ok_roots,
    int max_seeds,
    double *out_re,
    double *out_im,
    double *out_residual,
    int *out_ok,
    int *out_iters,
    int *out_seed_count,
    double *out_eig_seconds,
    double *out_newton_seconds
) {
#ifndef PNP_USE_LAPACK
    (void)qdim; (void)ax; (void)ay; (void)az; (void)n_weights; (void)weights;
    (void)n_terms; (void)term_eq; (void)exp_a; (void)exp_b; (void)exp_c;
    (void)coeff_re; (void)coeff_im; (void)norms; (void)max_iter;
    (void)target_rel_tol; (void)residual_tol; (void)max_abs_root; (void)target_ok_roots;
    (void)max_seeds; (void)out_re; (void)out_im; (void)out_residual;
    (void)out_ok; (void)out_iters; (void)out_seed_count;
    (void)out_eig_seconds; (void)out_newton_seconds;
    return -90;
#else
    if (qdim <= 0 || n_weights <= 0 || max_seeds < qdim * n_weights) {
        return -1;
    }
    if (!ax || !ay || !az || !weights || !out_re || !out_im || !out_residual || !out_ok || !out_iters) {
        return -2;
    }

    const int n = qdim;
    const int lda = qdim;
    const int ldvl = 1;
    const int ldvr = qdim;
    const int nrhs = qdim;
    const int ldb = qdim;
    const char jobvl = 'N';
    const char jobvr = 'V';
    int seed_count = 0;
    int info = 0;
    double eig_start = wall_seconds();
    double newton_seconds_total = 0.0;
    int ok_count = 0;

    double *seed_re = (double *)calloc((size_t)max_seeds * 3, sizeof(double));
    double *seed_im = (double *)calloc((size_t)max_seeds * 3, sizeof(double));
    double complex *T = (double complex *)malloc((size_t)n * n * sizeof(double complex));
    double complex *W = (double complex *)malloc((size_t)n * sizeof(double complex));
    double complex *VR = (double complex *)malloc((size_t)n * n * sizeof(double complex));
    double complex *Vcopy = (double complex *)malloc((size_t)n * n * sizeof(double complex));
    double complex *B = (double complex *)malloc((size_t)n * n * sizeof(double complex));
    double complex *work = NULL;
    double *rwork = (double *)malloc((size_t)2 * n * sizeof(double));
    int *ipiv = (int *)malloc((size_t)n * sizeof(int));
    if (!seed_re || !seed_im || !T || !W || !VR || !Vcopy || !B || !rwork || !ipiv) {
        free(seed_re); free(seed_im); free(T); free(W); free(VR); free(Vcopy); free(B); free(rwork); free(ipiv);
        return -3;
    }

    for (int wi = 0; wi < n_weights; ++wi) {
        double wx = weights[3 * wi + 0];
        double wy = weights[3 * wi + 1];
        double wz = weights[3 * wi + 2];

        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < n; ++row) {
                size_t rm = (size_t)row * n + col;
                T[(size_t)col * n + row] = wx * ax[rm] + wy * ay[rm] + wz * az[rm];
            }
        }

        double complex work_query = 0.0 + 0.0 * I;
        int lwork = -1;
        zgeev_(&jobvl, &jobvr, &n, T, &lda, W, NULL, &ldvl, VR, &ldvr, &work_query, &lwork, rwork, &info);
        if (info != 0) {
            continue;
        }
        lwork = (int)creal(work_query);
        if (lwork < 4 * n) {
            lwork = 4 * n;
        }
        work = (double complex *)malloc((size_t)lwork * sizeof(double complex));
        if (!work) {
            free(seed_re); free(seed_im); free(T); free(W); free(VR); free(Vcopy); free(B); free(rwork); free(ipiv);
            return -4;
        }

        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < n; ++row) {
                size_t rm = (size_t)row * n + col;
                T[(size_t)col * n + row] = wx * ax[rm] + wy * ay[rm] + wz * az[rm];
            }
        }
        zgeev_(&jobvl, &jobvr, &n, T, &lda, W, NULL, &ldvl, VR, &ldvr, work, &lwork, rwork, &info);
        free(work);
        work = NULL;
        if (info != 0) {
            continue;
        }

        const double *actions[3] = {ax, ay, az};
        double complex coords[3][256];
        if (n > 256) {
            free(seed_re); free(seed_im); free(T); free(W); free(VR); free(Vcopy); free(B); free(rwork); free(ipiv);
            return -5;
        }

        int action_ok = 1;
        for (int ai = 0; ai < 3; ++ai) {
            const double *A = actions[ai];
            for (int col = 0; col < n; ++col) {
                for (int row = 0; row < n; ++row) {
                    double complex sum = 0.0 + 0.0 * I;
                    for (int k = 0; k < n; ++k) {
                        sum += A[(size_t)row * n + k] * VR[(size_t)col * n + k];
                    }
                    B[(size_t)col * n + row] = sum;
                }
            }
            memcpy(Vcopy, VR, (size_t)n * n * sizeof(double complex));
            zgesv_(&n, &nrhs, Vcopy, &lda, ipiv, B, &ldb, &info);
            if (info != 0) {
                action_ok = 0;
                break;
            }
            for (int i = 0; i < n; ++i) {
                coords[ai][i] = B[(size_t)i * n + i];
            }
        }
        if (!action_ok) {
            continue;
        }

        int weight_seed_start = seed_count;
        for (int i = 0; i < n; ++i) {
            if (seed_count >= max_seeds) {
                break;
            }
            for (int ai = 0; ai < 3; ++ai) {
                seed_re[3 * seed_count + ai] = creal(coords[ai][i]);
                seed_im[3 * seed_count + ai] = cimag(coords[ai][i]);
            }
            seed_count += 1;
        }
        int n_new = seed_count - weight_seed_start;
        if (n_new > 0) {
            double newton_start = wall_seconds();
            int remaining = target_ok_roots > 0 ? target_ok_roots - ok_count : 0;
            int new_ok = pnp_refine_roots(
                n_new,
                seed_re + 3 * weight_seed_start,
                seed_im + 3 * weight_seed_start,
                n_terms,
                term_eq,
                exp_a,
                exp_b,
                exp_c,
                coeff_re,
                coeff_im,
                norms,
                max_iter,
                target_rel_tol,
                residual_tol,
                max_abs_root,
                remaining,
                out_re + 3 * weight_seed_start,
                out_im + 3 * weight_seed_start,
                out_residual + weight_seed_start,
                out_ok + weight_seed_start,
                out_iters + weight_seed_start
            );
            newton_seconds_total += wall_seconds() - newton_start;
            ok_count += new_ok;
            if (target_ok_roots > 0 && ok_count >= target_ok_roots) {
                break;
            }
        }
    }

    double eig_seconds = wall_seconds() - eig_start;
    if (out_seed_count) {
        out_seed_count[0] = seed_count;
    }
    if (out_eig_seconds) {
        out_eig_seconds[0] = eig_seconds;
    }

    if (out_newton_seconds) {
        out_newton_seconds[0] = newton_seconds_total;
    }

    free(seed_re); free(seed_im); free(T); free(W); free(VR); free(Vcopy); free(B); free(rwork); free(ipiv);
    return ok_count;
#endif
}

static int find_col_index(
    int a,
    int b,
    int c,
    int ncols,
    const int *col_a,
    const int *col_b,
    const int *col_c
) {
    for (int i = 0; i < ncols; ++i) {
        if (col_a[i] == a && col_b[i] == b && col_c[i] == c) {
            return i;
        }
    }
    return -1;
}

static int solve_complex_system(
    int n,
    int nrhs,
    double complex *G,
    double complex *H
) {
    for (int k = 0; k < n; ++k) {
        int piv = k;
        double best = cabs(G[(size_t)k * n + k]);
        for (int i = k + 1; i < n; ++i) {
            double v = cabs(G[(size_t)i * n + k]);
            if (v > best) {
                best = v;
                piv = i;
            }
        }
        if (best < 1e-24 || !isfinite(best)) {
            return 0;
        }
        if (piv != k) {
            for (int j = k; j < n; ++j) {
                double complex tmp = G[(size_t)k * n + j];
                G[(size_t)k * n + j] = G[(size_t)piv * n + j];
                G[(size_t)piv * n + j] = tmp;
            }
            for (int r = 0; r < nrhs; ++r) {
                double complex tmp = H[(size_t)k * nrhs + r];
                H[(size_t)k * nrhs + r] = H[(size_t)piv * nrhs + r];
                H[(size_t)piv * nrhs + r] = tmp;
            }
        }

        double complex diag = G[(size_t)k * n + k];
        for (int i = k + 1; i < n; ++i) {
            double complex factor = G[(size_t)i * n + k] / diag;
            G[(size_t)i * n + k] = 0.0 + 0.0 * I;
            for (int j = k + 1; j < n; ++j) {
                G[(size_t)i * n + j] -= factor * G[(size_t)k * n + j];
            }
            for (int r = 0; r < nrhs; ++r) {
                H[(size_t)i * nrhs + r] -= factor * H[(size_t)k * nrhs + r];
            }
        }
    }

    for (int r = 0; r < nrhs; ++r) {
        for (int i = n - 1; i >= 0; --i) {
            double complex sum = H[(size_t)i * nrhs + r];
            for (int j = i + 1; j < n; ++j) {
                sum -= G[(size_t)i * n + j] * H[(size_t)j * nrhs + r];
            }
            double complex diag = G[(size_t)i * n + i];
            if (cabs(diag) < 1e-24 || !isfinite(cabs(diag))) {
                return 0;
            }
            H[(size_t)i * nrhs + r] = sum / diag;
        }
    }
    return 1;
}

#ifdef PNP_USE_LAPACK
extern void zgesv_(
    const int *n,
    const int *nrhs,
    double complex *a,
    const int *lda,
    int *ipiv,
    double complex *b,
    const int *ldb,
    int *info
);

static int solve_complex_system_lapack(
    int n,
    int nrhs,
    double complex *G,
    double complex *H
) {
    double complex *A = (double complex *)malloc((size_t)n * n * sizeof(double complex));
    double complex *B = (double complex *)malloc((size_t)n * nrhs * sizeof(double complex));
    int *ipiv = (int *)malloc((size_t)n * sizeof(int));
    if (!A || !B || !ipiv) {
        free(A);
        free(B);
        free(ipiv);
        return 0;
    }

    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            A[(size_t)col * n + row] = G[(size_t)row * n + col];
        }
    }
    for (int row = 0; row < n; ++row) {
        for (int rhs = 0; rhs < nrhs; ++rhs) {
            B[(size_t)rhs * n + row] = H[(size_t)row * nrhs + rhs];
        }
    }

    int info = 0;
    int lda = n;
    int ldb = n;
    zgesv_(&n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
    if (info != 0) {
        free(A);
        free(B);
        free(ipiv);
        return 0;
    }

    for (int row = 0; row < n; ++row) {
        for (int rhs = 0; rhs < nrhs; ++rhs) {
            H[(size_t)row * nrhs + rhs] = B[(size_t)rhs * n + row];
        }
    }

    free(A);
    free(B);
    free(ipiv);
    return 1;
}
#endif

static int solve_complex_system_auto(
    int n,
    int nrhs,
    double complex *G,
    double complex *H
) {
#ifdef PNP_USE_LAPACK
    return solve_complex_system_lapack(n, nrhs, G, H);
#else
    return solve_complex_system(n, nrhs, G, H);
#endif
}

int pnp_project_uses_lapack(void) {
#ifdef PNP_USE_LAPACK
    return 1;
#else
    return 0;
#endif
}

int pnp_project_actions(
    int n_terms,
    const int *term_eq,
    const int *term_a,
    const int *term_b,
    const int *term_c,
    const double *coeff_re,
    const double *coeff_im,
    int ncols,
    const int *col_a,
    const int *col_b,
    const int *col_c,
    int nsel,
    const int *sel_eq,
    const int *sel_a,
    const int *sel_b,
    const int *sel_c,
    int qdim,
    const int *qbasis,
    int ntarget,
    const int *targets,
    double *actions_re,
    double *actions_im,
    double *projection_residual,
    int *missing_targets
) {
    if (ncols <= 0 || nsel <= 0 || qdim <= 0 || ntarget != 3 * qdim) {
        return 0;
    }

    int nunknown = nsel + qdim;
    size_t M_count = (size_t)nsel * (size_t)ncols;
    size_t G_count = (size_t)nunknown * (size_t)nunknown;
    size_t H_count = (size_t)nunknown * (size_t)ntarget;

    double complex *M = (double complex *)calloc(M_count, sizeof(double complex));
    double complex *G = (double complex *)calloc(G_count, sizeof(double complex));
    double complex *H = (double complex *)calloc(H_count, sizeof(double complex));
    int *target_valid = (int *)calloc((size_t)ntarget, sizeof(int));
    if (!M || !G || !H || !target_valid) {
        free(M);
        free(G);
        free(H);
        free(target_valid);
        return 0;
    }

    for (int r = 0; r < nsel; ++r) {
        int eq_idx = sel_eq[r];
        int ma = sel_a[r];
        int mb = sel_b[r];
        int mc = sel_c[r];
        for (int t = 0; t < n_terms; ++t) {
            if (term_eq[t] != eq_idx) {
                continue;
            }
            int target_col = find_col_index(
                ma + term_a[t],
                mb + term_b[t],
                mc + term_c[t],
                ncols,
                col_a,
                col_b,
                col_c
            );
            if (target_col >= 0) {
                M[(size_t)r * ncols + target_col] += coeff_re[t] + coeff_im[t] * I;
            }
        }
    }

    int missing = 0;
    for (int k = 0; k < ntarget; ++k) {
        int col = targets[k];
        if (col >= 0 && col < ncols) {
            target_valid[k] = 1;
        } else {
            missing += 1;
        }
    }

    /* Build A^H A without materializing the dense ncols x (nsel+qdim) matrix.
       The selected Macaulay rows are sparse; exploiting that structure is the
       main point of this C path. */
    for (int col = 0; col < ncols; ++col) {
        for (int i = 0; i < nsel; ++i) {
            double complex vi = M[(size_t)i * ncols + col];
            if (vi == 0.0) {
                continue;
            }
            double complex cvi = conj(vi);
            for (int j = 0; j < nsel; ++j) {
                double complex vj = M[(size_t)j * ncols + col];
                if (vj != 0.0) {
                    G[(size_t)i * nunknown + j] += cvi * vj;
                }
            }
        }
    }

    for (int q = 0; q < qdim; ++q) {
        int qcol = qbasis[q];
        int uq = nsel + q;
        if (qcol < 0 || qcol >= ncols) {
            continue;
        }
        G[(size_t)uq * nunknown + uq] += 1.0 + 0.0 * I;
        for (int r = 0; r < nsel; ++r) {
            double complex v = M[(size_t)r * ncols + qcol];
            if (v != 0.0) {
                G[(size_t)r * nunknown + uq] += conj(v);
                G[(size_t)uq * nunknown + r] += v;
            }
        }
    }

    for (int k = 0; k < ntarget; ++k) {
        if (!target_valid[k]) {
            continue;
        }
        int col = targets[k];
        for (int r = 0; r < nsel; ++r) {
            H[(size_t)r * ntarget + k] = conj(M[(size_t)r * ncols + col]);
        }
        for (int q = 0; q < qdim; ++q) {
            if (qbasis[q] == col) {
                H[(size_t)(nsel + q) * ntarget + k] = 1.0 + 0.0 * I;
            }
        }
    }

    if (!solve_complex_system_auto(nunknown, ntarget, G, H)) {
        free(M);
        free(G);
        free(H);
        free(target_valid);
        return 0;
    }

    double num = 0.0;
    double den = 0.0;
    for (int col = 0; col < ncols; ++col) {
        for (int k = 0; k < ntarget; ++k) {
            double complex pred = 0.0 + 0.0 * I;
            for (int r = 0; r < nsel; ++r) {
                double complex v = M[(size_t)r * ncols + col];
                if (v != 0.0) {
                    pred += v * H[(size_t)r * ntarget + k];
                }
            }
            for (int q = 0; q < qdim; ++q) {
                if (qbasis[q] == col) {
                    pred += H[(size_t)(nsel + q) * ntarget + k];
                }
            }
            double complex rhs = (target_valid[k] && targets[k] == col) ? 1.0 + 0.0 * I : 0.0 + 0.0 * I;
            double complex diff = pred - rhs;
            num += creal(diff) * creal(diff) + cimag(diff) * cimag(diff);
            den += creal(rhs) * creal(rhs) + cimag(rhs) * cimag(rhs);
        }
    }
    if (projection_residual) {
        projection_residual[0] = sqrt(num) / fmax(1.0, sqrt(den));
    }
    if (missing_targets) {
        missing_targets[0] = missing;
    }

    for (int var = 0; var < 3; ++var) {
        for (int j = 0; j < qdim; ++j) {
            int k = 3 * j + var;
            for (int i = 0; i < qdim; ++i) {
                double complex v = H[(size_t)(nsel + i) * ntarget + k];
                size_t out_idx = (size_t)var * qdim * qdim + (size_t)i * qdim + j;
                actions_re[out_idx] = creal(v);
                actions_im[out_idx] = cimag(v);
            }
        }
    }

    free(M);
    free(G);
    free(H);
    free(target_valid);
    return 1;
}
