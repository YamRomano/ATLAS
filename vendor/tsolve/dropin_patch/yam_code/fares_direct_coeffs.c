#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int e0, e1, e2, e3;
    double c;
} Term4;

static int inv3(const double A[9], double B[9]) {
    double det =
        A[0] * (A[4] * A[8] - A[5] * A[7])
        - A[1] * (A[3] * A[8] - A[5] * A[6])
        + A[2] * (A[3] * A[7] - A[4] * A[6]);
    if (fabs(det) < 1e-15) return -1;
    double invdet = 1.0 / det;
    B[0] = (A[4] * A[8] - A[5] * A[7]) * invdet;
    B[1] = (A[2] * A[7] - A[1] * A[8]) * invdet;
    B[2] = (A[1] * A[5] - A[2] * A[4]) * invdet;
    B[3] = (A[5] * A[6] - A[3] * A[8]) * invdet;
    B[4] = (A[0] * A[8] - A[2] * A[6]) * invdet;
    B[5] = (A[2] * A[3] - A[0] * A[5]) * invdet;
    B[6] = (A[3] * A[7] - A[4] * A[6]) * invdet;
    B[7] = (A[1] * A[6] - A[0] * A[7]) * invdet;
    B[8] = (A[0] * A[4] - A[1] * A[3]) * invdet;
    return 0;
}

static int idx4(int a, int b, int c, int d) {
    return (((a * 5) + b) * 5 + c) * 5 + d;
}

static int idx3(int a, int b, int c) {
    return ((a * 5) + b) * 5 + c;
}

static void add_eq_term(double eq[3][125], int eqi, int e0, int e1, int e2, int e3, double coeff, double clear_scale) {
    (void)e0;
    if (e1 < 0 || e2 < 0 || e3 < 0 || e1 > 4 || e2 > 4 || e3 > 4) return;
    eq[eqi][idx3(e1, e2, e3)] += clear_scale * coeff;
}

int fares_direct_coeffs(
    int n,
    const double *K,
    const double *p3d,
    const double *p2d,
    const double *weights,
    int n_terms,
    const int *term_eq,
    const int *exp_a,
    const int *exp_b,
    const int *exp_c,
    double round_scale,
    double clear_scale,
    double *out_coeffs
) {
    if (n < 4 || !K || !p3d || !p2d || !weights || !out_coeffs) return -1;
    if (round_scale <= 0.0) round_scale = 100.0;
    double invK[9];
    if (inv3(K, invK) != 0) return -2;

    double *rays = (double *)calloc((size_t)n * 3, sizeof(double));
    double *pnorm = (double *)calloc((size_t)n * 3, sizeof(double));
    if (!rays || !pnorm) {
        free(rays);
        free(pnorm);
        return -3;
    }

    int valid_count = 0;
    double wsum = 0.0;
    double mean[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < n; ++i) {
        double w = weights[i];
        if (w <= 1e-9) continue;
        valid_count++;
        wsum += w;
        mean[0] += w * p3d[3 * i + 0];
        mean[1] += w * p3d[3 * i + 1];
        mean[2] += w * p3d[3 * i + 2];
    }
    if (valid_count < 4 || wsum <= 0.0) {
        free(rays);
        free(pnorm);
        return -4;
    }
    mean[0] /= wsum;
    mean[1] /= wsum;
    mean[2] /= wsum;

    double dist_sum = 0.0;
    for (int i = 0; i < n; ++i) {
        if (weights[i] <= 1e-9) continue;
        double x = p3d[3 * i + 0] - mean[0];
        double y = p3d[3 * i + 1] - mean[1];
        double z = p3d[3 * i + 2] - mean[2];
        dist_sum += sqrt(x * x + y * y + z * z);
    }
    double dist_avg = dist_sum / (double)valid_count;
    if (dist_avg <= 1e-15) {
        free(rays);
        free(pnorm);
        return -5;
    }
    double scale3 = sqrt(3.0) / dist_avg;

    for (int i = 0; i < n; ++i) {
        double uv0 = p2d[2 * i + 0], uv1 = p2d[2 * i + 1];
        double r0 = invK[0] * uv0 + invK[1] * uv1 + invK[2];
        double r1 = invK[3] * uv0 + invK[4] * uv1 + invK[5];
        double r2 = invK[6] * uv0 + invK[7] * uv1 + invK[8];
        double rn = sqrt(r0 * r0 + r1 * r1 + r2 * r2);
        if (rn <= 1e-15) rn = 1e-15;
        rays[3 * i + 0] = r0 / rn;
        rays[3 * i + 1] = r1 / rn;
        rays[3 * i + 2] = r2 / rn;
        pnorm[3 * i + 0] = (p3d[3 * i + 0] - mean[0]) * scale3;
        pnorm[3 * i + 1] = (p3d[3 * i + 1] - mean[1]) * scale3;
        pnorm[3 * i + 2] = (p3d[3 * i + 2] - mean[2]) * scale3;
    }

    double W[9] = {0.0};
    double sumVK[27] = {0.0};
    for (int i = 0; i < n; ++i) {
        double w = weights[i];
        if (w <= 1e-9) continue;
        double r[3] = {rays[3 * i + 0], rays[3 * i + 1], rays[3 * i + 2]};
        double V[9];
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                V[3 * a + b] = (a == b ? 1.0 : 0.0) - r[a] * r[b];
                W[3 * a + b] += w * V[3 * a + b];
            }
        }
        double Ki[27] = {0.0};
        Ki[0] = pnorm[3 * i + 0]; Ki[1] = pnorm[3 * i + 1]; Ki[2] = pnorm[3 * i + 2];
        Ki[12] = pnorm[3 * i + 0]; Ki[13] = pnorm[3 * i + 1]; Ki[14] = pnorm[3 * i + 2];
        Ki[24] = pnorm[3 * i + 0]; Ki[25] = pnorm[3 * i + 1]; Ki[26] = pnorm[3 * i + 2];
        for (int a = 0; a < 3; ++a) {
            for (int c = 0; c < 9; ++c) {
                double s = 0.0;
                for (int b = 0; b < 3; ++b) s += V[3 * a + b] * Ki[9 * b + c];
                sumVK[9 * a + c] += w * s;
            }
        }
    }
    double Winv[9];
    if (inv3(W, Winv) != 0) {
        free(rays);
        free(pnorm);
        return -6;
    }
    double T[27] = {0.0};
    for (int a = 0; a < 3; ++a) {
        for (int c = 0; c < 9; ++c) {
            double s = 0.0;
            for (int b = 0; b < 3; ++b) s += Winv[3 * a + b] * sumVK[9 * b + c];
            T[9 * a + c] = -s;
        }
    }

    double Q[81] = {0.0};
    for (int i = 0; i < n; ++i) {
        double w = weights[i];
        if (w <= 1e-9) continue;
        double r[3] = {rays[3 * i + 0], rays[3 * i + 1], rays[3 * i + 2]};
        double V[9];
        for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) V[3 * a + b] = (a == b ? 1.0 : 0.0) - r[a] * r[b];
        double M[27];
        memcpy(M, T, sizeof(M));
        M[0] += pnorm[3 * i + 0]; M[1] += pnorm[3 * i + 1]; M[2] += pnorm[3 * i + 2];
        M[12] += pnorm[3 * i + 0]; M[13] += pnorm[3 * i + 1]; M[14] += pnorm[3 * i + 2];
        M[24] += pnorm[3 * i + 0]; M[25] += pnorm[3 * i + 1]; M[26] += pnorm[3 * i + 2];
        double A[27] = {0.0};
        for (int a = 0; a < 3; ++a) {
            for (int c = 0; c < 9; ++c) {
                for (int b = 0; b < 3; ++b) A[9 * a + c] += V[3 * a + b] * M[9 * b + c];
            }
        }
        for (int c1 = 0; c1 < 9; ++c1) {
            for (int c2 = 0; c2 < 9; ++c2) {
                double s = 0.0;
                for (int a = 0; a < 3; ++a) s += A[9 * a + c1] * A[9 * a + c2];
                Q[9 * c1 + c2] += w * s;
            }
        }
    }

    static const Term4 R[9][4] = {
        {{0,2,0,0,1},{2,0,0,0,1},{0,0,2,0,-1},{0,0,0,2,-1}},
        {{0,1,1,0,2},{1,0,0,1,-2},{0,0,0,0,0},{0,0,0,0,0}},
        {{0,1,0,1,2},{1,0,1,0,2},{0,0,0,0,0},{0,0,0,0,0}},
        {{0,1,1,0,2},{1,0,0,1,2},{0,0,0,0,0},{0,0,0,0,0}},
        {{2,0,0,0,1},{0,2,0,0,-1},{0,0,2,0,1},{0,0,0,2,-1}},
        {{0,0,1,1,2},{1,1,0,0,-2},{0,0,0,0,0},{0,0,0,0,0}},
        {{0,1,0,1,2},{1,0,1,0,-2},{0,0,0,0,0},{0,0,0,0,0}},
        {{0,0,1,1,2},{1,1,0,0,2},{0,0,0,0,0},{0,0,0,0,0}},
        {{2,0,0,0,1},{0,2,0,0,-1},{0,0,2,0,-1},{0,0,0,2,1}},
    };
    static const int Rn[9] = {4,2,2,2,4,2,2,2,4};

    double obj[625] = {0.0};
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            double q = nearbyint(Q[9 * i + j] * round_scale) / round_scale;
            if (q == 0.0) continue;
            for (int a = 0; a < Rn[i]; ++a) {
                for (int b = 0; b < Rn[j]; ++b) {
                    int e0 = R[i][a].e0 + R[j][b].e0;
                    int e1 = R[i][a].e1 + R[j][b].e1;
                    int e2 = R[i][a].e2 + R[j][b].e2;
                    int e3 = R[i][a].e3 + R[j][b].e3;
                    obj[idx4(e0, e1, e2, e3)] += q * R[i][a].c * R[j][b].c;
                }
            }
        }
    }

    double eq[3][125];
    memset(eq, 0, sizeof(eq));
    for (int e0 = 0; e0 <= 4; ++e0) {
        for (int e1 = 0; e1 <= 4; ++e1) {
            for (int e2 = 0; e2 <= 4; ++e2) {
                for (int e3 = 0; e3 <= 4; ++e3) {
                    double c = obj[idx4(e0, e1, e2, e3)];
                    if (c == 0.0) continue;
                    int e[4] = {e0, e1, e2, e3};
                    for (int gv = 0; gv < 4; ++gv) {
                        if (e[gv] == 0) continue;
                        double gc = c * (double)e[gv];
                        int g[4] = {e0, e1, e2, e3};
                        g[gv] -= 1;
                        if (gv == 1) add_eq_term(eq, 0, g[0] + 1, g[1], g[2], g[3], gc, clear_scale);
                        if (gv == 0) add_eq_term(eq, 0, g[0], g[1] + 1, g[2], g[3], -gc, clear_scale);
                        if (gv == 2) add_eq_term(eq, 1, g[0], g[1] + 1, g[2], g[3], gc, clear_scale);
                        if (gv == 1) add_eq_term(eq, 1, g[0], g[1], g[2] + 1, g[3], -gc, clear_scale);
                        if (gv == 3) add_eq_term(eq, 2, g[0], g[1], g[2] + 1, g[3], gc, clear_scale);
                        if (gv == 2) add_eq_term(eq, 2, g[0], g[1], g[2], g[3] + 1, -gc, clear_scale);
                    }
                }
            }
        }
    }

    for (int k = 0; k < n_terms; ++k) {
        int qi = term_eq[k], a = exp_a[k], b = exp_b[k], c = exp_c[k];
        if (qi < 0 || qi >= 3 || a < 0 || a > 4 || b < 0 || b > 4 || c < 0 || c > 4) {
            out_coeffs[k] = 0.0;
        } else {
            out_coeffs[k] = eq[qi][idx3(a, b, c)];
        }
    }

    free(rays);
    free(pnorm);
    return 0;
}
