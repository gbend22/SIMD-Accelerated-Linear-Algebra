package com.performance;

import com.core.MatrixValidation;
import com.decomp.QRDecomposition;

/**
 * Blocked Householder QR decomposition using the compact WY representation. Each panel of
 * {@code blockSize} columns is reduced with ordinary Householder reflectors, then the
 * product of those reflectors is assembled as {@code I - V*T*Vᵀ} so that both the trailing
 * update and the formation of {@code Q} run as matrix multiplies through the register-tiled
 * GEMM kernel. Internal, not part of the public API.
 */
public class BlockedQrSweep {

    private static final int GEMM_TILE_ROWS = 8;

    private final RegisterTileSweepMatrixOps gemm = new RegisterTileSweepMatrixOps();

    /**
     * Factors a matrix with at least as many rows as columns as {@code A = Q * R} using
     * blocked Householder QR.
     *
     * @param matrix    an {@code m x n} matrix with {@code m >= n}
     * @param blockSize the panel width, in columns
     * @return the orthogonal factor {@code Q} and the upper-triangular factor {@code R}
     * @throws IllegalArgumentException if {@code matrix} is empty, not rectangular, has
     *         fewer rows than columns, or if {@code blockSize} is less than {@code 1}
     */
    public QRDecomposition qr(float[][] matrix, int blockSize) {
        int m = matrix.length;
        int n = MatrixValidation.requireRectangular(matrix, "matrix");
        if (m < n) {
            throw new IllegalArgumentException("QR requires rows >= columns, got " + m + "x" + n);
        }
        if (blockSize < 1) {
            throw new IllegalArgumentException("blockSize must be at least 1, got " + blockSize);
        }

        float[][] a = new float[m][n];
        for (int i = 0; i < m; i++) {
            a[i] = matrix[i].clone();
        }

        float[] tau = new float[n];

        for (int kk = 0; kk < n; kk += blockSize) {
            int panelEnd = Math.min(kk + blockSize, n);
            int kb = panelEnd - kk;

            for (int k = kk; k < panelEnd; k++) {
                computeReflector(a, k, m, tau);
                for (int j = k + 1; j < panelEnd; j++) {
                    applyReflectorToColumn(a, k, j, m, tau[k]);
                }
            }

            int cc = n - panelEnd;
            if (cc > 0) {
                float[][] vmat = buildV(a, kk, kb, m);
                float[][] vmatT = buildVt(a, kk, kb, m);
                float[][] t = buildT(a, kk, kb, m, tau);
                float[][] tT = transposeSquare(t, kb);
                applyBlockLeft(a, kk, m, panelEnd, cc, vmat, vmatT, tT);
            }
        }

        float[][] r = new float[m][n];
        for (int i = 0; i < m; i++) {
            int cols = Math.min(i, n);
            System.arraycopy(a[i], cols, r[i], cols, n - cols);
        }

        float[][] q = new float[m][m];
        for (int i = 0; i < m; i++) {
            q[i][i] = 1f;
        }
        if (n > 0) {
            int lastStart = ((n - 1) / blockSize) * blockSize;
            for (int kk = lastStart; kk >= 0; kk -= blockSize) {
                int panelEnd = Math.min(kk + blockSize, n);
                int kb = panelEnd - kk;
                float[][] vmat = buildV(a, kk, kb, m);
                float[][] vmatT = buildVt(a, kk, kb, m);
                float[][] t = buildT(a, kk, kb, m, tau);
                applyBlockLeft(q, kk, m, 0, m, vmat, vmatT, t);
            }
        }

        return new QRDecomposition(q, r);
    }

    private void applyBlockLeft(float[][] target, int kk, int m,
                                int c0, int cc, float[][] vmat, float[][] vmatT, float[][] tsel) {
        int rr = m - kk;

        float[][] b = new float[rr][cc];
        for (int i = 0; i < rr; i++) {
            System.arraycopy(target[kk + i], c0, b[i], 0, cc);
        }

        float[][] w = gemm.multiply(vmatT, b, GEMM_TILE_ROWS);
        float[][] w2 = gemm.multiply(tsel, w, GEMM_TILE_ROWS);
        float[][] p = gemm.multiply(vmat, w2, GEMM_TILE_ROWS);

        for (int i = 0; i < rr; i++) {
            float[] trow = target[kk + i];
            float[] prow = p[i];
            for (int t = 0; t < cc; t++) {
                trow[c0 + t] -= prow[t];
            }
        }
    }

    private static float[][] buildV(float[][] a, int kk, int kb, int m) {
        int rr = m - kk;
        float[][] vmat = new float[rr][kb];
        for (int col = 0; col < kb; col++) {
            int global = kk + col;
            vmat[col][col] = 1f;
            for (int i = global + 1; i < m; i++) {
                vmat[i - kk][col] = a[i][global];
            }
        }
        return vmat;
    }

    private static float[][] buildVt(float[][] a, int kk, int kb, int m) {
        int rr = m - kk;
        float[][] vmatT = new float[kb][rr];
        for (int col = 0; col < kb; col++) {
            int global = kk + col;
            vmatT[col][col] = 1f;
            for (int i = global + 1; i < m; i++) {
                vmatT[col][i - kk] = a[i][global];
            }
        }
        return vmatT;
    }

    private static float[][] buildT(float[][] a, int kk, int kb, int m, float[] tau) {
        float[][] t = new float[kb][kb];
        for (int p = 0; p < kb; p++) {
            int colP = kk + p;
            float taup = tau[colP];
            t[p][p] = taup;
            if (taup == 0f) {
                continue;
            }

            float[] z = new float[p];
            for (int q = 0; q < p; q++) {
                int colQ = kk + q;
                float dot = a[colP][colQ];
                for (int i = colP + 1; i < m; i++) {
                    dot += a[i][colQ] * a[i][colP];
                }
                z[q] = -taup * dot;
            }

            for (int row = 0; row < p; row++) {
                float s = 0f;
                for (int col = row; col < p; col++) {
                    s += t[row][col] * z[col];
                }
                t[row][p] = s;
            }
        }
        return t;
    }

    private static float[][] transposeSquare(float[][] t, int kb) {
        float[][] out = new float[kb][kb];
        for (int i = 0; i < kb; i++) {
            for (int j = 0; j < kb; j++) {
                out[j][i] = t[i][j];
            }
        }
        return out;
    }

    private static void computeReflector(float[][] a, int k, int m, float[] tau) {
        float normx = 0f;
        for (int i = k; i < m; i++) {
            float x = a[i][k];
            normx += x * x;
        }
        normx = (float) Math.sqrt(normx);
        if (normx == 0f) {
            tau[k] = 0f;
            return;
        }

        float xk = a[k][k];
        float alpha = xk >= 0f ? -normx : normx;
        float vk = xk - alpha;

        for (int i = k + 1; i < m; i++) {
            a[i][k] /= vk;
        }

        tau[k] = (alpha - xk) / alpha;
        a[k][k] = alpha;
    }

    private static void applyReflectorToColumn(float[][] a, int k, int j, int m, float tauk) {
        if (tauk == 0f) {
            return;
        }
        float s = a[k][j];
        for (int i = k + 1; i < m; i++) {
            s += a[i][k] * a[i][j];
        }
        s *= tauk;
        a[k][j] -= s;
        for (int i = k + 1; i < m; i++) {
            a[i][j] -= s * a[i][k];
        }
    }
}
