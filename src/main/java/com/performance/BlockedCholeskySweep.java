package com.performance;

import com.decomp.CholeskyDecomposition;

public class BlockedCholeskySweep {

    private static final int GEMM_TILE_ROWS = 8;

    private static final int SYRK_PANEL = 64;

    private final RegisterTileSweepMatrixOps gemm = new RegisterTileSweepMatrixOps();

    private static void checkSquare(float[][] matrix) {
        int n = matrix.length;
        if (n == 0) {
            throw new IllegalArgumentException("Matrix must not be empty");
        }
        for (float[] row : matrix) {
            if (row.length != n) {
                throw new IllegalArgumentException(
                        "Matrix must be square, got " + n + " rows but a row of length " + row.length);
            }
        }
    }

    public CholeskyDecomposition cholesky(float[][] matrix, int blockSize) {
        checkSquare(matrix);
        if (blockSize < 1) {
            throw new IllegalArgumentException("blockSize must be at least 1, got " + blockSize);
        }

        int n = matrix.length;

        float[][] a = new float[n][n];
        for (int r = 0; r < n; r++) {
            a[r] = matrix[r].clone();
        }

        for (int kk = 0; kk < n; kk += blockSize) {
            int kb = Math.min(blockSize, n - kk);
            factorPanel(a, kk, kb, n);

            int rest = kk + kb;
            if (rest < n) {
                updateTrailing(a, kk, kb, rest, n);
            }
        }

        return extractL(a, n);
    }

    private void updateTrailing(float[][] a, int kk, int kb, int rest, int n) {
        int m = n - rest;
        if (m == 0) {
            return;
        }

        float[][] left = new float[m][kb];
        for (int i = 0; i < m; i++) {
            System.arraycopy(a[rest + i], kk, left[i], 0, kb);
        }

        for (int c0 = 0; c0 < m; c0 += SYRK_PANEL) {
            int c1 = Math.min(c0 + SYRK_PANEL, m);
            int pw = c1 - c0;

            syrkDiagonal(a, left, rest, c0, c1, kb);

            int belowRows = m - c1;
            if (belowRows == 0) {
                continue;
            }

            float[][] below = new float[belowRows][];
            System.arraycopy(left, c1, below, 0, belowRows);

            float[][] right = new float[kb][pw];
            for (int p = 0; p < kb; p++) {
                float[] dst = right[p];
                for (int t = 0; t < pw; t++) {
                    dst[t] = left[c0 + t][p];
                }
            }

            float[][] product = gemm.multiply(below, right, GEMM_TILE_ROWS);

            for (int i = 0; i < belowRows; i++) {
                float[] arow = a[rest + c1 + i];
                float[] prow = product[i];
                for (int t = 0; t < pw; t++) {
                    arow[rest + c0 + t] -= prow[t];
                }
            }
        }
    }

    private static void syrkDiagonal(float[][] a, float[][] left, int rest,
                                     int c0, int c1, int kb) {
        for (int i = c0; i < c1; i++) {
            float[] li = left[i];
            float[] arow = a[rest + i];
            for (int j = c0; j <= i; j++) {
                float[] lj = left[j];
                float s = 0f;
                for (int p = 0; p < kb; p++) {
                    s += li[p] * lj[p];
                }
                arow[rest + j] -= s;
            }
        }
    }

    private static void factorPanel(float[][] a, int kk, int kb, int n) {
        int panelEnd = kk + kb;
        for (int k = kk; k < panelEnd; k++) {
            float diag = a[k][k];
            if (diag <= 0f) {
                throw new ArithmeticException("Matrix is not positive definite");
            }

            float root = (float) Math.sqrt(diag);
            a[k][k] = root;

            for (int i = k + 1; i < n; i++) {
                a[i][k] /= root;
            }

            for (int j = k + 1; j < panelEnd; j++) {
                float ajk = a[j][k];
                for (int i = j; i < n; i++) {
                    a[i][j] -= a[i][k] * ajk;
                }
            }
        }
    }

    private static CholeskyDecomposition extractL(float[][] a, int n) {
        float[][] l = new float[n][n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a[i], 0, l[i], 0, i + 1);
        }
        return new CholeskyDecomposition(l);
    }
}
