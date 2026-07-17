package com.performance;

import com.decomp.CholeskyDecomposition;

public class BlockedCholeskySweep {

    private static final int GEMM_TILE_ROWS = 8;

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

        float[][] right = new float[kb][m];
        for (int p = 0; p < kb; p++) {
            float[] dst = right[p];
            for (int j = 0; j < m; j++) {
                dst[j] = a[rest + j][kk + p];
            }
        }

        float[][] product = gemm.multiply(left, right, GEMM_TILE_ROWS);

        for (int j = 0; j < m; j++) {
            for (int i = j; i < m; i++) {
                a[rest + i][rest + j] -= product[i][j];
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
