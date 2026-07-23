package com.performance;

import com.decomp.LUDecomposition;

/**
 * Right-looking blocked LU decomposition with partial pivoting. Each step factors a panel
 * of {@code blockSize} columns, applies the triangular update to the panel rows, and casts
 * the trailing-submatrix update as a single matrix multiply, so most of the work runs
 * through the register-tiled GEMM kernel. Internal, not part of the public API.
 */
public class BlockedLuSweep {

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

    /**
     * Factors a square matrix as {@code P * A = L * U} using blocked LU with partial
     * pivoting.
     *
     * @param matrix    a square {@code n x n} matrix
     * @param blockSize the panel width, in columns; larger panels shift more work into GEMM
     * @return the {@code L} and {@code U} factors together with the row permutation
     * @throws IllegalArgumentException if {@code matrix} is empty or not square, or if
     *         {@code blockSize} is less than {@code 1}
     */
    public LUDecomposition lu(float[][] matrix, int blockSize) {
        checkSquare(matrix);
        if (blockSize < 1) {
            throw new IllegalArgumentException("blockSize must be at least 1, got " + blockSize);
        }

        int n = matrix.length;

        float[][] a = new float[n][n];
        for (int r = 0; r < n; r++) {
            a[r] = matrix[r].clone();
        }

        int[] pivot = new int[n];
        for (int i = 0; i < n; i++) {
            pivot[i] = i;
        }
        int[] sign = {1};

        for (int kk = 0; kk < n; kk += blockSize) {
            int kb = Math.min(blockSize, n - kk);
            factorPanel(a, kk, kb, n, pivot, sign);

            int rest = kk + kb;
            if (rest < n) {
                applyTrsm(a, kk, kb, rest, n);
                updateTrailing(a, kk, kb, rest, n);
            }
        }

        return extractLu(a, pivot, sign[0], n);
    }

    private static void applyTrsm(float[][] a, int kk, int kb, int rest, int n) {
        int panelEnd = kk + kb;
        for (int r = kk; r < panelEnd; r++) {
            for (int p = kk; p < r; p++) {
                float factor = a[r][p];
                for (int j = rest; j < n; j++) {
                    a[r][j] -= factor * a[p][j];
                }
            }
        }
    }

    private void updateTrailing(float[][] a, int kk, int kb, int rest, int n) {
        int m = n - rest;
        if (m == 0) {
            return;
        }

        float[][] left = new float[m][kb];
        for (int i = 0; i < m; i++) {
            float[] src = a[rest + i];
            float[] dst = left[i];
            System.arraycopy(src, kk, dst, 0, kb);
        }

        float[][] right = new float[kb][m];
        for (int p = 0; p < kb; p++) {
            System.arraycopy(a[kk + p], rest, right[p], 0, m);
        }

        float[][] product = gemm.multiply(left, right, GEMM_TILE_ROWS);

        for (int i = 0; i < m; i++) {
            float[] dst = a[rest + i];
            float[] prod = product[i];
            for (int j = 0; j < m; j++) {
                dst[rest + j] -= prod[j];
            }
        }
    }

    private static void factorPanel(float[][] a, int kk, int kb, int n, int[] pivot, int[] sign) {
        int panelEnd = kk + kb;
        for (int k = kk; k < panelEnd; k++) {

            int maxRow = k;
            float maxVal = Math.abs(a[k][k]);
            for (int i = k + 1; i < n; i++) {
                float v = Math.abs(a[i][k]);
                if (v > maxVal) {
                    maxVal = v;
                    maxRow = i;
                }
            }

            if (maxRow != k) {
                float[] tmp = a[k];
                a[k] = a[maxRow];
                a[maxRow] = tmp;

                int tp = pivot[k];
                pivot[k] = pivot[maxRow];
                pivot[maxRow] = tp;

                sign[0] = -sign[0];
            }

            float diag = a[k][k];
            if (diag == 0f) {
                continue;
            }

            for (int i = k + 1; i < n; i++) {
                a[i][k] /= diag;
            }

            for (int i = k + 1; i < n; i++) {
                float factor = a[i][k];
                for (int j = k + 1; j < panelEnd; j++) {
                    a[i][j] -= factor * a[k][j];
                }
            }
        }
    }

    private static LUDecomposition extractLu(float[][] a, int[] pivot, int pivotSign, int n) {
        float[][] l = new float[n][n];
        float[][] u = new float[n][n];
        for (int i = 0; i < n; i++) {
            l[i][i] = 1f;
            System.arraycopy(a[i], 0, l[i], 0, i);
            System.arraycopy(a[i], i, u[i], i, n - i);
        }
        return new LUDecomposition(l, u, pivot, pivotSign);
    }
}
