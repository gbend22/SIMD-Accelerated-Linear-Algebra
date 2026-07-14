package com.performance;

import com.decomp.LUDecomposition;

public class BlockedLuSweep {

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

    private static void updateTrailing(float[][] a, int kk, int kb, int rest, int n) {
        int panelEnd = kk + kb;
        for (int i = rest; i < n; i++) {
            for (int p = kk; p < panelEnd; p++) {
                float factor = a[i][p];
                for (int j = rest; j < n; j++) {
                    a[i][j] -= factor * a[p][j];
                }
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
