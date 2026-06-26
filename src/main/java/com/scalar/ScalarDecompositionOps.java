package com.scalar;

import com.core.DecompositionBackend;
import com.decomp.LUDecomposition;

public class ScalarDecompositionOps implements DecompositionBackend {

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

    @Override
    public LUDecomposition lu(float[][] matrix) {
        checkSquare(matrix);

        int n = matrix.length;

        float[][] a = new float[n][n];
        for (int r = 0; r < n; r++) {
            a[r] = matrix[r].clone();
        }

        int[] pivot = new int[n];
        for (int i = 0; i < n; i++) {
            pivot[i] = i;
        }
        int pivotSign = 1;

        for (int k = 0; k < n; k++) {

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

                pivotSign = -pivotSign;
            }

            float diag = a[k][k];
            if (diag == 0f) {
                continue;
            }

            for (int i = k + 1; i < n; i++) {
                float factor = a[i][k] / diag;
                a[i][k] = factor;
                for (int j = k + 1; j < n; j++) {
                    a[i][j] -= factor * a[k][j];
                }
            }
        }

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
