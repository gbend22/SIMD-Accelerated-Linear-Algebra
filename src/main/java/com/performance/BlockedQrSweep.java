package com.performance;

import com.decomp.QRDecomposition;

public class BlockedQrSweep {

    private static void checkShape(float[][] matrix) {
        int m = matrix.length;
        if (m == 0) {
            throw new IllegalArgumentException("Matrix must not be empty");
        }
        int n = matrix[0].length;
        for (float[] row : matrix) {
            if (row.length != n) {
                throw new IllegalArgumentException(
                        "Matrix must be rectangular, got a row of length " + row.length + " expected " + n);
            }
        }
        if (m < n) {
            throw new IllegalArgumentException("QR requires rows >= columns, got " + m + "x" + n);
        }
    }

    public QRDecomposition qr(float[][] matrix, int blockSize) {
        checkShape(matrix);
        if (blockSize < 1) {
            throw new IllegalArgumentException("blockSize must be at least 1, got " + blockSize);
        }

        int m = matrix.length;
        int n = matrix[0].length;

        float[][] a = new float[m][n];
        for (int i = 0; i < m; i++) {
            a[i] = matrix[i].clone();
        }

        float[] tau = new float[n];

        for (int kk = 0; kk < n; kk += blockSize) {
            int panelEnd = Math.min(kk + blockSize, n);

            for (int k = kk; k < panelEnd; k++) {
                computeReflector(a, k, m, tau);
                for (int j = k + 1; j < panelEnd; j++) {
                    applyReflectorToColumn(a, k, j, m, tau[k]);
                }
            }

            for (int k = kk; k < panelEnd; k++) {
                for (int j = panelEnd; j < n; j++) {
                    applyReflectorToColumn(a, k, j, m, tau[k]);
                }
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
        for (int k = n - 1; k >= 0; k--) {
            if (tau[k] == 0f) {
                continue;
            }
            for (int j = 0; j < m; j++) {
                float s = q[k][j];
                for (int i = k + 1; i < m; i++) {
                    s += a[i][k] * q[i][j];
                }
                s *= tau[k];
                q[k][j] -= s;
                for (int i = k + 1; i < m; i++) {
                    q[i][j] -= s * a[i][k];
                }
            }
        }

        return new QRDecomposition(q, r);
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
