package com.scalar;

import com.core.DecompositionBackend;
import com.decomp.CholeskyDecomposition;
import com.decomp.LUDecomposition;
import com.decomp.QRDecomposition;

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

    @Override
    public CholeskyDecomposition cholesky(float[][] matrix) {
        checkSquare(matrix);

        int n = matrix.length;
        float[][] l = new float[n][n];

        for (int i = 0; i < n; i++) {
            float[] li = l[i];
            for (int j = 0; j <= i; j++) {
                float[] lj = l[j];
                float sum = matrix[i][j];
                for (int k = 0; k < j; k++) {
                    sum -= li[k] * lj[k];
                }
                if (i == j) {
                    if (sum <= 0f) {
                        throw new ArithmeticException("Matrix is not positive definite");
                    }
                    li[i] = (float) Math.sqrt(sum);
                } else {
                    li[j] = sum / lj[j];
                }
            }
        }

        return new CholeskyDecomposition(l);
    }

    @Override
    public QRDecomposition qr(float[][] matrix) {
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

        float[][] r = new float[m][n];
        for (int i = 0; i < m; i++) {
            r[i] = matrix[i].clone();
        }

        float[][] q = new float[m][m];
        for (int i = 0; i < m; i++) {
            q[i][i] = 1f;
        }

        float[] v = new float[m];

        for (int k = 0; k < n; k++) {
            float normx = 0f;
            for (int i = k; i < m; i++) {
                float x = r[i][k];
                normx += x * x;
            }
            normx = (float) Math.sqrt(normx);
            if (normx == 0f) {
                continue;
            }

            float rkk = r[k][k];
            float alpha = rkk >= 0f ? -normx : normx;

            for (int i = 0; i < k; i++) {
                v[i] = 0f;
            }
            v[k] = rkk - alpha;
            for (int i = k + 1; i < m; i++) {
                v[i] = r[i][k];
            }

            float vnorm2 = 0f;
            for (int i = k; i < m; i++) {
                vnorm2 += v[i] * v[i];
            }
            if (vnorm2 == 0f) {
                continue;
            }
            float beta = 2f / vnorm2;

            for (int j = k; j < n; j++) {
                float s = 0f;
                for (int i = k; i < m; i++) {
                    s += v[i] * r[i][j];
                }
                s *= beta;
                for (int i = k; i < m; i++) {
                    r[i][j] -= s * v[i];
                }
            }

            for (int i = 0; i < m; i++) {
                float s = 0f;
                for (int l = k; l < m; l++) {
                    s += q[i][l] * v[l];
                }
                s *= beta;
                for (int l = k; l < m; l++) {
                    q[i][l] -= s * v[l];
                }
            }
        }

        for (int j = 0; j < n; j++) {
            for (int i = j + 1; i < m; i++) {
                r[i][j] = 0f;
            }
        }

        return new QRDecomposition(q, r);
    }

    @Override
    public float[] solve(float[][] matrix, float[] b) {
        checkSquare(matrix);

        int n = matrix.length;
        if (b.length != n) {
            throw new IllegalArgumentException(
                    "Right-hand side length must match matrix dimension, got " + b.length + " for " + n);
        }

        LUDecomposition lu = lu(matrix);
        float[][] l = lu.getL();
        float[][] u = lu.getU();
        int[] pivot = lu.getPivot();

        float[] y = new float[n];
        for (int i = 0; i < n; i++) {
            float sum = b[pivot[i]];
            for (int j = 0; j < i; j++) {
                sum -= l[i][j] * y[j];
            }
            y[i] = sum;
        }

        float[] x = new float[n];
        for (int i = n - 1; i >= 0; i--) {
            float diag = u[i][i];
            if (diag == 0f) {
                throw new ArithmeticException("Matrix is singular; cannot solve");
            }
            float sum = y[i];
            for (int j = i + 1; j < n; j++) {
                sum -= u[i][j] * x[j];
            }
            x[i] = sum / diag;
        }

        return x;
    }

    @Override
    public float[][] inverse(float[][] matrix) {
        checkSquare(matrix);

        int n = matrix.length;

        LUDecomposition lu = lu(matrix);
        float[][] l = lu.getL();
        float[][] u = lu.getU();
        int[] pivot = lu.getPivot();

        for (int i = 0; i < n; i++) {
            if (u[i][i] == 0f) {
                throw new ArithmeticException("Matrix is singular; cannot invert");
            }
        }

        float[][] inverse = new float[n][n];

        for (int col = 0; col < n; col++) {

            float[] y = new float[n];
            for (int i = 0; i < n; i++) {
                float sum = (pivot[i] == col) ? 1f : 0f;
                for (int j = 0; j < i; j++) {
                    sum -= l[i][j] * y[j];
                }
                y[i] = sum;
            }

            for (int i = n - 1; i >= 0; i--) {
                float sum = y[i];
                for (int j = i + 1; j < n; j++) {
                    sum -= u[i][j] * inverse[j][col];
                }
                inverse[i][col] = sum / u[i][i];
            }
        }

        return inverse;
    }

    @Override
    public float determinant(float[][] matrix) {
        checkSquare(matrix);

        LUDecomposition lu = lu(matrix);
        float[][] u = lu.getU();

        float det = lu.getPivotSign();
        for (int i = 0; i < u.length; i++) {
            det *= u[i][i];
        }

        return det;
    }
}
