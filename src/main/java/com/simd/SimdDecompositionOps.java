package com.simd;

import com.core.DecompositionBackend;
import com.decomp.CholeskyDecomposition;
import com.decomp.LUDecomposition;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class SimdDecompositionOps implements DecompositionBackend {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

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

            int start = k + 1;
            int bound = SPECIES.loopBound(n - start) + start;

            for (int i = start; i < n; i++) {
                float factor = a[i][k] / diag;
                a[i][k] = factor;

                FloatVector negFactor = FloatVector.broadcast(SPECIES, -factor);

                int j = start;
                for (; j < bound; j += SPECIES.length()) {
                    FloatVector vk = FloatVector.fromArray(SPECIES, a[k], j);
                    FloatVector vi = FloatVector.fromArray(SPECIES, a[i], j);
                    vk.fma(negFactor, vi).intoArray(a[i], j);
                }
                for (; j < n; j++) {
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

                int bound = SPECIES.loopBound(j);
                FloatVector acc = FloatVector.zero(SPECIES);

                int k = 0;
                for (; k < bound; k += SPECIES.length()) {
                    FloatVector vi = FloatVector.fromArray(SPECIES, li, k);
                    FloatVector vj = FloatVector.fromArray(SPECIES, lj, k);
                    acc = vi.fma(vj, acc);
                }

                float sum = matrix[i][j] - acc.reduceLanes(VectorOperators.ADD);
                for (; k < j; k++) {
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

    private static float[] forwardSubstitution(float[][] l, float[] rhs, int n) {
        float[] y = new float[n];
        for (int i = 0; i < n; i++) {
            int bound = SPECIES.loopBound(i);
            FloatVector acc = FloatVector.zero(SPECIES);

            int j = 0;
            for (; j < bound; j += SPECIES.length()) {
                FloatVector vl = FloatVector.fromArray(SPECIES, l[i], j);
                FloatVector vy = FloatVector.fromArray(SPECIES, y, j);
                acc = vl.fma(vy, acc);
            }

            float sum = rhs[i] - acc.reduceLanes(VectorOperators.ADD);
            for (; j < i; j++) {
                sum -= l[i][j] * y[j];
            }
            y[i] = sum;
        }
        return y;
    }

    private static float[] backSubstitution(float[][] u, float[] y, int n) {
        float[] x = new float[n];
        for (int i = n - 1; i >= 0; i--) {
            float diag = u[i][i];
            if (diag == 0f) {
                throw new ArithmeticException("Matrix is singular; cannot solve");
            }

            int start = i + 1;
            int bound = SPECIES.loopBound(n - start) + start;
            FloatVector acc = FloatVector.zero(SPECIES);

            int j = start;
            for (; j < bound; j += SPECIES.length()) {
                FloatVector vu = FloatVector.fromArray(SPECIES, u[i], j);
                FloatVector vx = FloatVector.fromArray(SPECIES, x, j);
                acc = vu.fma(vx, acc);
            }

            float sum = y[i] - acc.reduceLanes(VectorOperators.ADD);
            for (; j < n; j++) {
                sum -= u[i][j] * x[j];
            }
            x[i] = sum / diag;
        }
        return x;
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

        float[] permuted = new float[n];
        for (int i = 0; i < n; i++) {
            permuted[i] = b[pivot[i]];
        }

        float[] y = forwardSubstitution(l, permuted, n);
        return backSubstitution(u, y, n);
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
            float[] e = new float[n];
            for (int i = 0; i < n; i++) {
                e[i] = (pivot[i] == col) ? 1f : 0f;
            }

            float[] y = forwardSubstitution(l, e, n);
            float[] x = backSubstitution(u, y, n);

            for (int i = 0; i < n; i++) {
                inverse[i][col] = x[i];
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
