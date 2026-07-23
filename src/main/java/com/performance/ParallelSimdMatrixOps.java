package com.performance;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.stream.IntStream;

/**
 * Row-parallel SIMD matrix multiply. {@code B} is transposed once so every output element
 * is a contiguous vectorized dot product, and the rows of the result are computed across
 * threads with a parallel stream. One of several matrix-multiply strategies explored in
 * {@code com.performance}; internal, not part of the public API.
 */
public class ParallelSimdMatrixOps {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private static float[][] transpose(float[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        float[][] result = new float[cols][rows];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                result[c][r] = matrix[r][c];
            }
        }
        return result;
    }

    private static void multiplyRow(float[][] a, float[][] bt, float[][] result, int i, int cols, int inner) {
        int bound = SPECIES.loopBound(inner);
        for (int j = 0; j < cols; j++) {
            FloatVector acc = FloatVector.zero(SPECIES);
            int k = 0;
            for (; k < bound; k += SPECIES.length()) {
                FloatVector va = FloatVector.fromArray(SPECIES, a[i], k);
                FloatVector vb = FloatVector.fromArray(SPECIES, bt[j], k);
                acc = va.fma(vb, acc);
            }
            float sum = acc.reduceLanes(VectorOperators.ADD);
            for (; k < inner; k++) {
                sum += a[i][k] * bt[j][k];
            }
            result[i][j] = sum;
        }
    }

    public float[][] multiply(float[][] a, float[][] b) {
        if (a[0].length != b.length) {
            throw new IllegalArgumentException("Matrix dimensions do not allow multiplication");
        }

        int rows = a.length;
        int cols = b[0].length;
        int inner = b.length;

        float[][] bt = transpose(b);
        float[][] result = new float[rows][cols];

        IntStream.range(0, rows)
                .parallel()
                .forEach(i -> multiplyRow(a, bt, result, i, cols, inner));

        return result;
    }
}
