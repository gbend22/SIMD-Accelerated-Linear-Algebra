package com.baseline;

import com.core.MatrixValidation;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Deliberately simple single-threaded SIMD GEMM baseline. It transposes {@code B} once,
 * then computes each output element as one vectorized dot product. It performs no
 * register tiling, cache blocking, or parallel execution.
 */
public final class NaiveSimdMatrixOps {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    public float[][] multiply(float[][] a, float[][] b) {
        int inner = MatrixValidation.requireRectangular(a, "a");
        int columns = MatrixValidation.requireRectangular(b, "b");
        if (inner != b.length) {
            throw new IllegalArgumentException("Matrix dimensions do not allow multiplication");
        }

        int rows = a.length;
        int vectorBound = SPECIES.loopBound(inner);
        float[][] transposedB = transpose(b, inner, columns);
        float[][] result = new float[rows][columns];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                FloatVector accumulator = FloatVector.zero(SPECIES);
                int k = 0;
                for (; k < vectorBound; k += SPECIES.length()) {
                    FloatVector left = FloatVector.fromArray(SPECIES, a[i], k);
                    FloatVector right = FloatVector.fromArray(SPECIES, transposedB[j], k);
                    accumulator = left.fma(right, accumulator);
                }
                float sum = accumulator.reduceLanes(VectorOperators.ADD);
                for (; k < inner; k++) {
                    sum += a[i][k] * transposedB[j][k];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }

    private static float[][] transpose(float[][] matrix, int rows, int columns) {
        float[][] result = new float[columns][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }
}
