package com.simd;

import com.core.MatrixBackend;
import com.core.MatrixValidation;
import com.performance.RegisterTileSweepMatrixOps;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD implementation of {@link com.core.MatrixBackend}, built on the Java Vector API
 * ({@code jdk.incubator.vector}). Selected at runtime when a usable vector width is
 * available; internal, not part of the public API.
 */
public class SimdMatrixOps implements MatrixBackend {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private static final int MULTIPLY_MR = 8;

    private final RegisterTileSweepMatrixOps registerTile = new RegisterTileSweepMatrixOps();

    @Override
    public float[] multiply(float[][] matrix, float[] vector) {
        int rows = matrix.length;
        int cols = MatrixValidation.requireRectangular(matrix, "matrix");

        if (vector.length != cols) {
            throw new IllegalArgumentException(
                    "Vector length must match matrix columns"
            );
        }

        float[] result = new float[rows];

        for (int r = 0; r < rows; r++) {

            int i = 0;
            int bound = SPECIES.loopBound(cols);

            var acc = FloatVector.zero(SPECIES);

            for (; i < bound; i += SPECIES.length()) {
                var vm = FloatVector.fromArray(SPECIES, matrix[r], i);
                var vv = FloatVector.fromArray(SPECIES, vector, i);
                acc = vm.fma(vv, acc);
            }

            float sum = acc.reduceLanes(VectorOperators.ADD);

            for (; i < cols; i++) {
                sum += matrix[r][i] * vector[i];
            }

            result[r] = sum;
        }

        return result;
    }

    @Override
    public float[][] add(float[][] a, float[][] b) {
        int rows = a.length;
        int cols = MatrixValidation.requireSameShape(a, b);

        float[][] result = new float[rows][cols];

        for (int r = 0; r < rows; r++) {

            int i = 0;
            int bound = SPECIES.loopBound(cols);

            for (; i < bound; i += SPECIES.length()) {

                FloatVector.fromArray(SPECIES, a[r], i)
                        .add(FloatVector.fromArray(SPECIES, b[r], i))
                        .intoArray(result[r], i);
            }

            for (; i < cols; i++) {
                result[r][i] = a[r][i] + b[r][i];
            }
        }

        return result;
    }

    @Override
    public float[][] multiply(float[][] a, float[][] b) {
        int inner = MatrixValidation.requireRectangular(a, "a");
        MatrixValidation.requireRectangular(b, "b");
        if (inner != b.length) {
            throw new IllegalArgumentException("Matrix dimensions do not allow multiplication");
        }
        return registerTile.multiply(a, b, MULTIPLY_MR);
    }

    @Override
    public float[][] transpose(float[][] matrix) {
        int rows = matrix.length;
        int cols = MatrixValidation.requireRectangular(matrix, "matrix");

        float[][] result = new float[cols][rows];

        final int BLOCK = 32;

        for (int ii = 0; ii < rows; ii += BLOCK) {

            for (int jj = 0; jj < cols; jj += BLOCK) {

                int iMax = Math.min(ii + BLOCK, rows);
                int jMax = Math.min(jj + BLOCK, cols);

                for (int i = ii; i < iMax; i++) {

                    int j = jj;

                    int bound = SPECIES.loopBound(jMax - jj) + jj;

                    for (; j < bound; j += SPECIES.length()) {
                        FloatVector vec = FloatVector.fromArray(SPECIES, matrix[i], j);
                        float[] temp = new float[SPECIES.length()];
                        vec.intoArray(temp, 0);

                        for (int k = 0; k < SPECIES.length(); k++) {
                            result[j + k][i] = temp[k];
                        }
                    }

                    for (; j < jMax; j++) {
                        result[j][i] = matrix[i][j];
                    }
                }
            }
        }

        return result;
    }

    public static int simdWidth() {
        return SPECIES.length();
    }
}
