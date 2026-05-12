package com.simd;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class SimdMatrixOps {

    private static final VectorSpecies<Float> SPECIES =
            FloatVector.SPECIES_PREFERRED;

    private static void checkSameDimensions(float[][] a, float[][] b) {

        if (a.length != b.length ||
                a[0].length != b[0].length) {

            throw new IllegalArgumentException(
                    "Matrix dimensions must match"
            );
        }
    }

    public static float[] multiplyMatrixVector(
            float[][] matrix,
            float[] vector
    ) {

        int rows = matrix.length;
        int cols = matrix[0].length;

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

                var vm = FloatVector.fromArray(
                        SPECIES,
                        matrix[r],
                        i
                );

                var vv = FloatVector.fromArray(
                        SPECIES,
                        vector,
                        i
                );

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

    public static float[][] add(
            float[][] a,
            float[][] b
    ) {

        checkSameDimensions(a, b);

        int rows = a.length;
        int cols = a[0].length;

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

    public static float[][] transpose(float[][] matrix) {

        int rows = matrix.length;
        int cols = matrix[0].length;

        float[][] result = new float[cols][rows];

        for (int i = 0; i < rows; i++) {

            for (int j = 0; j < cols; j++) {

                result[j][i] = matrix[i][j];
            }
        }

        return result;
    }

    public static float[][] multiply(
            float[][] a,
            float[][] b
    ) {

        if (a[0].length != b.length) {
            throw new IllegalArgumentException(
                    "Matrix dimensions do not allow multiplication"
            );
        }

        int rows = a.length;
        int cols = b[0].length;
        int inner = b.length;

        float[][] result = new float[rows][cols];

        float[][] bTransposed = transpose(b);

        for (int i = 0; i < rows; i++) {

            for (int j = 0; j < cols; j++) {

                int k = 0;

                int bound = SPECIES.loopBound(inner);

                var acc = FloatVector.zero(SPECIES);

                for (; k < bound; k += SPECIES.length()) {

                    var va = FloatVector.fromArray(
                            SPECIES,
                            a[i],
                            k
                    );

                    var vb = FloatVector.fromArray(
                            SPECIES,
                            bTransposed[j],
                            k
                    );

                    acc = va.fma(vb, acc);
                }

                float sum =
                        acc.reduceLanes(VectorOperators.ADD);

                for (; k < inner; k++) {
                    sum += a[i][k] * bTransposed[j][k];
                }

                result[i][j] = sum;
            }
        }

        return result;
    }

    public static int simdWidth() {
        return SPECIES.length();
    }
}