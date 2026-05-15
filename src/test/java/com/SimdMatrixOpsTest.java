package com;

import com.scalar.ScalarMatrixOps;
import com.simd.SimdMatrixOps;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class SimdMatrixOpsTest {

    private static final float DELTA = 1e-3f;

    private static final long SEED = 42L;

    private Random random;

    @BeforeEach
    void setUp() {
        random = new Random(SEED);
    }

    private float[][] randomMatrix(int rows, int cols) {

        float[][] matrix = new float[rows][cols];

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                matrix[r][c] =
                        (random.nextFloat() - 0.5f) * 10f;
            }
        }

        return matrix;
    }

    private float[] randomVector(int size) {

        float[] vector = new float[size];

        for (int i = 0; i < size; i++) {
            vector[i] =
                    (random.nextFloat() - 0.5f) * 10f;
        }

        return vector;
    }

    @ParameterizedTest
    @ValueSource(ints = {
            1, 3, 4, 7, 8,
            15, 16, 17,
            64, 128
    })
    void matrixVectorMultiply_matchesScalar(int size) {

        float[][] matrix = randomMatrix(size, size);

        float[] vector = randomVector(size);

        float[] scalar =
                ScalarMatrixOps.multiplyMatrixVector(
                        matrix,
                        vector
                );

        float[] simd =
                SimdMatrixOps.multiplyMatrixVector(
                        matrix,
                        vector
                );

        assertArrayEquals(
                scalar,
                simd,
                DELTA,
                "matrix-vector multiply mismatch at size " + size
        );
    }

    @ParameterizedTest
    @ValueSource(ints = {
            1, 3, 4, 7, 8,
            15, 16, 17,
            64, 128
    })
    void add_matchesScalar(int size) {

        float[][] a = randomMatrix(size, size);

        float[][] b = randomMatrix(size, size);

        float[][] scalar =
                ScalarMatrixOps.add(a, b);

        float[][] simd =
                SimdMatrixOps.add(a, b);

        for (int r = 0; r < size; r++) {

            assertArrayEquals(
                    scalar[r],
                    simd[r],
                    DELTA,
                    "matrix add mismatch at row " + r
            );
        }
    }

    @Test
    void simdWidth_isAtLeastFour() {

        assertTrue(
                SimdMatrixOps.simdWidth() >= 4,
                "Expected at least 4 SIMD lanes"
        );

        System.out.println(
                "Active SIMD width: "
                        + SimdMatrixOps.simdWidth()
                        + " float lanes"
        );
    }

    @ParameterizedTest
    @ValueSource(ints = {
            1, 3, 4, 7, 8,
            15, 16, 17,
            32, 64
    })
    void matrixMultiply_matchesScalar(int size) {

        float[][] a = randomMatrix(size, size);

        float[][] b = randomMatrix(size, size);

        float[][] scalar =
                ScalarMatrixOps.multiply(a, b);

        float[][] simd =
                SimdMatrixOps.multiply(a, b);

        for (int r = 0; r < size; r++) {

            assertArrayEquals(
                    scalar[r],
                    simd[r],
                    DELTA,
                    "matrix multiply mismatch at row " + r
            );
        }
    }
}
