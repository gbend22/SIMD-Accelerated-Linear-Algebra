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

    private final ScalarMatrixOps scalarOps = new ScalarMatrixOps();
    private final SimdMatrixOps simdOps = new SimdMatrixOps();

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
                scalarOps.multiply(
                        matrix,
                        vector
                );

        float[] simd =
                simdOps.multiply(
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
                scalarOps.add(a, b);

        float[][] simd =
                simdOps.add(a, b);

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
                scalarOps.multiply(a, b);

        float[][] simd =
                simdOps.multiply(a, b);

        for (int r = 0; r < size; r++) {

            assertArrayEquals(
                    scalar[r],
                    simd[r],
                    DELTA,
                    "matrix multiply mismatch at row " + r
            );
        }
    }

    @Test
    void matrixVectorMultiply_withNaN_matchesScalar() {

        float[][] matrix = {{1, Float.NaN}, {3, 4}};
        float[] vector = {1, 1};

        float[] scalar = scalarOps.multiply(matrix, vector);
        float[] simd = simdOps.multiply(matrix, vector);

        assertTrue(Float.isNaN(simd[0]), "NaN should propagate in SIMD mat-vec multiply");
        assertEquals(scalar[1], simd[1], DELTA);
    }

    @Test
    void matrixVectorMultiply_withInfinity_matchesScalar() {

        float[][] matrix = {{1, Float.POSITIVE_INFINITY}, {3, 4}};
        float[] vector = {1, 1};

        float[] scalar = scalarOps.multiply(matrix, vector);
        float[] simd = simdOps.multiply(matrix, vector);

        assertEquals(scalar[0], simd[0]);
        assertEquals(scalar[1], simd[1], DELTA);
    }

    @Test
    void add_withNaN_matchesScalar() {

        float[][] a = {{1, Float.NaN}, {3, 4}};
        float[][] b = {{10, 20}, {30, 40}};

        float[][] scalar = scalarOps.add(a, b);
        float[][] simd = simdOps.add(a, b);

        assertTrue(Float.isNaN(simd[0][1]));
        assertEquals(scalar[1][0], simd[1][0], DELTA);
    }

    @Test
    void transpose_withNaN_matchesScalar() {

        float[][] matrix = {{1, Float.NaN}, {3, 4}};

        float[][] scalar = scalarOps.transpose(matrix);
        float[][] simd = simdOps.transpose(matrix);

        assertTrue(Float.isNaN(simd[1][0]));
        assertEquals(scalar[0][0], simd[0][0], DELTA);
    }
}
