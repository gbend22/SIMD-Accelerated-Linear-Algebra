package com;

import com.scalar.ScalarMatrixOps;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ScalarMatrixOpsTest {

    private static final float DELTA = 1e-3f;

    private final ScalarMatrixOps scalar = new ScalarMatrixOps();

    @Test
    void multiplyMatrixVector_2x3() {

        float[][] matrix = {
                {1, 2, 3},
                {4, 5, 6}
        };

        float[] vector = {10, 20, 30};

        float[] expected = {
                140,
                320
        };

        assertArrayEquals(
                expected,
                scalar.multiply(matrix, vector)
        );
    }

    @Test
    void add_2x2() {

        float[][] a = {
                {1, 2},
                {3, 4}
        };

        float[][] b = {
                {10, 20},
                {30, 40}
        };

        float[][] result = scalar.add(a, b);

        assertArrayEquals(new float[]{11, 22}, result[0]);
        assertArrayEquals(new float[]{33, 44}, result[1]);
    }

    @Test
    void transpose_2x3() {

        float[][] matrix = {
                {1, 2, 3},
                {4, 5, 6}
        };

        float[][] result = scalar.transpose(matrix);

        assertArrayEquals(new float[]{1, 4}, result[0]);
        assertArrayEquals(new float[]{2, 5}, result[1]);
        assertArrayEquals(new float[]{3, 6}, result[2]);
    }

    private void assertMatrixEquals(
            float[][] expected,
            float[][] actual
    ) {

        for (int i = 0; i < expected.length; i++) {

            assertArrayEquals(
                    expected[i],
                    actual[i],
                    DELTA
            );
        }
    }

    @Test
    void multiply_smallMatrices() {

        float[][] a = {
                {1, 2},
                {3, 4}
        };

        float[][] b = {
                {5, 6},
                {7, 8}
        };

        float[][] expected = {
                {19, 22},
                {43, 50}
        };

        float[][] result =
                scalar.multiply(a, b);

        assertMatrixEquals(expected, result);
    }

    @Test
    void multiplyMatrixVector_withNaN_propagatesNaN() {

        float[][] matrix = {{1, Float.NaN}, {3, 4}};
        float[] vector = {1, 1};

        float[] result = scalar.multiply(matrix, vector);

        assertTrue(Float.isNaN(result[0]), "NaN in matrix should propagate to result row");
        assertEquals(7f, result[1], DELTA);
    }

    @Test
    void multiplyMatrixVector_withInfinity_propagatesInfinity() {

        float[][] matrix = {{1, Float.POSITIVE_INFINITY}, {3, 4}};
        float[] vector = {1, 1};

        float[] result = scalar.multiply(matrix, vector);

        assertEquals(Float.POSITIVE_INFINITY, result[0]);
    }

    @Test
    void add_withNaN_propagatesNaN() {

        float[][] a = {{1, Float.NaN}, {3, 4}};
        float[][] b = {{10, 20}, {30, 40}};

        float[][] result = scalar.add(a, b);

        assertTrue(Float.isNaN(result[0][1]));
        assertEquals(33f, result[1][0], DELTA);
    }

    @Test
    void add_withInfinity_propagatesInfinity() {

        float[][] a = {{Float.POSITIVE_INFINITY, 2}, {3, 4}};
        float[][] b = {{10, 20}, {30, 40}};

        float[][] result = scalar.add(a, b);

        assertEquals(Float.POSITIVE_INFINITY, result[0][0]);
    }

    @Test
    void transpose_withNaN_preservesNaN() {

        float[][] matrix = {{1, Float.NaN}, {3, 4}};

        float[][] result = scalar.transpose(matrix);

        assertTrue(Float.isNaN(result[1][0]));
        assertEquals(1f, result[0][0], DELTA);
    }
}
