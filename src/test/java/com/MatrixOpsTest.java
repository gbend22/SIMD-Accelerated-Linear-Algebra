package com;

import com.matrix.MatrixOps;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests the public {@link MatrixOps} façade. Each method delegates through
 * {@link com.core.Dispatcher} to the active backend, so these tests also cover the
 * dispatcher's matrix delegation methods.
 */
class MatrixOpsTest {

    private static final float DELTA = 1e-3f;

    @Test
    void add() {
        float[][] a = {{1, 2}, {3, 4}};
        float[][] b = {{10, 20}, {30, 40}};

        float[][] result = MatrixOps.add(a, b);

        assertArrayEquals(new float[]{11, 22}, result[0], DELTA);
        assertArrayEquals(new float[]{33, 44}, result[1], DELTA);
    }

    @Test
    void multiplyMatrixVector() {
        float[][] matrix = {{1, 2, 3}, {4, 5, 6}};
        float[] vector = {10, 20, 30};

        assertArrayEquals(new float[]{140, 320},
                MatrixOps.multiply(matrix, vector), DELTA);
    }

    @Test
    void multiplyMatrixMatrix() {
        float[][] a = {{1, 2}, {3, 4}};
        float[][] b = {{5, 6}, {7, 8}};

        float[][] result = MatrixOps.multiply(a, b);

        assertArrayEquals(new float[]{19, 22}, result[0], DELTA);
        assertArrayEquals(new float[]{43, 50}, result[1], DELTA);
    }

    @Test
    void transpose() {
        float[][] matrix = {{1, 2, 3}, {4, 5, 6}};

        float[][] result = MatrixOps.transpose(matrix);

        assertArrayEquals(new float[]{1, 4}, result[0], DELTA);
        assertArrayEquals(new float[]{2, 5}, result[1], DELTA);
        assertArrayEquals(new float[]{3, 6}, result[2], DELTA);
    }

    @Test
    void add_mismatchedShape_throws() {
        float[][] a = {{1, 2}, {3, 4}};
        float[][] b = {{1, 2, 3}, {4, 5, 6}};

        assertThrows(IllegalArgumentException.class, () -> MatrixOps.add(a, b));
    }

    @Test
    void multiplyMatrixVector_wrongLength_throws() {
        float[][] matrix = {{1, 2, 3}, {4, 5, 6}};
        float[] vector = {1, 2};

        assertThrows(IllegalArgumentException.class, () -> MatrixOps.multiply(matrix, vector));
    }

    @Test
    void multiplyMatrixMatrix_incompatibleDimensions_throws() {
        float[][] a = {{1, 2, 3}, {4, 5, 6}};
        float[][] b = {{1, 2}, {3, 4}};

        assertThrows(IllegalArgumentException.class, () -> MatrixOps.multiply(a, b));
    }
}
