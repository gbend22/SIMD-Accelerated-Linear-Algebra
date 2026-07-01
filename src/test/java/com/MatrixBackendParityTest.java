package com;

import com.core.MatrixBackend;
import com.scalar.ScalarMatrixOps;
import com.simd.SimdMatrixOps;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Runs the same assertions against both {@link MatrixBackend} implementations so the
 * scalar and SIMD matrix code paths are covered identically, regardless of which backend
 * the {@link com.core.Dispatcher} selects on this machine. Includes a wide (20-column)
 * case so the SIMD main loop plus tail are both exercised. Does not touch the main code.
 */
class MatrixBackendParityTest {

    private static final float DELTA = 1e-3f;

    static Stream<MatrixBackend> backends() {
        return Stream.of(new ScalarMatrixOps(), new SimdMatrixOps());
    }

    @ParameterizedTest
    @MethodSource("backends")
    void add(MatrixBackend ops) {
        float[][] a = {{1, 2}, {3, 4}};
        float[][] b = {{10, 20}, {30, 40}};

        float[][] result = ops.add(a, b);

        assertArrayEquals(new float[]{11, 22}, result[0], DELTA);
        assertArrayEquals(new float[]{33, 44}, result[1], DELTA);
    }

    @ParameterizedTest
    @MethodSource("backends")
    void add_mismatchedShape_throws(MatrixBackend ops) {
        float[][] a = {{1, 2}, {3, 4}};
        float[][] b = {{1, 2, 3}, {4, 5, 6}};

        assertThrows(IllegalArgumentException.class, () -> ops.add(a, b));
    }

    @ParameterizedTest
    @MethodSource("backends")
    void multiplyMatrixVector(MatrixBackend ops) {
        float[][] matrix = {{1, 2, 3}, {4, 5, 6}};
        float[] vector = {10, 20, 30};

        assertArrayEquals(new float[]{140, 320}, ops.multiply(matrix, vector), DELTA);
    }

    @ParameterizedTest
    @MethodSource("backends")
    void multiplyMatrixVector_wrongLength_throws(MatrixBackend ops) {
        float[][] matrix = {{1, 2, 3}, {4, 5, 6}};
        float[] vector = {1, 2};

        assertThrows(IllegalArgumentException.class, () -> ops.multiply(matrix, vector));
    }

    @ParameterizedTest
    @MethodSource("backends")
    void multiplyMatrixMatrix(MatrixBackend ops) {
        float[][] a = {{1, 2}, {3, 4}};
        float[][] b = {{5, 6}, {7, 8}};

        float[][] result = ops.multiply(a, b);

        assertArrayEquals(new float[]{19, 22}, result[0], DELTA);
        assertArrayEquals(new float[]{43, 50}, result[1], DELTA);
    }

    @ParameterizedTest
    @MethodSource("backends")
    void multiplyMatrixMatrix_incompatible_throws(MatrixBackend ops) {
        float[][] a = {{1, 2, 3}, {4, 5, 6}};
        float[][] b = {{1, 2}, {3, 4}};

        assertThrows(IllegalArgumentException.class, () -> ops.multiply(a, b));
    }

    @ParameterizedTest
    @MethodSource("backends")
    void transpose(MatrixBackend ops) {
        float[][] matrix = {{1, 2, 3}, {4, 5, 6}};

        float[][] result = ops.transpose(matrix);

        assertArrayEquals(new float[]{1, 4}, result[0], DELTA);
        assertArrayEquals(new float[]{2, 5}, result[1], DELTA);
        assertArrayEquals(new float[]{3, 6}, result[2], DELTA);
    }

    /**
     * Multiplying by the identity returns the original matrix. The 20-wide inner
     * dimension pushes the SIMD backend through several full lane iterations plus a tail.
     */
    @ParameterizedTest
    @MethodSource("backends")
    void multiplyMatrixMatrix_wide_byIdentity(MatrixBackend ops) {
        int n = 20;
        float[][] m = new float[3][n];
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < n; c++) {
                m[r][c] = r * n + c + 1;
            }
        }
        float[][] identity = new float[n][n];
        for (int i = 0; i < n; i++) identity[i][i] = 1f;

        float[][] result = ops.multiply(m, identity);

        for (int r = 0; r < 3; r++) {
            assertArrayEquals(m[r], result[r], DELTA);
        }
    }

    @ParameterizedTest
    @MethodSource("backends")
    void multiplyMatrixVector_wide(MatrixBackend ops) {
        int n = 20;
        float[][] matrix = new float[2][n];
        float[] vector = new float[n];
        for (int c = 0; c < n; c++) {
            matrix[0][c] = 1f;
            matrix[1][c] = 2f;
            vector[c] = c + 1;
        }

        float[] result = ops.multiply(matrix, vector);

        assertEquals(210f, result[0], DELTA);
        assertEquals(420f, result[1], DELTA);
    }
}
