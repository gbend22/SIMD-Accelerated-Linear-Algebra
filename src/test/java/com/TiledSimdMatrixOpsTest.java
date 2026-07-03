package com;

import com.performance.TiledSimdMatrixOps;
import com.scalar.ScalarMatrixOps;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class TiledSimdMatrixOpsTest {

    private static final float DELTA = 1e-3f;

    private final TiledSimdMatrixOps tiled = new TiledSimdMatrixOps();
    private final ScalarMatrixOps scalar = new ScalarMatrixOps();

    private static float[][] random(int rows, int cols, long seed) {
        Random rng = new Random(seed);
        float[][] m = new float[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                m[r][c] = rng.nextFloat() * 2f - 1f;
            }
        }
        return m;
    }

    @Test
    void multiply_knownValue() {
        float[][] a = {{1, 2}, {3, 4}};
        float[][] b = {{5, 6}, {7, 8}};
        float[][] expected = {{19, 22}, {43, 50}};
        float[][] result = tiled.multiply(a, b);
        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals(expected[i], result[i], DELTA);
        }
    }

    @Test
    void multiply_identityReturnsOriginal() {
        float[][] a = random(6, 6, 1);
        float[][] id = new float[6][6];
        for (int i = 0; i < 6; i++) {
            id[i][i] = 1f;
        }
        float[][] result = tiled.multiply(a, id);
        for (int i = 0; i < a.length; i++) {
            assertArrayEquals(a[i], result[i], DELTA);
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 2, 7, 16, 33, 64, 65, 100, 129})
    void multiply_matchesScalar_square(int n) {
        float[][] a = random(n, n, n * 31L);
        float[][] b = random(n, n, n * 97L);
        float[][] expected = scalar.multiply(a, b);
        float[][] result = tiled.multiply(a, b);
        for (int i = 0; i < n; i++) {
            assertArrayEquals(expected[i], result[i], DELTA);
        }
    }

    @Test
    void multiply_matchesScalar_rectangular() {
        float[][] a = random(70, 45, 7);
        float[][] b = random(45, 90, 11);
        float[][] expected = scalar.multiply(a, b);
        float[][] result = tiled.multiply(a, b);
        for (int i = 0; i < a.length; i++) {
            assertArrayEquals(expected[i], result[i], DELTA);
        }
    }

    @Test
    void multiply_mismatchedDimensions_throws() {
        float[][] a = {{1, 2, 3}, {4, 5, 6}};
        float[][] b = {{1, 2}, {3, 4}};
        assertThrows(IllegalArgumentException.class, () -> tiled.multiply(a, b));
    }
}
