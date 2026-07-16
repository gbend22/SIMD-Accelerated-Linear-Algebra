package com.performance;

import com.decomp.CholeskyDecomposition;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class BlockedCholeskySweepTest {

    private static final float DELTA = 1e-2f;
    private static final int[] BLOCK_SIZES = {1, 2, 3, 4, 5, 8, 16, 32};

    private final BlockedCholeskySweep sweep = new BlockedCholeskySweep();

    private static float[][] symmetricPositiveDefinite(int n, long seed) {
        Random rng = new Random(seed);
        float[][] m = new float[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                float v = rng.nextFloat() * 2f - 1f;
                m[i][j] = v;
                m[j][i] = v;
            }
        }
        for (int i = 0; i < n; i++) {
            float off = 0f;
            for (int j = 0; j < n; j++) {
                if (j != i) {
                    off += Math.abs(m[i][j]);
                }
            }
            m[i][i] = off + 1f + rng.nextFloat();
        }
        return m;
    }

    private static float[][] reconstruct(float[][] l) {
        int n = l.length;
        float[][] out = new float[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float s = 0f;
                for (int k = 0; k < n; k++) {
                    s += l[i][k] * l[j][k];
                }
                out[i][j] = s;
            }
        }
        return out;
    }

    private void assertReconstructs(float[][] a, int blockSize) {
        CholeskyDecomposition chol = sweep.cholesky(a, blockSize);
        float[][] product = reconstruct(chol.getL());
        for (int i = 0; i < a.length; i++) {
            assertArrayEquals(a[i], product[i], DELTA, "blockSize=" + blockSize + " row=" + i);
        }
    }

    @Test
    void cholesky_reconstructsKnownMatrix() {
        float[][] a = {
                {4, 2, 2},
                {2, 5, 3},
                {2, 3, 6}
        };
        for (int blockSize : BLOCK_SIZES) {
            assertReconstructs(a, blockSize);
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 2, 3, 5, 8, 13, 16, 31, 32, 64, 65})
    void cholesky_reconstructs_variousSizes(int n) {
        float[][] a = symmetricPositiveDefinite(n, n * 137L);
        for (int blockSize : BLOCK_SIZES) {
            assertReconstructs(a, blockSize);
        }
    }

    @Test
    void cholesky_blockSizeLargerThanMatrix_reconstructs() {
        float[][] a = symmetricPositiveDefinite(6, 19L);
        assertReconstructs(a, 100);
    }

    @Test
    void cholesky_lowerTriangleIsFactor() {
        float[][] a = symmetricPositiveDefinite(10, 7L);
        float[][] l = sweep.cholesky(a, 4).getL();
        for (int i = 0; i < a.length; i++) {
            for (int j = i + 1; j < a.length; j++) {
                assertEquals(0f, l[i][j], 0f, "upper entry should be zero at " + i + "," + j);
            }
        }
    }

    @Test
    void cholesky_notPositiveDefinite_throws() {
        float[][] a = {{1, 2}, {2, 1}};
        assertThrows(ArithmeticException.class, () -> sweep.cholesky(a, 1));
    }

    @Test
    void cholesky_blockSizeBelowOne_throws() {
        float[][] a = {{4, 0}, {0, 4}};
        assertThrows(IllegalArgumentException.class, () -> sweep.cholesky(a, 0));
    }

    @Test
    void cholesky_nonSquare_throws() {
        float[][] a = {{1, 2, 3}, {4, 5, 6}};
        assertThrows(IllegalArgumentException.class, () -> sweep.cholesky(a, 2));
    }

    @Test
    void cholesky_empty_throws() {
        float[][] a = new float[0][0];
        assertThrows(IllegalArgumentException.class, () -> sweep.cholesky(a, 2));
    }
}
