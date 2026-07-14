package com;

import com.decomp.LUDecomposition;
import com.performance.BlockedLuSweep;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class BlockedLuSweepTest {

    private static final float DELTA = 1e-3f;
    private static final int[] BLOCK_SIZES = {1, 2, 3, 4, 5, 8, 16, 32};

    private final BlockedLuSweep sweep = new BlockedLuSweep();

    private static float[][] diagonallyDominant(int n, long seed) {
        Random rng = new Random(seed);
        float[][] m = new float[n][n];
        for (int i = 0; i < n; i++) {
            float offDiagonal = 0f;
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    m[i][j] = rng.nextFloat() * 2f - 1f;
                    offDiagonal += Math.abs(m[i][j]);
                }
            }
            m[i][i] = offDiagonal + 1f + rng.nextFloat();
        }
        return m;
    }

    private static float[][] multiply(float[][] a, float[][] b) {
        int n = a.length;
        float[][] out = new float[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float s = 0f;
                for (int k = 0; k < n; k++) {
                    s += a[i][k] * b[k][j];
                }
                out[i][j] = s;
            }
        }
        return out;
    }

    private void assertReconstructsPermuted(float[][] a, int blockSize) {
        LUDecomposition lu = sweep.lu(a, blockSize);
        float[][] product = multiply(lu.getL(), lu.getU());
        int[] pivot = lu.getPivot();
        for (int i = 0; i < a.length; i++) {
            assertArrayEquals(a[pivot[i]], product[i], DELTA, "blockSize=" + blockSize + " row=" + i);
        }
    }

    @Test
    void lu_reconstructsKnownMatrix() {
        float[][] a = {
                {4, 3, 2, 1},
                {3, 5, 1, 2},
                {2, 1, 6, 3},
                {1, 2, 3, 7}
        };
        for (int blockSize : BLOCK_SIZES) {
            assertReconstructsPermuted(a, blockSize);
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 2, 3, 5, 8, 13, 16, 31, 32, 64, 65})
    void lu_reconstructsPermutedMatrix_variousSizes(int n) {
        float[][] a = diagonallyDominant(n, n * 131L);
        for (int blockSize : BLOCK_SIZES) {
            assertReconstructsPermuted(a, blockSize);
        }
    }

    @Test
    void lu_blockSizeLargerThanMatrix_reconstructs() {
        float[][] a = diagonallyDominant(6, 17L);
        assertReconstructsPermuted(a, 100);
    }

    @Test
    void lu_pivotingMatrix_reconstructs() {
        float[][] a = {
                {0, 1, 2},
                {1, 0, 3},
                {2, 3, 0}
        };
        for (int blockSize : BLOCK_SIZES) {
            assertReconstructsPermuted(a, blockSize);
        }
    }

    @Test
    void lu_blockSizeBelowOne_throws() {
        float[][] a = {{1, 2}, {3, 4}};
        assertThrows(IllegalArgumentException.class, () -> sweep.lu(a, 0));
    }

    @Test
    void lu_nonSquare_throws() {
        float[][] a = {{1, 2, 3}, {4, 5, 6}};
        assertThrows(IllegalArgumentException.class, () -> sweep.lu(a, 2));
    }

    @Test
    void lu_empty_throws() {
        float[][] a = new float[0][0];
        assertThrows(IllegalArgumentException.class, () -> sweep.lu(a, 2));
    }
}
