package com;

import com.decomp.QRDecomposition;
import com.performance.BlockedQrSweep;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class BlockedQrSweepTest {

    private static final float DELTA = 1e-2f;
    private static final int[] BLOCK_SIZES = {1, 2, 3, 4, 5, 8, 16, 32};

    private final BlockedQrSweep sweep = new BlockedQrSweep();

    private static float[][] random(int m, int n, long seed) {
        Random rng = new Random(seed);
        float[][] a = new float[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a[i][j] = rng.nextFloat() * 2f - 1f;
            }
        }
        return a;
    }

    private static float[][] multiply(float[][] a, float[][] b) {
        int m = a.length;
        int inner = b.length;
        int p = b[0].length;
        float[][] out = new float[m][p];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                float s = 0f;
                for (int k = 0; k < inner; k++) {
                    s += a[i][k] * b[k][j];
                }
                out[i][j] = s;
            }
        }
        return out;
    }

    private static float[][] transpose(float[][] a) {
        int m = a.length;
        int n = a[0].length;
        float[][] out = new float[n][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                out[j][i] = a[i][j];
            }
        }
        return out;
    }

    private void assertFactors(float[][] a, int blockSize) {
        QRDecomposition qr = sweep.qr(a, blockSize);
        float[][] q = qr.getQ();
        float[][] r = qr.getR();

        float[][] product = multiply(q, r);
        for (int i = 0; i < a.length; i++) {
            assertArrayEquals(a[i], product[i], DELTA, "blockSize=" + blockSize + " row=" + i);
        }

        float[][] qtq = multiply(transpose(q), q);
        for (int i = 0; i < qtq.length; i++) {
            for (int j = 0; j < qtq.length; j++) {
                assertEquals(i == j ? 1f : 0f, qtq[i][j], DELTA,
                        "orthonormality blockSize=" + blockSize + " at " + i + "," + j);
            }
        }

        for (int j = 0; j < r[0].length; j++) {
            for (int i = j + 1; i < r.length; i++) {
                assertEquals(0f, r[i][j], 0f, "upper entry should be zero at " + i + "," + j);
            }
        }
    }

    @Test
    void qr_reconstructsKnownMatrix() {
        float[][] a = {
                {12, -51, 4},
                {6, 167, -68},
                {-4, 24, -41}
        };
        for (int blockSize : BLOCK_SIZES) {
            assertFactors(a, blockSize);
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 2, 3, 5, 8, 13, 16, 31, 32, 64, 65})
    void qr_reconstructs_squareVariousSizes(int n) {
        float[][] a = random(n, n, n * 137L);
        for (int blockSize : BLOCK_SIZES) {
            assertFactors(a, blockSize);
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 2, 3, 5, 8, 13, 16, 31, 32})
    void qr_reconstructs_tallVariousSizes(int n) {
        float[][] a = random(2 * n + 1, n, n * 251L);
        for (int blockSize : BLOCK_SIZES) {
            assertFactors(a, blockSize);
        }
    }

    @Test
    void qr_blockSizeLargerThanMatrix_reconstructs() {
        float[][] a = random(6, 4, 19L);
        assertFactors(a, 100);
    }

    @Test
    void qr_blockSizeBelowOne_throws() {
        float[][] a = {{1, 0}, {0, 1}};
        assertThrows(IllegalArgumentException.class, () -> sweep.qr(a, 0));
    }

    @Test
    void qr_wide_throws() {
        float[][] a = {{1, 2, 3}, {4, 5, 6}};
        assertThrows(IllegalArgumentException.class, () -> sweep.qr(a, 2));
    }

    @Test
    void qr_empty_throws() {
        float[][] a = new float[0][0];
        assertThrows(IllegalArgumentException.class, () -> sweep.qr(a, 2));
    }
}
