package com;

import com.decomp.CholeskyDecomposition;
import com.decomp.LUDecomposition;
import com.decomp.QRDecomposition;
import com.matrix.MatrixOps;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests the public {@link MatrixOps} façade. Each method delegates through
 * {@link com.core.Dispatcher} to the active backend, so these tests also cover the
 * dispatcher's matrix and decomposition delegation methods and the accessors on the
 * decomposition result holders.
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

    @Test
    void lu() {
        float[][] a = {{4, 3}, {6, 3}};

        LUDecomposition lu = MatrixOps.lu(a);
        assertEquals(2, lu.size());

        float[][] l = lu.getL();
        float[][] u = lu.getU();
        int[] pivot = lu.getPivot();

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                float sum = 0f;
                for (int k = 0; k < 2; k++) {
                    sum += l[i][k] * u[k][j];
                }
                assertEquals(a[pivot[i]][j], sum, DELTA);
            }
        }
    }

    @Test
    void cholesky() {
        float[][] a = {{4, 2}, {2, 3}};

        CholeskyDecomposition chol = MatrixOps.cholesky(a);
        assertEquals(2, chol.size());

        float[][] l = chol.getL();

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                float sum = 0f;
                for (int k = 0; k < 2; k++) {
                    sum += l[i][k] * l[j][k];
                }
                assertEquals(a[i][j], sum, DELTA);
            }
        }
    }

    @Test
    void qr() {
        float[][] a = {{1, 2}, {3, 4}, {5, 6}};

        QRDecomposition qr = MatrixOps.qr(a);
        assertEquals(3, qr.rows());
        assertEquals(2, qr.columns());

        float[][] q = qr.getQ();
        float[][] r = qr.getR();

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                float sum = 0f;
                for (int k = 0; k < 3; k++) {
                    sum += q[i][k] * r[k][j];
                }
                assertEquals(a[i][j], sum, DELTA);
            }
        }
    }

    @Test
    void solve() {
        float[][] a = {{3, 2}, {1, 2}};
        float[] b = {7, 5};

        assertArrayEquals(new float[]{1, 2}, MatrixOps.solve(a, b), DELTA);
    }

    @Test
    void determinant() {
        float[][] a = {{4, 3}, {6, 3}};

        assertEquals(-6f, MatrixOps.determinant(a), DELTA);
    }

    @Test
    void inverse() {
        float[][] a = {{4, 3}, {6, 3}};

        float[][] inv = MatrixOps.inverse(a);

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                float sum = 0f;
                for (int k = 0; k < 2; k++) {
                    sum += a[i][k] * inv[k][j];
                }
                assertEquals(i == j ? 1f : 0f, sum, DELTA);
            }
        }
    }
}
