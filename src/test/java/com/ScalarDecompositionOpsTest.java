package com;

import com.decomp.LUDecomposition;
import com.scalar.ScalarDecompositionOps;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ScalarDecompositionOpsTest {

    private static final float DELTA = 1e-6f;

    private final ScalarDecompositionOps scalar = new ScalarDecompositionOps();

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

    private void assertReconstructs(float[][] matrix) {
        LUDecomposition lu = scalar.lu(matrix);
        float[][] product = multiply(lu.getL(), lu.getU());
        int[] pivot = lu.getPivot();
        for (int i = 0; i < matrix.length; i++) {
            assertArrayEquals(matrix[pivot[i]], product[i], DELTA);
        }
    }

    @Test
    void lu_reconstructs_3x3() {
        float[][] a = {{4, 3, 2}, {2, 1, 3}, {6, 5, 1}};
        assertReconstructs(a);
    }

    @Test
    void lu_lIsUnitLowerTriangular() {
        float[][] a = {{4, 3, 2}, {2, 1, 3}, {6, 5, 1}};
        float[][] l = scalar.lu(a).getL();
        for (int i = 0; i < l.length; i++) {
            assertEquals(1f, l[i][i], DELTA);
            for (int j = i + 1; j < l.length; j++) {
                assertEquals(0f, l[i][j], DELTA);
            }
        }
    }

    @Test
    void lu_uIsUpperTriangular() {
        float[][] a = {{4, 3, 2}, {2, 1, 3}, {6, 5, 1}};
        float[][] u = scalar.lu(a).getU();
        for (int i = 0; i < u.length; i++) {
            for (int j = 0; j < i; j++) {
                assertEquals(0f, u[i][j], DELTA);
            }
        }
    }

    @Test
    void lu_identity() {
        float[][] a = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        LUDecomposition lu = scalar.lu(a);
        assertArrayEquals(new int[]{0, 1, 2}, lu.getPivot());
        assertEquals(1, lu.getPivotSign());
        assertReconstructs(a);
    }

    @Test
    void lu_requiresRowSwap_setsNegativeSign() {
        float[][] a = {{0, 1}, {1, 0}};
        LUDecomposition lu = scalar.lu(a);
        assertEquals(-1, lu.getPivotSign());
        assertArrayEquals(new int[]{1, 0}, lu.getPivot());
        assertReconstructs(a);
    }

    @Test
    void lu_reconstructs_5x5() {
        float[][] a = {
                {10, 2, 0, 1, 3},
                {3, 12, 1, 0, 2},
                {1, 4, 14, 2, 0},
                {0, 1, 3, 11, 4},
                {2, 0, 1, 5, 13}
        };
        assertReconstructs(a);
    }

    @Test
    void lu_singularMatrix_stillReconstructs() {
        float[][] a = {{2, 4}, {1, 2}};
        assertReconstructs(a);
    }

    @Test
    void lu_nonSquare_throws() {
        float[][] a = {{1, 2, 3}, {4, 5, 6}};
        assertThrows(IllegalArgumentException.class, () -> scalar.lu(a));
    }

    @Test
    void lu_empty_throws() {
        float[][] a = new float[0][0];
        assertThrows(IllegalArgumentException.class, () -> scalar.lu(a));
    }
}
