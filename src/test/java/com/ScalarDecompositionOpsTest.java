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

    private static float[] matVec(float[][] a, float[] x) {
        float[] out = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            float s = 0f;
            for (int j = 0; j < x.length; j++) {
                s += a[i][j] * x[j];
            }
            out[i] = s;
        }
        return out;
    }

    @Test
    void solve_knownSystem() {
        float[][] a = {{3, 2, -1}, {2, -2, 4}, {-1, 0.5f, -1}};
        float[] b = {1, -2, 0};
        assertArrayEquals(new float[]{1, -2, -2}, scalar.solve(a, b), 1e-3f);
    }

    @Test
    void solve_residualIsSmall_5x5() {
        float[][] a = {
                {10, 2, 0, 1, 3},
                {3, 12, 1, 0, 2},
                {1, 4, 14, 2, 0},
                {0, 1, 3, 11, 4},
                {2, 0, 1, 5, 13}
        };
        float[] expected = {1, -2, 3, -4, 5};
        float[] b = matVec(a, expected);
        float[] x = scalar.solve(a, b);
        assertArrayEquals(b, matVec(a, x), 1e-3f);
        assertArrayEquals(expected, x, 1e-2f);
    }

    @Test
    void solve_identityReturnsRhs() {
        float[][] a = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        float[] b = {7, -3, 5};
        assertArrayEquals(b, scalar.solve(a, b), DELTA);
    }

    @Test
    void solve_mismatchedRhs_throws() {
        float[][] a = {{1, 2}, {3, 4}};
        float[] b = {1, 2, 3};
        assertThrows(IllegalArgumentException.class, () -> scalar.solve(a, b));
    }

    @Test
    void solve_singular_throws() {
        float[][] a = {{2, 4}, {1, 2}};
        float[] b = {1, 1};
        assertThrows(ArithmeticException.class, () -> scalar.solve(a, b));
    }

    @Test
    void solve_nonSquare_throws() {
        float[][] a = {{1, 2, 3}, {4, 5, 6}};
        float[] b = {1, 2};
        assertThrows(IllegalArgumentException.class, () -> scalar.solve(a, b));
    }

    @Test
    void determinant_2x2() {
        float[][] a = {{4, 3}, {6, 3}};
        assertEquals(-6f, scalar.determinant(a), 1e-3f);
    }

    @Test
    void determinant_3x3() {
        float[][] a = {{6, 1, 1}, {4, -2, 5}, {2, 8, 7}};
        assertEquals(-306f, scalar.determinant(a), 1e-1f);
    }

    @Test
    void determinant_triangular_isProductOfDiagonal() {
        float[][] a = {{2, 0, 0}, {5, 3, 0}, {1, 7, 4}};
        assertEquals(24f, scalar.determinant(a), 1e-3f);
    }

    @Test
    void determinant_identity_isOne() {
        float[][] a = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        assertEquals(1f, scalar.determinant(a), DELTA);
    }

    @Test
    void determinant_singular_isZero() {
        float[][] a = {{2, 4}, {1, 2}};
        assertEquals(0f, scalar.determinant(a), 1e-3f);
    }

    @Test
    void determinant_nonSquare_throws() {
        float[][] a = {{1, 2, 3}, {4, 5, 6}};
        assertThrows(IllegalArgumentException.class, () -> scalar.determinant(a));
    }

    private static float[][] identity(int n) {
        float[][] id = new float[n][n];
        for (int i = 0; i < n; i++) {
            id[i][i] = 1f;
        }
        return id;
    }

    private void assertIsInverse(float[][] a) {
        float[][] inv = scalar.inverse(a);
        float[][] product = multiply(a, inv);
        float[][] id = identity(a.length);
        for (int i = 0; i < a.length; i++) {
            assertArrayEquals(id[i], product[i], 1e-3f);
        }
    }

    @Test
    void inverse_2x2_knownValue() {
        float[][] a = {{4, 7}, {2, 6}};
        float[][] expected = {{0.6f, -0.7f}, {-0.2f, 0.4f}};
        float[][] inv = scalar.inverse(a);
        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals(expected[i], inv[i], 1e-3f);
        }
    }

    @Test
    void inverse_timesOriginal_isIdentity_3x3() {
        float[][] a = {{2, 1, 1}, {1, 3, 2}, {1, 0, 0}};
        assertIsInverse(a);
    }

    @Test
    void inverse_timesOriginal_isIdentity_5x5() {
        float[][] a = {
                {10, 2, 0, 1, 3},
                {3, 12, 1, 0, 2},
                {1, 4, 14, 2, 0},
                {0, 1, 3, 11, 4},
                {2, 0, 1, 5, 13}
        };
        assertIsInverse(a);
    }

    @Test
    void inverse_requiresRowSwap() {
        float[][] a = {{0, 1}, {1, 0}};
        assertIsInverse(a);
    }

    @Test
    void inverse_identity_isIdentity() {
        float[][] a = identity(4);
        float[][] inv = scalar.inverse(a);
        for (int i = 0; i < a.length; i++) {
            assertArrayEquals(a[i], inv[i], DELTA);
        }
    }

    @Test
    void inverse_singular_throws() {
        float[][] a = {{2, 4}, {1, 2}};
        assertThrows(ArithmeticException.class, () -> scalar.inverse(a));
    }

    @Test
    void inverse_nonSquare_throws() {
        float[][] a = {{1, 2, 3}, {4, 5, 6}};
        assertThrows(IllegalArgumentException.class, () -> scalar.inverse(a));
    }
}
