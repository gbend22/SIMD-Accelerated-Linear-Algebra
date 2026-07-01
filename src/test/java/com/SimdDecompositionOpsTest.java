package com;

import com.decomp.LUDecomposition;
import com.simd.SimdDecompositionOps;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SimdDecompositionOpsTest {

    private static final float DELTA = 1e-6f;

    private final SimdDecompositionOps simd = new SimdDecompositionOps();

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

    private static float[][] identity(int n) {
        float[][] id = new float[n][n];
        for (int i = 0; i < n; i++) {
            id[i][i] = 1f;
        }
        return id;
    }

    private void assertReconstructs(float[][] matrix) {
        assertReconstructs(matrix, DELTA);
    }

    private void assertReconstructs(float[][] matrix, float delta) {
        LUDecomposition lu = simd.lu(matrix);
        float[][] product = multiply(lu.getL(), lu.getU());
        int[] pivot = lu.getPivot();
        for (int i = 0; i < matrix.length; i++) {
            assertArrayEquals(matrix[pivot[i]], product[i], delta);
        }
    }

    private static float[][] wellConditioned9x9() {
        float[][] a = new float[9][9];
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                a[i][j] = ((i * 7 + j * 3) % 5) - 2;
            }
            a[i][i] += 30f;
        }
        return a;
    }

    @Test
    void lu_reconstructs_3x3() {
        float[][] a = {{4, 3, 2}, {2, 1, 3}, {6, 5, 1}};
        assertReconstructs(a);
    }

    @Test
    void lu_reconstructs_9x9() {
        assertReconstructs(wellConditioned9x9(), 1e-3f);
    }

    @Test
    void lu_lIsUnitLowerTriangular() {
        float[][] a = {{4, 3, 2}, {2, 1, 3}, {6, 5, 1}};
        float[][] l = simd.lu(a).getL();
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
        float[][] u = simd.lu(a).getU();
        for (int i = 0; i < u.length; i++) {
            for (int j = 0; j < i; j++) {
                assertEquals(0f, u[i][j], DELTA);
            }
        }
    }

    @Test
    void lu_requiresRowSwap_setsNegativeSign() {
        float[][] a = {{0, 1}, {1, 0}};
        LUDecomposition lu = simd.lu(a);
        assertEquals(-1, lu.getPivotSign());
        assertArrayEquals(new int[]{1, 0}, lu.getPivot());
        assertReconstructs(a);
    }

    @Test
    void lu_nonSquare_throws() {
        float[][] a = {{1, 2, 3}, {4, 5, 6}};
        assertThrows(IllegalArgumentException.class, () -> simd.lu(a));
    }

    @Test
    void lu_empty_throws() {
        float[][] a = new float[0][0];
        assertThrows(IllegalArgumentException.class, () -> simd.lu(a));
    }

    @Test
    void solve_knownSystem() {
        float[][] a = {{3, 2, -1}, {2, -2, 4}, {-1, 0.5f, -1}};
        float[] b = {1, -2, 0};
        assertArrayEquals(new float[]{1, -2, -2}, simd.solve(a, b), 1e-3f);
    }

    @Test
    void solve_residualIsSmall_9x9() {
        float[][] a = wellConditioned9x9();
        float[] expected = {1, -2, 3, -4, 5, -6, 7, -8, 9};
        float[] b = matVec(a, expected);
        float[] x = simd.solve(a, b);
        assertArrayEquals(b, matVec(a, x), 1e-3f);
        assertArrayEquals(expected, x, 1e-2f);
    }

    @Test
    void solve_identityReturnsRhs() {
        float[][] a = identity(3);
        float[] b = {7, -3, 5};
        assertArrayEquals(b, simd.solve(a, b), DELTA);
    }

    @Test
    void solve_mismatchedRhs_throws() {
        float[][] a = {{1, 2}, {3, 4}};
        float[] b = {1, 2, 3};
        assertThrows(IllegalArgumentException.class, () -> simd.solve(a, b));
    }

    @Test
    void solve_singular_throws() {
        float[][] a = {{2, 4}, {1, 2}};
        float[] b = {1, 1};
        assertThrows(ArithmeticException.class, () -> simd.solve(a, b));
    }

    @Test
    void inverse_2x2_knownValue() {
        float[][] a = {{4, 7}, {2, 6}};
        float[][] expected = {{0.6f, -0.7f}, {-0.2f, 0.4f}};
        float[][] inv = simd.inverse(a);
        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals(expected[i], inv[i], 1e-3f);
        }
    }

    @Test
    void inverse_timesOriginal_isIdentity_9x9() {
        float[][] a = wellConditioned9x9();
        float[][] product = multiply(a, simd.inverse(a));
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a.length; j++) {
                assertEquals(i == j ? 1f : 0f, product[i][j], 1e-3f);
            }
        }
    }

    @Test
    void inverse_singular_throws() {
        float[][] a = {{2, 4}, {1, 2}};
        assertThrows(ArithmeticException.class, () -> simd.inverse(a));
    }

    @Test
    void determinant_3x3() {
        float[][] a = {{6, 1, 1}, {4, -2, 5}, {2, 8, 7}};
        assertEquals(-306f, simd.determinant(a), 1e-1f);
    }

    @Test
    void determinant_triangular_isProductOfDiagonal() {
        float[][] a = {{2, 0, 0}, {5, 3, 0}, {1, 7, 4}};
        assertEquals(24f, simd.determinant(a), 1e-3f);
    }

    @Test
    void determinant_singular_isZero() {
        float[][] a = {{2, 4}, {1, 2}};
        assertEquals(0f, simd.determinant(a), 1e-3f);
    }

    @Test
    void determinant_nonSquare_throws() {
        float[][] a = {{1, 2, 3}, {4, 5, 6}};
        assertThrows(IllegalArgumentException.class, () -> simd.determinant(a));
    }
}
