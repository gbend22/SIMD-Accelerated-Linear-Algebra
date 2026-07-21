package com;

import com.core.DecompositionBackend;
import com.decomp.LUDecomposition;
import com.scalar.ScalarDecompositionOps;
import com.simd.SimdDecompositionOps;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

class DecompositionBackendParityTest {

    private static final float DELTA = 1e-3f;

    static Stream<DecompositionBackend> backends() {
        return Stream.of(new ScalarDecompositionOps(), new SimdDecompositionOps());
    }

    private static float[][] wellConditioned7x7() {
        return new float[][]{
                {20, 2, 1, 0, 3, 1, 0},
                {1, 22, 0, 4, 1, 0, 2},
                {0, 3, 24, 1, 0, 2, 1},
                {2, 0, 1, 26, 3, 1, 0},
                {1, 1, 0, 2, 21, 0, 3},
                {0, 2, 3, 0, 1, 23, 1},
                {3, 0, 1, 1, 0, 2, 25}
        };
    }

    private static float[][] symmetricPositiveDefinite7x7() {
        return new float[][]{
                {20, 2, 1, 0, 3, 1, 0},
                {2, 22, 3, 4, 1, 2, 0},
                {1, 3, 24, 1, 0, 2, 1},
                {0, 4, 1, 26, 3, 1, 1},
                {3, 1, 0, 3, 21, 0, 3},
                {1, 2, 2, 1, 0, 23, 1},
                {0, 0, 1, 1, 3, 1, 25}
        };
    }

    private static float[][] reconstructLLt(float[][] l) {
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

    private static float[][] tall5x3() {
        return new float[][]{
                {12, -1, 2},
                {3, 15, 1},
                {-2, 4, 18},
                {1, 0, 5},
                {0, 3, 2}
        };
    }

    private static float[][] multiplyRect(float[][] a, float[][] b) {
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

    @ParameterizedTest
    @MethodSource("backends")
    void lu_reconstructsPermutedMatrix(DecompositionBackend ops) {
        float[][] a = wellConditioned7x7();
        LUDecomposition lu = ops.lu(a);
        float[][] product = multiply(lu.getL(), lu.getU());
        int[] pivot = lu.getPivot();
        for (int i = 0; i < a.length; i++) {
            assertArrayEquals(a[pivot[i]], product[i], DELTA);
        }
    }

    @ParameterizedTest
    @MethodSource("backends")
    void cholesky_reconstructsMatrix(DecompositionBackend ops) {
        float[][] a = symmetricPositiveDefinite7x7();
        float[][] product = reconstructLLt(ops.cholesky(a).getL());
        for (int i = 0; i < a.length; i++) {
            assertArrayEquals(a[i], product[i], 1e-2f);
        }
    }

    @ParameterizedTest
    @MethodSource("backends")
    void cholesky_factorIsLowerTriangular(DecompositionBackend ops) {
        float[][] l = ops.cholesky(symmetricPositiveDefinite7x7()).getL();
        for (int i = 0; i < l.length; i++) {
            for (int j = i + 1; j < l.length; j++) {
                assertEquals(0f, l[i][j], 0f);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("backends")
    void cholesky_notPositiveDefinite_throws(DecompositionBackend ops) {
        float[][] a = {{1, 2}, {2, 1}};
        assertThrows(ArithmeticException.class, () -> ops.cholesky(a));
    }

    @ParameterizedTest
    @MethodSource("backends")
    void qr_reconstructsMatrix(DecompositionBackend ops) {
        float[][] a = tall5x3();
        var qr = ops.qr(a);
        float[][] product = multiplyRect(qr.getQ(), qr.getR());
        for (int i = 0; i < a.length; i++) {
            assertArrayEquals(a[i], product[i], 1e-2f);
        }
    }

    @ParameterizedTest
    @MethodSource("backends")
    void qr_squareReconstructsMatrix(DecompositionBackend ops) {
        float[][] a = wellConditioned7x7();
        var qr = ops.qr(a);
        float[][] product = multiplyRect(qr.getQ(), qr.getR());
        for (int i = 0; i < a.length; i++) {
            assertArrayEquals(a[i], product[i], 1e-2f);
        }
    }

    @ParameterizedTest
    @MethodSource("backends")
    void qr_qIsOrthonormal(DecompositionBackend ops) {
        float[][] q = ops.qr(tall5x3()).getQ();
        float[][] qtq = multiplyRect(transpose(q), q);
        for (int i = 0; i < qtq.length; i++) {
            for (int j = 0; j < qtq.length; j++) {
                assertEquals(i == j ? 1f : 0f, qtq[i][j], 1e-2f);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("backends")
    void qr_rIsUpperTriangular(DecompositionBackend ops) {
        float[][] r = ops.qr(tall5x3()).getR();
        for (int j = 0; j < r[0].length; j++) {
            for (int i = j + 1; i < r.length; i++) {
                assertEquals(0f, r[i][j], 0f);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("backends")
    void solve_residualIsSmall(DecompositionBackend ops) {
        float[][] a = wellConditioned7x7();
        float[] expected = {1, -2, 3, -4, 5, -6, 7};
        float[] b = matVec(a, expected);
        float[] x = ops.solve(a, b);
        assertArrayEquals(b, matVec(a, x), DELTA);
        assertArrayEquals(expected, x, 1e-2f);
    }

    @ParameterizedTest
    @MethodSource("backends")
    void solve_singular_throws(DecompositionBackend ops) {
        float[][] a = {{2, 4}, {1, 2}};
        assertThrows(ArithmeticException.class, () -> ops.solve(a, new float[]{1, 1}));
    }

    @ParameterizedTest
    @MethodSource("backends")
    void inverse_timesOriginal_isIdentity(DecompositionBackend ops) {
        float[][] a = wellConditioned7x7();
        float[][] product = multiply(a, ops.inverse(a));
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a.length; j++) {
                assertEquals(i == j ? 1f : 0f, product[i][j], DELTA);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("backends")
    void inverse_singular_throws(DecompositionBackend ops) {
        float[][] a = {{2, 4}, {1, 2}};
        assertThrows(ArithmeticException.class, () -> ops.inverse(a));
    }

    @ParameterizedTest
    @MethodSource("backends")
    void determinant_triangular_isProductOfDiagonal(DecompositionBackend ops) {
        float[][] a = {{2, 0, 0}, {5, 3, 0}, {1, 7, 4}};
        assertEquals(24f, ops.determinant(a), DELTA);
    }

    @ParameterizedTest
    @MethodSource("backends")
    void determinant_singular_isZero(DecompositionBackend ops) {
        float[][] a = {{2, 4}, {1, 2}};
        assertEquals(0f, ops.determinant(a), DELTA);
    }

    @ParameterizedTest
    @MethodSource("backends")
    void nonSquare_throws(DecompositionBackend ops) {
        float[][] a = {{1, 2, 3}, {4, 5, 6}};
        assertThrows(IllegalArgumentException.class, () -> ops.lu(a));
        assertThrows(IllegalArgumentException.class, () -> ops.cholesky(a));
        assertThrows(IllegalArgumentException.class, () -> ops.qr(a));
        assertThrows(IllegalArgumentException.class, () -> ops.solve(a, new float[]{1, 2}));
        assertThrows(IllegalArgumentException.class, () -> ops.inverse(a));
        assertThrows(IllegalArgumentException.class, () -> ops.determinant(a));
    }
}
