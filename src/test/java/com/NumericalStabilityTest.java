package com;

import com.core.DecompositionBackend;
import com.scalar.ScalarDecompositionOps;
import com.simd.SimdDecompositionOps;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertTrue;

/** Numerical checks based on scale-relative residuals rather than elementwise deltas. */
class NumericalStabilityTest {

    private static final double MAX_RELATIVE_RESIDUAL = 5e-5;

    static Stream<DecompositionBackend> backends() {
        return Stream.of(
                new ScalarDecompositionOps(),
                new SimdDecompositionOps(),
                SimdDecompositionOps.unblocked());
    }

    private static float[][] scaledWellConditionedMatrix(float scale) {
        float[][] base = {
                {20, 2, 1, 0, 3, 1, 0},
                {1, 22, 0, 4, 1, 0, 2},
                {0, 3, 24, 1, 0, 2, 1},
                {2, 0, 1, 26, 3, 1, 0},
                {1, 1, 0, 2, 21, 0, 3},
                {0, 2, 3, 0, 1, 23, 1},
                {3, 0, 1, 1, 0, 2, 25}
        };
        for (int i = 0; i < base.length; i++) {
            for (int j = 0; j < base.length; j++) {
                base[i][j] *= scale;
            }
        }
        return base;
    }

    private static float[] multiply(float[][] a, float[] x) {
        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            double sum = 0d;
            for (int j = 0; j < x.length; j++) {
                sum += (double) a[i][j] * x[j];
            }
            result[i] = (float) sum;
        }
        return result;
    }

    private static double relativeSolveResidual(float[][] a, float[] x, float[] b) {
        double residual = 0d;
        double normA = 0d;
        double normX = 0d;
        double normB = 0d;
        for (int i = 0; i < a.length; i++) {
            double ax = 0d;
            double rowSum = 0d;
            for (int j = 0; j < a.length; j++) {
                ax += (double) a[i][j] * x[j];
                rowSum += Math.abs(a[i][j]);
            }
            residual = Math.max(residual, Math.abs(ax - b[i]));
            normA = Math.max(normA, rowSum);
            normB = Math.max(normB, Math.abs(b[i]));
        }
        for (float value : x) {
            normX = Math.max(normX, Math.abs(value));
        }
        return residual / (normA * normX + normB);
    }

    private static double relativeInverseResidual(float[][] a, float[][] inverse) {
        int n = a.length;
        double errorNorm = 0d;
        double normA = 0d;
        double normInverse = 0d;
        for (int i = 0; i < n; i++) {
            double aRowSum = 0d;
            double inverseRowSum = 0d;
            double errorRowSum = 0d;
            for (int j = 0; j < n; j++) {
                aRowSum += Math.abs(a[i][j]);
                inverseRowSum += Math.abs(inverse[i][j]);
                double product = 0d;
                for (int k = 0; k < n; k++) {
                    product += (double) a[i][k] * inverse[k][j];
                }
                errorRowSum += Math.abs(product - (i == j ? 1d : 0d));
            }
            normA = Math.max(normA, aRowSum);
            normInverse = Math.max(normInverse, inverseRowSum);
            errorNorm = Math.max(errorNorm, errorRowSum);
        }
        return errorNorm / (normA * normInverse);
    }

    @ParameterizedTest
    @MethodSource("backends")
    void solve_hasSmallRelativeResidualAcrossScales(DecompositionBackend backend) {
        float[] expected = {1, -2, 3, -4, 5, -6, 7};
        for (float scale : new float[]{1e-3f, 1f, 1e3f}) {
            float[][] a = scaledWellConditionedMatrix(scale);
            float[] b = multiply(a, expected);
            float[] solution = backend.solve(a, b);
            double residual = relativeSolveResidual(a, solution, b);
            assertTrue(residual < MAX_RELATIVE_RESIDUAL,
                    () -> "relative solve residual was " + residual + " at scale " + scale);
        }
    }

    @ParameterizedTest
    @MethodSource("backends")
    void inverse_hasSmallRelativeResidualAcrossScales(DecompositionBackend backend) {
        for (float scale : new float[]{1e-3f, 1f, 1e3f}) {
            float[][] a = scaledWellConditionedMatrix(scale);
            double residual = relativeInverseResidual(a, backend.inverse(a));
            assertTrue(residual < MAX_RELATIVE_RESIDUAL,
                    () -> "relative inverse residual was " + residual + " at scale " + scale);
        }
    }
}
