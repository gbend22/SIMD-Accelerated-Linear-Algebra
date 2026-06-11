package com;

import com.core.VectorBackend;
import com.scalar.ScalarVectorOps;
import com.simd.SimdVectorOps;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Runs the same assertions against both {@link VectorBackend} implementations so the
 * scalar and SIMD code paths are covered identically, independent of which backend the
 * {@link com.core.Dispatcher} happens to select on this machine. Test vectors use a
 * length that is not a multiple of the SIMD lane width so the SIMD tail loops are also
 * exercised. Does not touch the main code.
 */
class VectorBackendParityTest {

    private static final float DELTA = 1e-4f;

    static Stream<VectorBackend> backends() {
        return Stream.of(new ScalarVectorOps(), new SimdVectorOps());
    }

    // A 19-element vector: longer than a typical lane width and not a multiple of it.
    private static float[] rampA() {
        float[] a = new float[19];
        for (int i = 0; i < a.length; i++) a[i] = i + 1;       // 1..19
        return a;
    }

    private static float[] rampB() {
        float[] b = new float[19];
        for (int i = 0; i < b.length; i++) b[i] = 2f * (i + 1); // 2,4,..38
        return b;
    }

    @ParameterizedTest
    @MethodSource("backends")
    void dot(VectorBackend ops) {
        // sum of 2*(i+1)^2 for i=1..19 = 2 * (19*20*39/6) = 2 * 2470 = 4940
        assertEquals(4940f, ops.dot(rampA(), rampB()), DELTA);
    }

    @ParameterizedTest
    @MethodSource("backends")
    void norm(VectorBackend ops) {
        assertEquals(5f, ops.norm(new float[]{3, 4}), DELTA);
    }

    @ParameterizedTest
    @MethodSource("backends")
    void cosineSimilarity_parallel(VectorBackend ops) {
        assertEquals(1f, ops.cosineSimilarity(rampA(), rampB()), DELTA);
    }

    @ParameterizedTest
    @MethodSource("backends")
    void cosineSimilarity_orthogonal(VectorBackend ops) {
        assertEquals(0f, ops.cosineSimilarity(new float[]{1, 0}, new float[]{0, 1}), DELTA);
    }

    @ParameterizedTest
    @MethodSource("backends")
    void cosineSimilarity_zeroNorm_throws(VectorBackend ops) {
        assertThrows(IllegalArgumentException.class,
                () -> ops.cosineSimilarity(new float[]{0, 0}, new float[]{1, 2}));
        assertThrows(IllegalArgumentException.class,
                () -> ops.cosineSimilarity(new float[]{1, 2}, new float[]{0, 0}));
    }

    @ParameterizedTest
    @MethodSource("backends")
    void add(VectorBackend ops) {
        float[] result = ops.add(rampA(), rampB());
        for (int i = 0; i < result.length; i++) {
            assertEquals(3f * (i + 1), result[i], DELTA);
        }
    }

    @ParameterizedTest
    @MethodSource("backends")
    void subtract(VectorBackend ops) {
        float[] result = ops.subtract(rampB(), rampA());
        for (int i = 0; i < result.length; i++) {
            assertEquals(i + 1, result[i], DELTA);
        }
    }

    @ParameterizedTest
    @MethodSource("backends")
    void multiply(VectorBackend ops) {
        float[] result = ops.multiply(rampA(), rampB());
        for (int i = 0; i < result.length; i++) {
            assertEquals(2f * (i + 1) * (i + 1), result[i], DELTA);
        }
    }

    @ParameterizedTest
    @MethodSource("backends")
    void divide(VectorBackend ops) {
        float[] result = ops.divide(rampB(), rampA());
        for (float v : result) {
            assertEquals(2f, v, DELTA);
        }
    }

    @ParameterizedTest
    @MethodSource("backends")
    void mismatchedLength_throws(VectorBackend ops) {
        assertThrows(IllegalArgumentException.class,
                () -> ops.add(new float[]{1, 2}, new float[]{1}));
    }

    @ParameterizedTest
    @MethodSource("backends")
    void sum(VectorBackend ops) {
        assertEquals(190f, ops.sum(rampA()), DELTA); // 19*20/2 = 190
    }

    @ParameterizedTest
    @MethodSource("backends")
    void minMax(VectorBackend ops) {
        float[] a = {3, -1, 5, 2, -7, 8, 0, 4, 6, 1, -2, 9, 3, 5, -4, 7, 2, 8, 1};
        assertEquals(-7f, ops.min(a), DELTA);
        assertEquals(9f, ops.max(a), DELTA);
    }

    @ParameterizedTest
    @MethodSource("backends")
    void min_empty_throws(VectorBackend ops) {
        assertThrows(IllegalArgumentException.class, () -> ops.min(new float[]{}));
    }

    @ParameterizedTest
    @MethodSource("backends")
    void max_empty_throws(VectorBackend ops) {
        assertThrows(IllegalArgumentException.class, () -> ops.max(new float[]{}));
    }

    @ParameterizedTest
    @MethodSource("backends")
    void scale(VectorBackend ops) {
        float[] result = ops.scale(rampA(), 3f);
        for (int i = 0; i < result.length; i++) {
            assertEquals(3f * (i + 1), result[i], DELTA);
        }
    }

    @ParameterizedTest
    @MethodSource("backends")
    void copy_isIndependent(VectorBackend ops) {
        float[] original = rampA();
        float[] copy = ops.copy(original);
        assertArrayEquals(original, copy, DELTA);
        assertNotSame(original, copy);
    }

    @ParameterizedTest
    @MethodSource("backends")
    void fill(VectorBackend ops) {
        float[] a = new float[19];
        ops.fill(a, 7f);
        for (float v : a) assertEquals(7f, v, DELTA);
    }

    @ParameterizedTest
    @MethodSource("backends")
    void normalize(VectorBackend ops) {
        float[] result = ops.normalize(rampA());
        assertEquals(1f, ops.norm(result), DELTA);
    }

    @ParameterizedTest
    @MethodSource("backends")
    void normalize_zero_throws(VectorBackend ops) {
        assertThrows(IllegalArgumentException.class, () -> ops.normalize(new float[19]));
    }

    @ParameterizedTest
    @MethodSource("backends")
    void euclideanDistance(VectorBackend ops) {
        assertEquals(5f, ops.euclideanDistance(new float[]{0, 0}, new float[]{3, 4}), DELTA);
    }

    @ParameterizedTest
    @MethodSource("backends")
    void fma(VectorBackend ops) {
        float[] result = ops.fma(rampA(), rampB(), rampA());
        for (int i = 0; i < result.length; i++) {
            // (i+1)*2*(i+1) + (i+1)
            float n = i + 1;
            assertEquals(2f * n * n + n, result[i], DELTA);
        }
    }

    @ParameterizedTest
    @MethodSource("backends")
    void fma_mismatchedLength_throws(VectorBackend ops) {
        assertThrows(IllegalArgumentException.class,
                () -> ops.fma(new float[]{1, 2, 3}, new float[]{1, 2, 3}, new float[]{1, 2}));
    }

    @ParameterizedTest
    @MethodSource("backends")
    void argmax(VectorBackend ops) {
        float[] a = {3, -1, 5, 2, -7, 8, 0, 4, 6, 1, -2, 9, 3, 5, -4, 7, 2, 8, 1};
        assertEquals(11, ops.argmax(a)); // value 9 at index 11
    }

    @ParameterizedTest
    @MethodSource("backends")
    void argmax_empty_throws(VectorBackend ops) {
        assertThrows(IllegalArgumentException.class, () -> ops.argmax(new float[]{}));
    }

    @ParameterizedTest
    @MethodSource("backends")
    void softmax_sumsToOne(VectorBackend ops) {
        float[] result = ops.softmax(rampA());
        float total = 0f;
        for (float v : result) {
            assertTrue(v >= 0f);
            total += v;
        }
        assertEquals(1f, total, 1e-4f);
    }

    @ParameterizedTest
    @MethodSource("backends")
    void softmax_empty_throws(VectorBackend ops) {
        assertThrows(IllegalArgumentException.class, () -> ops.softmax(new float[]{}));
    }
}
