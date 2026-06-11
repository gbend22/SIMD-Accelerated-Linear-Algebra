package com;

import com.vector.VectorOps;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests the public {@link VectorOps} façade. Because every method delegates through
 * {@link com.core.Dispatcher} to the active backend, exercising the façade here also
 * covers the dispatcher's delegation methods and its one-time backend selection.
 */
class VectorOpsTest {

    private static final float DELTA = 1e-6f;

    @Test
    void dot() {
        assertEquals(32f, VectorOps.dot(new float[]{1, 2, 3}, new float[]{4, 5, 6}), DELTA);
    }

    @Test
    void norm() {
        assertEquals(5f, VectorOps.norm(new float[]{3, 4}), DELTA);
    }

    @Test
    void cosineSimilarity() {
        assertEquals(0f, VectorOps.cosineSimilarity(new float[]{1, 0}, new float[]{0, 1}), DELTA);
        assertEquals(1f, VectorOps.cosineSimilarity(new float[]{1, 2, 3}, new float[]{2, 4, 6}), DELTA);
    }

    @Test
    void cosineSimilarity_zeroNorm_throws() {
        assertThrows(IllegalArgumentException.class,
                () -> VectorOps.cosineSimilarity(new float[]{0, 0}, new float[]{1, 2}));
    }

    @Test
    void add() {
        assertArrayEquals(new float[]{5, 7, 9},
                VectorOps.add(new float[]{1, 2, 3}, new float[]{4, 5, 6}), DELTA);
    }

    @Test
    void subtract() {
        assertArrayEquals(new float[]{4, 5, 6},
                VectorOps.subtract(new float[]{5, 7, 9}, new float[]{1, 2, 3}), DELTA);
    }

    @Test
    void multiply() {
        assertArrayEquals(new float[]{4, 10, 18},
                VectorOps.multiply(new float[]{1, 2, 3}, new float[]{4, 5, 6}), DELTA);
    }

    @Test
    void divide() {
        assertArrayEquals(new float[]{2, 3, 9},
                VectorOps.divide(new float[]{4, 15, 27}, new float[]{2, 5, 3}), DELTA);
    }

    @Test
    void mismatchedLengths_throws() {
        assertThrows(IllegalArgumentException.class,
                () -> VectorOps.add(new float[]{1, 2}, new float[]{1}));
    }

    @Test
    void sum() {
        assertEquals(6f, VectorOps.sum(new float[]{1, 2, 3}), DELTA);
    }

    @Test
    void minMax() {
        float[] a = {3, -1, 5, 2};
        assertEquals(-1f, VectorOps.min(a), DELTA);
        assertEquals(5f, VectorOps.max(a), DELTA);
    }

    @Test
    void scale() {
        assertArrayEquals(new float[]{2, 4, 6},
                VectorOps.scale(new float[]{1, 2, 3}, 2), DELTA);
    }

    @Test
    void copy_isIndependent() {
        float[] original = {1, 2, 3};
        float[] copy = VectorOps.copy(original);
        assertArrayEquals(original, copy, DELTA);
        assertNotSame(original, copy);
    }

    @Test
    void fill() {
        float[] a = {1, 2, 3};
        VectorOps.fill(a, 7);
        assertArrayEquals(new float[]{7, 7, 7}, a, DELTA);
    }

    @Test
    void normalize() {
        float[] result = VectorOps.normalize(new float[]{3, 4});
        assertEquals(1f, VectorOps.norm(result), DELTA);
    }

    @Test
    void euclideanDistance() {
        assertEquals(5f, VectorOps.euclideanDistance(new float[]{0, 0}, new float[]{3, 4}), DELTA);
    }

    @Test
    void fma() {
        assertArrayEquals(new float[]{7, 16, 27},
                VectorOps.fma(new float[]{1, 2, 3}, new float[]{4, 5, 6}, new float[]{3, 6, 9}), DELTA);
    }

    @Test
    void argmax() {
        assertEquals(2, VectorOps.argmax(new float[]{1, 3, 9, 4}));
    }

    @Test
    void argmax_ties_returnsFirst() {
        assertEquals(1, VectorOps.argmax(new float[]{1, 5, 5, 2}));
    }

    @Test
    void softmax_sumsToOne() {
        float[] result = VectorOps.softmax(new float[]{1, 2, 3});
        float total = 0f;
        for (float v : result) {
            assertTrue(v >= 0f);
            total += v;
        }
        assertEquals(1f, total, 1e-5f);
    }
}
