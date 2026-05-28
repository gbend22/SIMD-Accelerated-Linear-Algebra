package com;

import com.scalar.ScalarVectorOps;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ScalarVectorOpsTest {

    private final ScalarVectorOps ops = new ScalarVectorOps();

    @Test
    void testDotProduct() {
        float[] a = {1, 2, 3};
        float[] b = {4, 5, 6};

        float result = ops.dot(a, b);

        assertEquals(32f, result, 1e-6);
    }

    @Test
    void testNorm() {
        float[] a = {3, 4};

        float result = ops.norm(a);

        assertEquals(5f, result, 1e-6);
    }

    @Test
    void testCosineSimilarity() {
        float[] a = {1, 0};
        float[] b = {0, 1};

        float result = ops.cosineSimilarity(a, b);

        assertEquals(0f, result, 1e-6);
    }

    @Test
    void testAdd() {
        float[] a = {1, 2, 3};
        float[] b = {4, 5, 6};

        float[] result = ops.add(a, b);

        assertArrayEquals(new float[]{5, 7, 9}, result, 1e-6f);
    }

    @Test
    void testDifferentLengths() {
        float[] a = {1, 2};
        float[] b = {1};

        assertThrows(IllegalArgumentException.class,
                () -> ops.dot(a, b));
    }

    @Test
    void testZeroVectorCosineSimilarity() {
        float[] a = {0, 0};
        float[] b = {1, 3};

        assertThrows(IllegalArgumentException.class,
                () -> ops.cosineSimilarity(a, b));
        assertThrows(IllegalArgumentException.class,
                () -> ops.cosineSimilarity(b, a));
    }

    @Test
    void testSubtract() {
        float[] a = {5, 7, 9};
        float[] b = {1, 2, 3};

        assertArrayEquals(new float[]{4, 5, 6},
                ops.subtract(a, b), 1e-6f);
    }

    @Test
    void testMultiply() {
        float[] a = {1, 2, 3};
        float[] b = {4, 5, 6};

        assertArrayEquals(new float[]{4, 10, 18},
                ops.multiply(a, b), 1e-6f);
    }

    @Test
    void testDivide() {
        float[] a = {4, 15, 27};
        float[] b = {2, 5, 3};

        assertArrayEquals(new float[]{2, 3, 9},
                ops.divide(a, b), 1e-6f);
    }

    @Test
    void testSum() {
        float[] a = {1, 2, 3};

        assertEquals(6f, ops.sum(a), 1e-6);
    }

    @Test
    void testMinMax() {
        float[] a = {3, -1, 5, 2};

        assertEquals(-1f, ops.min(a), 1e-6);
        assertEquals(5f, ops.max(a), 1e-6);
    }

    @Test
    void testScale() {
        float[] a = {1, 2, 3};

        assertArrayEquals(new float[]{2, 4, 6},
                ops.scale(a, 2), 1e-6f);
    }

    @Test
    void testNormalize() {
        float[] a = {3, 4};

        float[] result = ops.normalize(a);

        assertEquals(1f, ops.norm(result), 1e-6);
    }

    @Test
    void testCopy() {
        float[] original = {1, 2, 3};

        float[] copy = ops.copy(original);

        assertArrayEquals(original, copy, 1e-6f);

        assertNotSame(original, copy);
    }

    @Test
    void testFill() {
        float[] a = {1, 2, 3};

        ops.fill(a, 7);

        assertArrayEquals(new float[]{7, 7, 7}, a, 1e-6f);
    }

    @Test
    void testFillEmptyArray() {
        float[] a = {};

        ops.fill(a, 5);

        assertArrayEquals(new float[]{}, a, 1e-6f);
    }
}
