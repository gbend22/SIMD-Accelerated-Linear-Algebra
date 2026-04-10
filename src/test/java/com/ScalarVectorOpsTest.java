package com;

import com.scalar.ScalarVectorOps;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ScalarVectorOpsTest {

    @Test
    void testDotProduct() {
        float[] a = {1, 2, 3};
        float[] b = {4, 5, 6};

        float result = ScalarVectorOps.dot(a, b);

        assertEquals(32f, result, 1e-6);
    }

    @Test
    void testNorm() {
        float[] a = {3, 4};

        float result = ScalarVectorOps.norm(a);

        assertEquals(5f, result, 1e-6);
    }

    @Test
    void testCosineSimilarity() {
        float[] a = {1, 0};
        float[] b = {0, 1};

        float result = ScalarVectorOps.cosineSimilarity(a, b);

        assertEquals(0f, result, 1e-6);
    }

    @Test
    void testAdd() {
        float[] a = {1, 2, 3};
        float[] b = {4, 5, 6};

        float[] result = ScalarVectorOps.add(a, b);

        assertArrayEquals(new float[]{5, 7, 9}, result, 1e-6f);
    }

    @Test
    void testDifferentLengths() {
        float[] a = {1, 2};
        float[] b = {1};

        assertThrows(IllegalArgumentException.class,
                () -> ScalarVectorOps.dot(a, b));
    }

    @Test
    void testZeroVectorCosineSimilarity() {
        float[] a = {0, 0};
        float[] b = {1, 3};

        assertThrows(IllegalArgumentException.class,
                () -> ScalarVectorOps.cosineSimilarity(a, b));
        assertThrows(IllegalArgumentException.class,
                () -> ScalarVectorOps.cosineSimilarity(b, a));
    }

    @Test
    void testSubtract() {
        float[] a = {5, 7, 9};
        float[] b = {1, 2, 3};

        assertArrayEquals(new float[]{4, 5, 6},
                ScalarVectorOps.subtract(a, b), 1e-6f);
    }

    @Test
    void testMultiply() {
        float[] a = {1, 2, 3};
        float[] b = {4, 5, 6};

        assertArrayEquals(new float[]{4, 10, 18},
                ScalarVectorOps.multiply(a, b), 1e-6f);
    }

    @Test
    void testDivide() {
        float[] a = {4, 15, 27};
        float[] b = {2, 5, 3};

        assertArrayEquals(new float[]{2, 3, 9},
                ScalarVectorOps.divide(a, b), 1e-6f);
    }

    @Test
    void testSum() {
        float[] a = {1, 2, 3};

        assertEquals(6f, ScalarVectorOps.sum(a), 1e-6);
    }

    @Test
    void testMinMax() {
        float[] a = {3, -1, 5, 2};

        assertEquals(-1f, ScalarVectorOps.min(a), 1e-6);
        assertEquals(5f, ScalarVectorOps.max(a), 1e-6);
    }
}
