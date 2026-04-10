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
}
