package com;

import com.scalar.ScalarVectorOps;
import com.simd.SimdVectorOps;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;


class SimdCorrectnessTest {

    private static final float DELTA = 1e-4f;

    private static final long SEED = 42L;
    private Random random;

    @BeforeEach
    void setUp() {
        random = new Random(SEED);
    }

    private float[] randomVector(int size) {
        float[] v = new float[size];
        for (int i = 0; i < size; i++) v[i] = (random.nextFloat() - 0.5f) * 10f;
        return v;
    }

    private float[] randomPositiveVector(int size) {
        float[] v = new float[size];
        for (int i = 0; i < size; i++) v[i] = random.nextFloat() + 0.01f;
        return v;
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 4, 7, 8, 9, 15, 16, 17, 64, 255, 256, 257, 1024, 4096})
    void dot_matchesScalar(int size) {
        float[] a = randomVector(size);
        float[] b = randomVector(size);
        assertEquals(ScalarVectorOps.dot(a, b), SimdVectorOps.dot(a, b), DELTA,
                "dot mismatch at size " + size);
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 4, 7, 8, 9, 15, 16, 17, 64, 255, 256, 257, 1024, 4096})
    void norm_matchesScalar(int size) {
        float[] a = randomVector(size);
        assertEquals(ScalarVectorOps.norm(a), SimdVectorOps.norm(a), DELTA,
                "norm mismatch at size " + size);
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 8, 16, 64, 256, 1024})
    void cosine_matchesScalar(int size) {
        float[] a = randomVector(size);
        float[] b = randomVector(size);
        assertEquals(ScalarVectorOps.cosineSimilarity(a, b),
                SimdVectorOps.cosineSimilarity(a, b), DELTA,
                "cosine mismatch at size " + size);
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 4, 7, 8, 9, 16, 64, 256, 1024})
    void add_matchesScalar(int size) {
        float[] a = randomVector(size);
        float[] b = randomVector(size);
        assertArrayEquals(ScalarVectorOps.add(a, b), SimdVectorOps.add(a, b), DELTA,
                "add mismatch at size " + size);
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 4, 7, 8, 9, 16, 64, 256, 1024})
    void subtract_matchesScalar(int size) {
        float[] a = randomVector(size);
        float[] b = randomVector(size);
        assertArrayEquals(ScalarVectorOps.subtract(a, b), SimdVectorOps.subtract(a, b), DELTA,
                "subtract mismatch at size " + size);
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 4, 7, 8, 9, 16, 64, 256, 1024})
    void multiply_matchesScalar(int size) {
        float[] a = randomVector(size);
        float[] b = randomVector(size);
        assertArrayEquals(ScalarVectorOps.multiply(a, b), SimdVectorOps.multiply(a, b), DELTA,
                "multiply mismatch at size " + size);
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 4, 7, 8, 9, 16, 64, 256, 1024})
    void divide_matchesScalar(int size) {
        float[] a = randomVector(size);
        float[] b = randomPositiveVector(size);
        assertArrayEquals(ScalarVectorOps.divide(a, b), SimdVectorOps.divide(a, b), DELTA,
                "divide mismatch at size " + size);
    }
}
