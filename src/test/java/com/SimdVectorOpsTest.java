package com;

import com.scalar.ScalarVectorOps;
import com.simd.SimdVectorOps;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;


class SimdVectorOpsTest {

    private static final float DELTA = 1e-3f;

    private static final long SEED = 42L;

    private static final ScalarVectorOps scalar = new ScalarVectorOps();
    private static final SimdVectorOps simd = new SimdVectorOps();

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
        assertEquals(scalar.dot(a, b), simd.dot(a, b), DELTA,
                "dot mismatch at size " + size);
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 4, 7, 8, 9, 15, 16, 17, 64, 255, 256, 257, 1024, 4096})
    void norm_matchesScalar(int size) {
        float[] a = randomVector(size);
        assertEquals(scalar.norm(a), simd.norm(a), DELTA,
                "norm mismatch at size " + size);
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 8, 16, 64, 256, 1024})
    void cosine_matchesScalar(int size) {
        float[] a = randomVector(size);
        float[] b = randomVector(size);
        assertEquals(scalar.cosineSimilarity(a, b),
                simd.cosineSimilarity(a, b), DELTA,
                "cosine mismatch at size " + size);
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 4, 7, 8, 9, 16, 64, 256, 1024})
    void add_matchesScalar(int size) {
        float[] a = randomVector(size);
        float[] b = randomVector(size);
        assertArrayEquals(scalar.add(a, b), simd.add(a, b), DELTA,
                "add mismatch at size " + size);
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 4, 7, 8, 9, 16, 64, 256, 1024})
    void subtract_matchesScalar(int size) {
        float[] a = randomVector(size);
        float[] b = randomVector(size);
        assertArrayEquals(scalar.subtract(a, b), simd.subtract(a, b), DELTA,
                "subtract mismatch at size " + size);
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 4, 7, 8, 9, 16, 64, 256, 1024})
    void multiply_matchesScalar(int size) {
        float[] a = randomVector(size);
        float[] b = randomVector(size);
        assertArrayEquals(scalar.multiply(a, b), simd.multiply(a, b), DELTA,
                "multiply mismatch at size " + size);
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 4, 7, 8, 9, 16, 64, 256, 1024})
    void divide_matchesScalar(int size) {
        float[] a = randomVector(size);
        float[] b = randomPositiveVector(size);
        assertArrayEquals(scalar.divide(a, b), simd.divide(a, b), DELTA,
                "divide mismatch at size " + size);
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 4, 7, 8, 9, 16, 64, 256, 1024})
    void sum_matchesScalar(int size) {
        float[] a = randomVector(size);
        assertEquals(scalar.sum(a), simd.sum(a), DELTA,
                "sum mismatch at size " + size);
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 4, 7, 8, 9, 16, 64, 256, 1024})
    void min_matchesScalar(int size) {
        float[] a = randomVector(size);
        assertEquals(scalar.min(a), simd.min(a), DELTA,
                "min mismatch at size " + size);
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 4, 7, 8, 9, 16, 64, 256, 1024})
    void max_matchesScalar(int size) {
        float[] a = randomVector(size);
        assertEquals(scalar.max(a), simd.max(a), DELTA,
                "max mismatch at size " + size);
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 4, 7, 8, 9, 16, 64, 256, 1024})
    void scale_matchesScalar(int size) {
        float[] a = randomVector(size);
        float sc = 3.14f;
        assertArrayEquals(scalar.scale(a, sc), simd.scale(a, sc), DELTA,
                "scale mismatch at size " + size);
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 4, 7, 8, 9, 16, 64, 256, 1024})
    void normalize_matchesScalar(int size) {
        float[] a = randomVector(size);
        assertArrayEquals(scalar.normalize(a), simd.normalize(a), DELTA,
                "normalize mismatch at size " + size);
    }

    @Test
    void normalize_resultHasUnitLength() {
        float[] a = randomVector(1024);
        float[] result = simd.normalize(a);
        assertEquals(1f, simd.norm(result), DELTA,
                "Normalized SIMD vector should have unit length");
    }

    @Test
    void dot_withNaN_returnsNaN() {
        float[] a = {1f, Float.NaN};
        float[] b = {2f, 3f};

        assertTrue(Float.isNaN(simd.dot(a, b)));
    }

    @Test
    void norm_withNaN_returnsNaN() {
        float[] a = {1f, Float.NaN};

        assertTrue(Float.isNaN(simd.norm(a)));
    }

    @Test
    void sum_withInfinity_returnsInfinity() {
        float[] a = {1f, Float.POSITIVE_INFINITY};

        assertEquals(Float.POSITIVE_INFINITY, simd.sum(a));
    }

    @Test
    void multiply_withInfinity_behavesCorrectly() {
        float[] a = {2f, Float.POSITIVE_INFINITY};
        float[] b = {3f, 2f};

        float[] result = simd.multiply(a, b);

        assertEquals(6f, result[0], DELTA);
        assertEquals(Float.POSITIVE_INFINITY, result[1]);
    }

    @Test
    void normalize_zeroVector_throwsException() {
        float[] a = new float[16];

        assertThrows(
                IllegalArgumentException.class,
                () -> simd.normalize(a)
        );
    }

    @Test
    void cosine_zeroVector_throwsException() {
        float[] a = new float[16];
        float[] b = randomVector(16);

        assertThrows(
                IllegalArgumentException.class,
                () -> simd.cosineSimilarity(a, b)
        );
    }

    @Test
    void dot_mismatchedLengths_throwsException() {
        float[] a = new float[8];
        float[] b = new float[16];

        assertThrows(
                IllegalArgumentException.class,
                () -> simd.dot(a, b)
        );
    }

    @Test
    void add_mismatchedLengths_throwsException() {
        float[] a = new float[8];
        float[] b = new float[16];

        assertThrows(
                IllegalArgumentException.class,
                () -> simd.add(a, b)
        );
    }

    @Test
    void dot_largeValues_doesNotCrash() {
        float[] a = {1e10f, 1e10f};
        float[] b = {1e10f, 1e10f};

        float result = simd.dot(a, b);

        assertTrue(Float.isFinite(result) || Float.isInfinite(result));
    }

}
