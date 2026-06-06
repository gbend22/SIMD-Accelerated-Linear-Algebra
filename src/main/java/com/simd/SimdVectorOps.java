package com.simd;

import com.core.VectorBackend;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.Arrays;

/**
 * SIMD implementation of {@link com.core.VectorBackend}, built on the Java Vector API
 * ({@code jdk.incubator.vector}). Each operation processes the array in vector-width
 * chunks and handles the remaining tail elements with a scalar loop. Selected at
 * runtime when a usable vector width is available; internal, not part of the public API.
 */
public class SimdVectorOps implements VectorBackend {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private static void checkSameLength(float[] a, float[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException(
                    "Vectors must have the same length, got " + a.length + " and " + b.length);
        }
    }

    private static void checkNonEmpty(float[] a) {
        if (a.length == 0) throw new IllegalArgumentException("Array must not be empty");
    }

    @Override
    public float dot(float[] a, float[] b) {
        checkSameLength(a, b);

        int i = 0;
        int bound = SPECIES.loopBound(a.length);
        var acc = FloatVector.zero(SPECIES);

        for (; i < bound; i += SPECIES.length()) {
            var va = FloatVector.fromArray(SPECIES, a, i);
            var vb = FloatVector.fromArray(SPECIES, b, i);
            acc = va.fma(vb, acc);
        }

        float result = acc.reduceLanes(VectorOperators.ADD);

        for (; i < a.length; i++) result += a[i] * b[i];

        return result;
    }

    @Override
    public float norm(float[] a) {
        int i = 0;
        int bound = SPECIES.loopBound(a.length);
        var acc = FloatVector.zero(SPECIES);

        for (; i < bound; i += SPECIES.length()) {
            var v = FloatVector.fromArray(SPECIES, a, i);
            acc = v.fma(v, acc);
        }

        float result = acc.reduceLanes(VectorOperators.ADD);

        for (; i < a.length; i++) result += a[i] * a[i];

        return (float) Math.sqrt(result);
    }

    @Override
    public float cosineSimilarity(float[] a, float[] b) {
        checkSameLength(a, b);

        float dot = dot(a, b);
        float normA = norm(a);
        float normB = norm(b);

        if (normA == 0f || normB == 0f) {
            throw new IllegalArgumentException("Zero vector not allowed");
        }

        return dot / (normA * normB);
    }

    @Override
    public float[] add(float[] a, float[] b) {
        checkSameLength(a, b);

        float[] result = new float[a.length];
        int i = 0;
        int bound = SPECIES.loopBound(a.length);

        for (; i < bound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, a, i)
                    .add(FloatVector.fromArray(SPECIES, b, i))
                    .intoArray(result, i);
        }

        for (; i < a.length; i++) result[i] = a[i] + b[i];

        return result;
    }

    @Override
    public float[] subtract(float[] a, float[] b) {
        checkSameLength(a, b);

        float[] result = new float[a.length];
        int i = 0;
        int bound = SPECIES.loopBound(a.length);

        for (; i < bound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, a, i)
                    .sub(FloatVector.fromArray(SPECIES, b, i))
                    .intoArray(result, i);
        }

        for (; i < a.length; i++) result[i] = a[i] - b[i];

        return result;
    }

    @Override
    public float[] multiply(float[] a, float[] b) {
        checkSameLength(a, b);

        float[] result = new float[a.length];
        int i = 0;
        int bound = SPECIES.loopBound(a.length);

        for (; i < bound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, a, i)
                    .mul(FloatVector.fromArray(SPECIES, b, i))
                    .intoArray(result, i);
        }

        for (; i < a.length; i++) result[i] = a[i] * b[i];

        return result;
    }

    @Override
    public float[] divide(float[] a, float[] b) {
        checkSameLength(a, b);

        float[] result = new float[a.length];
        int i = 0;
        int bound = SPECIES.loopBound(a.length);

        for (; i < bound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, a, i)
                    .div(FloatVector.fromArray(SPECIES, b, i))
                    .intoArray(result, i);
        }

        for (; i < a.length; i++) result[i] = a[i] / b[i];

        return result;
    }

    @Override
    public float sum(float[] a) {
        int i = 0;
        int bound = SPECIES.loopBound(a.length);
        var acc = FloatVector.zero(SPECIES);
        for (; i < bound; i += SPECIES.length()) {
            acc = acc.add(FloatVector.fromArray(SPECIES, a, i));
        }
        float result = acc.reduceLanes(VectorOperators.ADD);
        for (; i < a.length; i++) result += a[i];
        return result;
    }

    @Override
    public float min(float[] a) {
        checkNonEmpty(a);
        int i = 0;
        int bound = SPECIES.loopBound(a.length);

        var acc = FloatVector.broadcast(SPECIES, Float.MAX_VALUE);
        for (; i < bound; i += SPECIES.length()) {
            acc = acc.min(FloatVector.fromArray(SPECIES, a, i));
        }
        float result = acc.reduceLanes(VectorOperators.MIN);
        for (; i < a.length; i++) result = Math.min(result, a[i]);
        return result;
    }

    @Override
    public float max(float[] a) {
        checkNonEmpty(a);
        int i = 0;
        int bound = SPECIES.loopBound(a.length);

        var acc = FloatVector.broadcast(SPECIES, -Float.MAX_VALUE);
        for (; i < bound; i += SPECIES.length()) {
            acc = acc.max(FloatVector.fromArray(SPECIES, a, i));
        }
        float result = acc.reduceLanes(VectorOperators.MAX);
        for (; i < a.length; i++) result = Math.max(result, a[i]);
        return result;
    }

    @Override
    public float[] scale(float[] a, float scalar) {
        float[] result = new float[a.length];
        int i = 0;
        int bound = SPECIES.loopBound(a.length);
        var scalarVec = FloatVector.broadcast(SPECIES, scalar);
        for (; i < bound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, a, i)
                    .mul(scalarVec)
                    .intoArray(result, i);
        }
        for (; i < a.length; i++) result[i] = a[i] * scalar;
        return result;
    }

    @Override
    public float[] copy(float[] a) {
        float[] result = new float[a.length];
        System.arraycopy(a, 0, result, 0, a.length);
        return result;
    }

    @Override
    public void fill(float[] a, float value) {
        Arrays.fill(a, value);
    }

    @Override
    public float[] normalize(float[] a) {
        float n = norm(a);
        if (n == 0f) throw new IllegalArgumentException("Cannot normalize a zero vector");
        return scale(a, 1f / n);
    }

    @Override
    public float euclideanDistance(float[] a, float[] b) {
        checkSameLength(a, b);

        int i = 0;
        int bound = SPECIES.loopBound(a.length);
        var acc = FloatVector.zero(SPECIES);

        for (; i < bound; i += SPECIES.length()) {
            var diff = FloatVector.fromArray(SPECIES, a, i)
                    .sub(FloatVector.fromArray(SPECIES, b, i));
            acc = diff.fma(diff, acc);
        }

        float result = acc.reduceLanes(VectorOperators.ADD);
        for (; i < a.length; i++) {
            float diff = a[i] - b[i];
            result += diff * diff;
        }
        return (float) Math.sqrt(result);
    }

    @Override
    public float[] fma(float[] a, float[] b, float[] c) {
        checkSameLength(a, b);
        checkSameLength(a, c);

        float[] result = new float[a.length];
        int i = 0;
        int bound = SPECIES.loopBound(a.length);

        for (; i < bound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, a, i)
                    .fma(FloatVector.fromArray(SPECIES, b, i),
                         FloatVector.fromArray(SPECIES, c, i))
                    .intoArray(result, i);
        }

        for (; i < a.length; i++) result[i] = Math.fma(a[i], b[i], c[i]);

        return result;
    }

    @Override
    public int argmax(float[] a) {
        checkNonEmpty(a);

        float maxVal = max(a);
        for (int i = 0; i < a.length; i++) {
            if (a[i] == maxVal) return i;
        }
        return 0;
    }

    @Override
    public float[] softmax(float[] a) {
        checkNonEmpty(a);

        float maxVal = max(a);
        var maxVec = FloatVector.broadcast(SPECIES, maxVal);

        float[] result = new float[a.length];
        int i = 0;
        int bound = SPECIES.loopBound(a.length);

        for (; i < bound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, a, i)
                    .sub(maxVec)
                    .lanewise(VectorOperators.EXP)
                    .intoArray(result, i);
        }
        for (; i < a.length; i++) result[i] = (float) Math.exp(a[i] - maxVal);

        float total = sum(result);
        return scale(result, 1f / total);
    }

    public static int simdWidth() {
        return SPECIES.length();
    }
}
