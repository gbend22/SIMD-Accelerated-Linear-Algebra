package com.simd;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class SimdVectorOps {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private static void checkSameLength(float[] a, float[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException(
                    "Vectors must have the same length, got " + a.length + " and " + b.length);
        }
    }

    public static float dot(float[] a, float[] b) {
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

    public static float norm(float[] a) {
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

    public static float cosineSimilarity(float[] a, float[] b) {
        checkSameLength(a, b);

        float dot = dot(a, b);
        float normA = norm(a);
        float normB = norm(b);

        if (normA == 0f || normB == 0f) {
            throw new IllegalArgumentException("Zero vector not allowed");
        }

        return dot / (normA * normB);
    }

    public static float[] add(float[] a, float[] b) {
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

    public static float[] subtract(float[] a, float[] b) {
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

    public static float[] multiply(float[] a, float[] b) {
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

    public static float[] divide(float[] a, float[] b) {
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

    public static float sum(float[] a) {
        throw  new UnsupportedOperationException("Not supported yet.");
    }

    public static float min(float[] a) {
        throw  new UnsupportedOperationException("Not supported yet.");
    }

    public static float max(float[] a) {
        throw  new UnsupportedOperationException("Not supported yet.");
    }

    public static float[] scale(float[] a, float scalar) {
        throw  new UnsupportedOperationException("Not supported yet.");
    }

    public static float[] copy(float[] a) {
        throw  new UnsupportedOperationException("Not supported yet.");
    }

    public static void fill(float[] a, float value) {
        throw  new UnsupportedOperationException("Not supported yet.");
    }

    public static float[] normalize(float[] a) {
        throw  new UnsupportedOperationException("Not supported yet.");
    }
}
