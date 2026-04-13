package com.scalar;

import java.util.Arrays;

public class ScalarVectorOps {
    public static float dot(float[] a, float[] b) {
        checkSameLength(a, b);

        float sum = 0f;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    public static float norm(float[] a) {
        float sum = 0f;
        for (float v : a) {
            sum += v * v;
        }
        return (float) Math.sqrt(sum);
    }

    public static float cosineSimilarity(float[] a, float[] b) {
        checkSameLength(a, b);

        float dot = dot(a, b);
        float normA = norm(a);
        float normB = norm(b);

        if (normA == 0 || normB == 0) {
            throw new IllegalArgumentException("Zero vector not allowed");
        }

        return dot / (normA * normB);
    }

    public static float[] add(float[] a, float[] b) {
        checkSameLength(a, b);

        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i];
        }
        return result;
    }

    private static void checkSameLength(float[] a, float[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Vectors must have same length");
        }
    }

    public static float[] subtract(float[] a, float[] b) {
        checkSameLength(a, b);

        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] - b[i];
        }
        return result;
    }

    public static float[] multiply(float[] a, float[] b) {
        checkSameLength(a, b);

        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] * b[i];
        }
        return result;
    }

    public static float[] divide(float[] a, float[] b) {
        checkSameLength(a, b);

        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] / b[i];
        }
        return result;
    }

    public static float sum(float[] a) {
        float s = 0f;
        for (float v : a) {
            s += v;
        }
        return s;
    }

    public static float min(float[] a) {
        if (a.length == 0) throw new IllegalArgumentException("Empty array");

        float m = a[0];
        for (int i = 1; i < a.length; i++) {
            if (a[i] < m) m = a[i];
        }
        return m;
    }

    public static float max(float[] a) {
        if (a.length == 0) throw new IllegalArgumentException("Empty array");

        float m = a[0];
        for (int i = 1; i < a.length; i++) {
            if (a[i] > m) m = a[i];
        }
        return m;
    }

    public static float[] scale(float[] a, float scalar) {
        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] * scalar;
        }
        return result;
    }

    public static float[] copy(float[] a) {
        float[] result = new float[a.length];
        System.arraycopy(a, 0, result, 0, a.length);
        return result;
    }

    public static void fill(float[] a, float value) {
        Arrays.fill(a, value);
    }

    public static float[] normalize(float[] a) {
        float norm = norm(a);

        if (norm == 0) {
            throw new IllegalArgumentException("Cannot normalize zero vector");
        }

        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] / norm;
        }
        return result;
    }
}
