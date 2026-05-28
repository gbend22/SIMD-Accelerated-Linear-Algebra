package com.scalar;

import com.core.VectorBackend;
import java.util.Arrays;

public class ScalarVectorOps implements VectorBackend {

    @Override
    public float dot(float[] a, float[] b) {
        checkSameLength(a, b);

        float sum = 0f;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    @Override
    public float norm(float[] a) {
        float sum = 0f;
        for (float v : a) {
            sum += v * v;
        }
        return (float) Math.sqrt(sum);
    }

    @Override
    public float cosineSimilarity(float[] a, float[] b) {
        checkSameLength(a, b);

        float dot = dot(a, b);
        float normA = norm(a);
        float normB = norm(b);

        if (normA == 0 || normB == 0) {
            throw new IllegalArgumentException("Zero vector not allowed");
        }

        return dot / (normA * normB);
    }

    @Override
    public float[] add(float[] a, float[] b) {
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

    @Override
    public float[] subtract(float[] a, float[] b) {
        checkSameLength(a, b);

        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] - b[i];
        }
        return result;
    }

    @Override
    public float[] multiply(float[] a, float[] b) {
        checkSameLength(a, b);

        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] * b[i];
        }
        return result;
    }

    @Override
    public float[] divide(float[] a, float[] b) {
        checkSameLength(a, b);

        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] / b[i];
        }
        return result;
    }

    @Override
    public float sum(float[] a) {
        float s = 0f;
        for (float v : a) {
            s += v;
        }
        return s;
    }

    @Override
    public float min(float[] a) {
        if (a.length == 0) throw new IllegalArgumentException("Empty array");

        float m = a[0];
        for (int i = 1; i < a.length; i++) {
            if (a[i] < m) m = a[i];
        }
        return m;
    }

    @Override
    public float max(float[] a) {
        if (a.length == 0) throw new IllegalArgumentException("Empty array");

        float m = a[0];
        for (int i = 1; i < a.length; i++) {
            if (a[i] > m) m = a[i];
        }
        return m;
    }

    @Override
    public float[] scale(float[] a, float scalar) {
        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] * scalar;
        }
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

    @Override
    public float euclideanDistance(float[] a, float[] b) {
        checkSameLength(a, b);

        float sum = 0f;
        for (int i = 0; i < a.length; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return (float) Math.sqrt(sum);
    }

    @Override
    public float[] fma(float[] a, float[] b, float[] c) {
        checkSameLength(a, b);
        checkSameLength(a, c);

        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = Math.fma(a[i], b[i], c[i]);
        }
        return result;
    }

    @Override
    public int argmax(float[] a) {
        if (a.length == 0) throw new IllegalArgumentException("Empty array");

        int idx = 0;
        float m = a[0];
        for (int i = 1; i < a.length; i++) {
            if (a[i] > m) {
                m = a[i];
                idx = i;
            }
        }
        return idx;
    }

    @Override
    public float[] softmax(float[] a) {
        if (a.length == 0) throw new IllegalArgumentException("Empty array");

        float maxVal = max(a);
        float[] result = new float[a.length];
        float sum = 0f;
        for (int i = 0; i < a.length; i++) {
            result[i] = (float) Math.exp(a[i] - maxVal);
            sum += result[i];
        }
        for (int i = 0; i < a.length; i++) {
            result[i] /= sum;
        }
        return result;
    }
}
