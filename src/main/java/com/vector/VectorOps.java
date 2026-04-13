package com.vector;

import com.core.Dispatcher;
import com.scalar.ScalarVectorOps;
import com.simd.SimdVectorOps;

public class VectorOps {
    public static float dot(float[] a, float[] b) {
        return Dispatcher.dot(a, b);
    }

    public static float norm(float[] a) {
        return Dispatcher.norm(a);
    }

    public static float cosineSimilarity(float[] a, float[] b) {
        return Dispatcher.cosineSimilarity(a, b);
    }

    public static float[] add(float[] a, float[] b) {
        return Dispatcher.add(a, b);
    }

    public static float[] subtract(float[] a, float[] b) {
        throw  new UnsupportedOperationException("Not supported yet.");
    }

    public static float[] multiply(float[] a, float[] b) {
        throw  new UnsupportedOperationException("Not supported yet.");
    }

    public static float[] divide(float[] a, float[] b) {
        throw  new UnsupportedOperationException("Not supported yet.");
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
