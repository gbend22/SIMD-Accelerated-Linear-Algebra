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
}
