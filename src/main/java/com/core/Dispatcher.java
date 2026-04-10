package com.core;

import com.scalar.ScalarVectorOps;
import com.simd.SimdVectorOps;

public class Dispatcher {
    private static final boolean USE_SIMD = false;

    public static float dot(float[] a, float[] b) {
        if (USE_SIMD) {
            return SimdVectorOps.dot(a, b);
        } else {
            return ScalarVectorOps.dot(a, b);
        }
    }

    public static float norm(float[] a) {
        if (USE_SIMD) {
            return SimdVectorOps.norm(a);
        } else {
            return ScalarVectorOps.norm(a);
        }
    }

    public static float cosineSimilarity(float[] a, float[] b) {
        if (USE_SIMD) {
            return SimdVectorOps.cosineSimilarity(a, b);
        } else {
            return ScalarVectorOps.cosineSimilarity(a, b);
        }
    }

    public static float[] add(float[] a, float[] b) {
        if (USE_SIMD) {
            return SimdVectorOps.add(a, b);
        } else {
            return ScalarVectorOps.add(a, b);
        }
    }
}
