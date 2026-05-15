package com.core;

import com.scalar.ScalarVectorOps;
import com.simd.SimdVectorOps;

public class Dispatcher {

    private static final boolean USE_SIMD;

    static {

        int width = SimdVectorOps.simdWidth();

        USE_SIMD = width >= 4;

        if (USE_SIMD) {
            System.out.println(
                    "[SimdLinalg] SIMD enabled with "
                            + width +
                            " float lanes"
            );
        } else {
            System.out.println(
                    "[SimdLinalg] SIMD unavailable, falling back to scalar backend"
            );
        }
    }

    public static boolean useSimd() {
        return USE_SIMD;
    }

    private Dispatcher() {}

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

    public static float[] subtract(float[] a, float[] b) {
        if (USE_SIMD) {
            return SimdVectorOps.subtract(a, b);
        } else {
            return ScalarVectorOps.subtract(a, b);
        }
    }

    public static float[] multiply(float[] a, float[] b) {
        if (USE_SIMD) {
            return SimdVectorOps.multiply(a, b);
        } else {
            return ScalarVectorOps.multiply(a, b);
        }
    }

    public static float[] divide(float[] a, float[] b) {
        if (USE_SIMD) {
            return SimdVectorOps.divide(a, b);
        } else {
            return ScalarVectorOps.divide(a, b);
        }
    }

    public static float sum(float[] a) {
        if (USE_SIMD) {
            return SimdVectorOps.sum(a);
        } else {
            return ScalarVectorOps.sum(a);
        }
    }

    public static float min(float[] a) {
        if (USE_SIMD) {
            return SimdVectorOps.min(a);
        } else {
            return ScalarVectorOps.min(a);
        }
    }

    public static float max(float[] a) {
        if (USE_SIMD) {
            return SimdVectorOps.max(a);
        } else {
            return ScalarVectorOps.max(a);
        }
    }

    public static float[] scale(float[] a, float scalar) {
        if (USE_SIMD) {
            return SimdVectorOps.scale(a, scalar);
        } else {
            return ScalarVectorOps.scale(a, scalar);
        }
    }

    public static float[] copy(float[] a) {
        if (USE_SIMD) {
            return SimdVectorOps.copy(a);
        } else {
            return ScalarVectorOps.copy(a);
        }
    }

    public static void fill(float[] a, float value) {
        if (USE_SIMD) {
            SimdVectorOps.fill(a, value);
        } else {
            ScalarVectorOps.fill(a, value);
        }
    }

    public static float[] normalize(float[] a) {
        if (USE_SIMD) {
            return SimdVectorOps.normalize(a);
        } else {
            return ScalarVectorOps.normalize(a);
        }
    }
}
