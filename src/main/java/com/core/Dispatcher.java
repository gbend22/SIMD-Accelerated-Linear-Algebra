package com.core;

import com.scalar.ScalarMatrixOps;
import com.scalar.ScalarVectorOps;
import com.simd.SimdMatrixOps;
import com.simd.SimdVectorOps;

import java.util.logging.Logger;

public class Dispatcher {

    private static final Logger LOGGER = Logger.getLogger(Dispatcher.class.getName());

    private static final boolean USE_SIMD;

    static {

        int width = SimdVectorOps.simdWidth();

        USE_SIMD = width >= 4;

        if (USE_SIMD) {
            LOGGER.info(
                    "[SimdLinalg] SIMD enabled with "
                            + width +
                            " float lanes"
            );
        } else {
            LOGGER.info(
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

    public static float euclideanDistance(float[] a, float[] b) {
        if (USE_SIMD) {
            return SimdVectorOps.euclideanDistance(a, b);
        } else {
            return ScalarVectorOps.euclideanDistance(a, b);
        }
    }

    public static float[] fma(float[] a, float[] b, float[] c) {
        if (USE_SIMD) {
            return SimdVectorOps.fma(a, b, c);
        } else {
            return ScalarVectorOps.fma(a, b, c);
        }
    }

    public static int argmax(float[] a) {
        if (USE_SIMD) {
            return SimdVectorOps.argmax(a);
        } else {
            return ScalarVectorOps.argmax(a);
        }
    }

    public static float[] softmax(float[] a) {
        if (USE_SIMD) {
            return SimdVectorOps.softmax(a);
        } else {
            return ScalarVectorOps.softmax(a);
        }
    }

    public static float[][] add(float[][] a, float[][] b) {
        if (USE_SIMD) {
            return SimdMatrixOps.add(a, b);
        } else {
            return ScalarMatrixOps.add(a, b);
        }
    }

    public static float[] multiply(float[][] matrix, float[] vector) {
        if (USE_SIMD) {
            return SimdMatrixOps.multiply(matrix, vector);
        } else {
            return ScalarMatrixOps.multiply(matrix, vector);
        }
    }

    public static float[][] multiply(float[][] a, float[][] b) {
        if (USE_SIMD) {
            return SimdMatrixOps.multiply(a, b);
        } else {
            return ScalarMatrixOps.multiply(a, b);
        }
    }

    public static float[][] transpose(float[][] matrix) {
        if (USE_SIMD) {
            return SimdMatrixOps.transpose(matrix);
        } else {
            return ScalarMatrixOps.transpose(matrix);
        }
    }
}
