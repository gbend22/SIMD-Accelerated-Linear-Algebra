package com.core;

import com.scalar.ScalarMatrixOps;
import com.scalar.ScalarVectorOps;
import com.simd.SimdMatrixOps;
import com.simd.SimdVectorOps;

import java.util.logging.Logger;

public class Dispatcher {

    private static final Logger LOGGER = Logger.getLogger(Dispatcher.class.getName());

    private static final boolean USE_SIMD;
    private static final VectorBackend VECTOR_BACKEND;

    static {

        int width = SimdVectorOps.simdWidth();

        USE_SIMD = width >= 4;

        if (USE_SIMD) {
            LOGGER.info("[SimdLinalg] SIMD enabled with " + width + " float lanes");
            VECTOR_BACKEND = new SimdVectorOps();
        } else {
            LOGGER.info("[SimdLinalg] SIMD unavailable, falling back to scalar backend");
            VECTOR_BACKEND = new ScalarVectorOps();
        }
    }

    private Dispatcher() {}

    public static float dot(float[] a, float[] b) {
        return VECTOR_BACKEND.dot(a, b);
    }

    public static float norm(float[] a) {
        return VECTOR_BACKEND.norm(a);
    }

    public static float cosineSimilarity(float[] a, float[] b) {
        return VECTOR_BACKEND.cosineSimilarity(a, b);
    }

    public static float[] add(float[] a, float[] b) {
        return VECTOR_BACKEND.add(a, b);
    }

    public static float[] subtract(float[] a, float[] b) {
        return VECTOR_BACKEND.subtract(a, b);
    }

    public static float[] multiply(float[] a, float[] b) {
        return VECTOR_BACKEND.multiply(a, b);
    }

    public static float[] divide(float[] a, float[] b) {
        return VECTOR_BACKEND.divide(a, b);
    }

    public static float sum(float[] a) {
        return VECTOR_BACKEND.sum(a);
    }

    public static float min(float[] a) {
        return VECTOR_BACKEND.min(a);
    }

    public static float max(float[] a) {
        return VECTOR_BACKEND.max(a);
    }

    public static float[] scale(float[] a, float scalar) {
        return VECTOR_BACKEND.scale(a, scalar);
    }

    public static float[] copy(float[] a) {
        return VECTOR_BACKEND.copy(a);
    }

    public static void fill(float[] a, float value) {
        VECTOR_BACKEND.fill(a, value);
    }

    public static float[] normalize(float[] a) {
        return VECTOR_BACKEND.normalize(a);
    }

    public static float euclideanDistance(float[] a, float[] b) {
        return VECTOR_BACKEND.euclideanDistance(a, b);
    }

    public static float[] fma(float[] a, float[] b, float[] c) {
        return VECTOR_BACKEND.fma(a, b, c);
    }

    public static int argmax(float[] a) {
        return VECTOR_BACKEND.argmax(a);
    }

    public static float[] softmax(float[] a) {
        return VECTOR_BACKEND.softmax(a);
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
