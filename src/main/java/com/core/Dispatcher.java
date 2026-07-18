package com.core;

import com.decomp.CholeskyDecomposition;
import com.decomp.LUDecomposition;
import com.scalar.ScalarDecompositionOps;
import com.scalar.ScalarMatrixOps;
import com.scalar.ScalarVectorOps;
import com.simd.SimdDecompositionOps;
import com.simd.SimdMatrixOps;
import com.simd.SimdVectorOps;

import java.util.logging.Logger;

/**
 * Internal runtime dispatcher. Detects the available SIMD vector width once at class
 * load, then binds the {@link VectorBackend} and {@link MatrixBackend} implementations
 * the library uses, falling back to the scalar backends when SIMD is unavailable.
 *
 * <p>Not part of the public API &mdash; application code should call
 * {@link com.vector.VectorOps} and {@link com.matrix.MatrixOps} instead.
 */
public class Dispatcher {

    private static final Logger LOGGER = Logger.getLogger(Dispatcher.class.getName());

    private static final VectorBackend VECTOR_BACKEND;
    private static final MatrixBackend MATRIX_BACKEND;
    private static final DecompositionBackend DECOMPOSITION_BACKEND;

    static {

        int width = SimdVectorOps.simdWidth();
        boolean useSIMD = width >= 4;

        if (useSIMD) {
            LOGGER.info("[SimdLinalg] SIMD enabled with " + width + " float lanes");
            VECTOR_BACKEND = new SimdVectorOps();
            MATRIX_BACKEND = new SimdMatrixOps();
            DECOMPOSITION_BACKEND = new SimdDecompositionOps();
        } else {
            LOGGER.info("[SimdLinalg] SIMD unavailable, falling back to scalar backend");
            VECTOR_BACKEND = new ScalarVectorOps();
            MATRIX_BACKEND = new ScalarMatrixOps();
            DECOMPOSITION_BACKEND = new ScalarDecompositionOps();
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
        return MATRIX_BACKEND.add(a, b);
    }

    public static float[] multiply(float[][] matrix, float[] vector) {
        return MATRIX_BACKEND.multiply(matrix, vector);
    }

    public static float[][] multiply(float[][] a, float[][] b) {
        return MATRIX_BACKEND.multiply(a, b);
    }

    public static float[][] transpose(float[][] matrix) {
        return MATRIX_BACKEND.transpose(matrix);
    }

    public static LUDecomposition lu(float[][] matrix) {
        return DECOMPOSITION_BACKEND.lu(matrix);
    }

    public static CholeskyDecomposition cholesky(float[][] matrix) {
        return DECOMPOSITION_BACKEND.cholesky(matrix);
    }

    public static float[] solve(float[][] matrix, float[] b) {
        return DECOMPOSITION_BACKEND.solve(matrix, b);
    }

    public static float determinant(float[][] matrix) {
        return DECOMPOSITION_BACKEND.determinant(matrix);
    }

    public static float[][] inverse(float[][] matrix) {
        return DECOMPOSITION_BACKEND.inverse(matrix);
    }
}
