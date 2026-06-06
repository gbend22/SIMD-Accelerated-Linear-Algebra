package com.core;

/**
 * Internal contract implemented by the scalar and SIMD matrix backends. The concrete
 * implementation is chosen at runtime by {@link Dispatcher}; this type is not part of
 * the public API.
 */
public interface MatrixBackend {
    float[] multiply(float[][] matrix, float[] vector);
    float[][] add(float[][] a, float[][] b);
    float[][] multiply(float[][] a, float[][] b);
    float[][] transpose(float[][] matrix);
}
