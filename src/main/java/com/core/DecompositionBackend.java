package com.core;

import com.decomp.CholeskyDecomposition;
import com.decomp.LUDecomposition;
import com.decomp.QRDecomposition;

/**
 * Internal contract implemented by the scalar and SIMD decomposition backends. Provides
 * the matrix factorizations (LU, Cholesky, QR) and the LU-based solvers built on them.
 * The concrete implementation is chosen at runtime by {@link Dispatcher}; this type is
 * not part of the public API.
 */
public interface DecompositionBackend {
    LUDecomposition lu(float[][] matrix);
    CholeskyDecomposition cholesky(float[][] matrix);
    QRDecomposition qr(float[][] matrix);
    float[] solve(float[][] matrix, float[] b);
    float determinant(float[][] matrix);
    float[][] inverse(float[][] matrix);
}
