package com.core;

import com.decomp.CholeskyDecomposition;
import com.decomp.LUDecomposition;
import com.decomp.QRDecomposition;

public interface DecompositionBackend {
    LUDecomposition lu(float[][] matrix);
    CholeskyDecomposition cholesky(float[][] matrix);
    QRDecomposition qr(float[][] matrix);
    float[] solve(float[][] matrix, float[] b);
    float determinant(float[][] matrix);
    float[][] inverse(float[][] matrix);
}
