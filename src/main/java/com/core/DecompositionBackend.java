package com.core;

import com.decomp.LUDecomposition;

public interface DecompositionBackend {
    LUDecomposition lu(float[][] matrix);
    float[] solve(float[][] matrix, float[] b);
    float determinant(float[][] matrix);
}
