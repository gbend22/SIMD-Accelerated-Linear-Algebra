package com.core;

public interface MatrixBackend {
    float[] multiply(float[][] matrix, float[] vector);
    float[][] add(float[][] a, float[][] b);
    float[][] multiply(float[][] a, float[][] b);
    float[][] transpose(float[][] matrix);
}
