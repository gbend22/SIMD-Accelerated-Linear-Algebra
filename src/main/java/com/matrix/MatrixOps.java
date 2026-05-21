package com.matrix;

import com.core.Dispatcher;

public class MatrixOps {

    private MatrixOps() {}

    public static float[][] add(float[][] a, float[][] b) {
        return Dispatcher.add(a, b);
    }

    public static float[] multiply(float[][] matrix, float[] vector) {
        return Dispatcher.multiply(matrix, vector);
    }

    public static float[][] multiply(float[][] a, float[][] b) {
        return Dispatcher.multiply(a, b);
    }

    public static float[][] transpose(float[][] matrix) {
        return Dispatcher.transpose(matrix);
    }
}
