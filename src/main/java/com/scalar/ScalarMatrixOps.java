package com.scalar;

public class ScalarMatrixOps {

    private static void checkSameDimensions(float[][] a, float[][] b) {

        if (a.length != b.length || a[0].length != b[0].length) {
            throw new IllegalArgumentException("Matrix dimensions must match");
        }
    }

    public static float[] multiplyMatrixVector(float[][] matrix, float[] vector) {

        int rows = matrix.length;
        int cols = matrix[0].length;

        if (vector.length != cols) {
            throw new IllegalArgumentException(
                    "Vector length must match matrix column count"
            );
        }

        float[] result = new float[rows];

        for (int r = 0; r < rows; r++) {

            float sum = 0f;

            for (int c = 0; c < cols; c++) {
                sum += matrix[r][c] * vector[c];
            }

            result[r] = sum;
        }

        return result;
    }

    public static float[][] add(float[][] a, float[][] b) {

        checkSameDimensions(a, b);

        int rows = a.length;
        int cols = a[0].length;

        float[][] result = new float[rows][cols];

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                result[r][c] = a[r][c] + b[r][c];
            }
        }

        return result;
    }

    public static float[][] transpose(float[][] matrix) {

        int rows = matrix.length;
        int cols = matrix[0].length;

        float[][] result = new float[cols][rows];

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                result[c][r] = matrix[r][c];
            }
        }

        return result;
    }
}