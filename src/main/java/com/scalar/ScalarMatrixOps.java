package com.scalar;

import com.core.MatrixBackend;
import com.core.MatrixValidation;

/**
 * Scalar (plain-loop) implementation of {@link MatrixBackend}. Serves as the
 * correctness reference for the SIMD backend and as the fallback when SIMD is
 * unavailable. Internal; not part of the public API.
 */
public class ScalarMatrixOps implements MatrixBackend {

    @Override
    public float[] multiply(float[][] matrix, float[] vector) {
        int rows = matrix.length;
        int cols = MatrixValidation.requireRectangular(matrix, "matrix");

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

    @Override
    public float[][] add(float[][] a, float[][] b) {
        int rows = a.length;
        int cols = MatrixValidation.requireSameShape(a, b);

        float[][] result = new float[rows][cols];

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                result[r][c] = a[r][c] + b[r][c];
            }
        }

        return result;
    }

    @Override
    public float[][] transpose(float[][] matrix) {
        int rows = matrix.length;
        int cols = MatrixValidation.requireRectangular(matrix, "matrix");

        float[][] result = new float[cols][rows];

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                result[c][r] = matrix[r][c];
            }
        }

        return result;
    }

    @Override
    public float[][] multiply(float[][] a, float[][] b) {
        int inner = MatrixValidation.requireRectangular(a, "a");
        int cols = MatrixValidation.requireRectangular(b, "b");
        if (inner != b.length) {
            throw new IllegalArgumentException(
                    "Matrix dimensions do not allow multiplication"
            );
        }

        int rows = a.length;

        float[][] result = new float[rows][cols];

        for (int i = 0; i < rows; i++) {

            for (int j = 0; j < cols; j++) {

                float sum = 0f;

                for (int k = 0; k < inner; k++) {
                    sum += a[i][k] * b[k][j];
                }

                result[i][j] = sum;
            }
        }

        return result;
    }
}
