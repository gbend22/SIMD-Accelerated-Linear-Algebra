package com.matrix;

import com.core.Dispatcher;
import com.decomp.CholeskyDecomposition;
import com.decomp.LUDecomposition;
import com.decomp.QRDecomposition;

/**
 * Public entry point for single-precision ({@code float}) matrix operations.
 *
 * <p>Like {@link com.vector.VectorOps}, this façade delegates to a SIMD or scalar
 * backend chosen once at JVM start-up, transparently to the caller. Matrices are
 * represented as row-major {@code float[][]} arrays: {@code m[r][c]} is the element in
 * row {@code r} and column {@code c}, and every row is expected to have the same length
 * (a rectangular matrix). Arguments are expected to be non-{@code null}.
 *
 * <p>Every method allocates and returns a new result; inputs are never modified.
 * This class is stateless, thread-safe, and cannot be instantiated.
 *
 * @since 1.0
 */
public class MatrixOps {

    private MatrixOps() {}

    /**
     * Computes the element-wise sum of two matrices of identical shape.
     *
     * @param a the first matrix
     * @param b the second matrix, with the same number of rows and columns as {@code a}
     * @return a new matrix where each element is {@code a[r][c] + b[r][c]}
     * @throws IllegalArgumentException if the matrices differ in shape
     */
    public static float[][] add(float[][] a, float[][] b) {
        return Dispatcher.add(a, b);
    }

    /**
     * Multiplies a matrix by a column vector, producing {@code matrix * vector}.
     *
     * @param matrix an {@code r x c} matrix
     * @param vector a vector of length {@code c}
     * @return a new vector of length {@code r}, where element {@code i} is the dot
     *         product of row {@code i} with {@code vector}
     * @throws IllegalArgumentException if {@code vector.length} does not equal the
     *         number of columns of {@code matrix}
     */
    public static float[] multiply(float[][] matrix, float[] vector) {
        return Dispatcher.multiply(matrix, vector);
    }

    /**
     * Computes the matrix product {@code a * b}.
     *
     * @param a an {@code m x n} matrix
     * @param b an {@code n x p} matrix, whose row count equals the column count of {@code a}
     * @return a new {@code m x p} matrix
     * @throws IllegalArgumentException if the column count of {@code a} does not equal
     *         the row count of {@code b}
     */
    public static float[][] multiply(float[][] a, float[][] b) {
        return Dispatcher.multiply(a, b);
    }

    /**
     * Returns the transpose of a matrix: an {@code r x c} input becomes a
     * {@code c x r} output where {@code result[c][r] == matrix[r][c]}.
     *
     * @param matrix the matrix to transpose
     * @return a new transposed matrix
     */
    public static float[][] transpose(float[][] matrix) {
        return Dispatcher.transpose(matrix);
    }

    public static LUDecomposition lu(float[][] matrix) {
        return Dispatcher.lu(matrix);
    }

    public static CholeskyDecomposition cholesky(float[][] matrix) {
        return Dispatcher.cholesky(matrix);
    }

    public static QRDecomposition qr(float[][] matrix) {
        return Dispatcher.qr(matrix);
    }

    public static float[] solve(float[][] matrix, float[] b) {
        return Dispatcher.solve(matrix, b);
    }

    public static float determinant(float[][] matrix) {
        return Dispatcher.determinant(matrix);
    }

    public static float[][] inverse(float[][] matrix) {
        return Dispatcher.inverse(matrix);
    }
}
