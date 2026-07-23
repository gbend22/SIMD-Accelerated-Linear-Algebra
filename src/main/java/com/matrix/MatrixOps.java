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

    /**
     * Computes the LU decomposition with partial pivoting of a square matrix,
     * factoring the row-permuted matrix as {@code P * A = L * U}.
     *
     * @param matrix a square {@code n x n} matrix
     * @return the {@code L} and {@code U} factors together with the row permutation
     * @throws IllegalArgumentException if {@code matrix} is empty or not square
     */
    public static LUDecomposition lu(float[][] matrix) {
        return Dispatcher.lu(matrix);
    }

    /**
     * Computes the Cholesky decomposition of a symmetric positive-definite matrix,
     * factoring it as {@code A = L * Lᵀ}.
     *
     * @param matrix a symmetric positive-definite {@code n x n} matrix
     * @return the lower-triangular factor {@code L}
     * @throws IllegalArgumentException if {@code matrix} is empty or not square
     * @throws ArithmeticException      if {@code matrix} is not positive definite
     */
    public static CholeskyDecomposition cholesky(float[][] matrix) {
        return Dispatcher.cholesky(matrix);
    }

    /**
     * Computes the QR decomposition of a matrix with at least as many rows as columns,
     * factoring it as {@code A = Q * R} with {@code Q} orthogonal and {@code R}
     * upper-triangular.
     *
     * @param matrix an {@code m x n} matrix with {@code m >= n}
     * @return the orthogonal factor {@code Q} and the upper-triangular factor {@code R}
     * @throws IllegalArgumentException if {@code matrix} is empty, not rectangular,
     *         or has fewer rows than columns
     */
    public static QRDecomposition qr(float[][] matrix) {
        return Dispatcher.qr(matrix);
    }

    /**
     * Solves the linear system {@code A * x = b} for a square matrix {@code A}, using
     * LU decomposition with partial pivoting.
     *
     * @param matrix the square {@code n x n} coefficient matrix {@code A}
     * @param b      the right-hand side vector of length {@code n}
     * @return the solution vector {@code x} of length {@code n}
     * @throws IllegalArgumentException if {@code matrix} is empty or not square, or if
     *         {@code b.length} does not equal the matrix dimension
     * @throws ArithmeticException      if {@code matrix} is singular
     */
    public static float[] solve(float[][] matrix, float[] b) {
        return Dispatcher.solve(matrix, b);
    }

    /**
     * Computes the determinant of a square matrix via LU decomposition.
     *
     * @param matrix a square {@code n x n} matrix
     * @return the determinant of {@code matrix}
     * @throws IllegalArgumentException if {@code matrix} is empty or not square
     */
    public static float determinant(float[][] matrix) {
        return Dispatcher.determinant(matrix);
    }

    /**
     * Computes the inverse of a square matrix via LU decomposition.
     *
     * @param matrix a square {@code n x n} matrix
     * @return a new {@code n x n} matrix that is the inverse of {@code matrix}
     * @throws IllegalArgumentException if {@code matrix} is empty or not square
     * @throws ArithmeticException      if {@code matrix} is singular
     */
    public static float[][] inverse(float[][] matrix) {
        return Dispatcher.inverse(matrix);
    }
}
