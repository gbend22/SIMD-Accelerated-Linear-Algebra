package com.core;

import java.util.Objects;

/**
 * Shared validation for the row-major {@code float[][]} matrices used by the library.
 * Internal; public operations expose the resulting argument errors through their own
 * contracts.
 */
public final class MatrixValidation {

    private static final float SYMMETRY_RELATIVE_TOLERANCE = 1e-5f;

    private MatrixValidation() {}

    /**
     * Verifies that a matrix is non-empty, has at least one column, and has no null or
     * ragged rows.
     *
     * @param matrix the matrix to validate
     * @param name   argument name used in exception messages
     * @return the common row length
     */
    public static int requireRectangular(float[][] matrix, String name) {
        Objects.requireNonNull(matrix, name + " must not be null");
        if (matrix.length == 0) {
            throw new IllegalArgumentException(name + " must not be empty");
        }

        Objects.requireNonNull(matrix[0], name + " row 0 must not be null");
        int columns = matrix[0].length;
        if (columns == 0) {
            throw new IllegalArgumentException(name + " must have at least one column");
        }

        for (int row = 1; row < matrix.length; row++) {
            Objects.requireNonNull(matrix[row], name + " row " + row + " must not be null");
            if (matrix[row].length != columns) {
                throw new IllegalArgumentException(
                        name + " must be rectangular: row 0 has " + columns
                                + " columns but row " + row + " has " + matrix[row].length);
            }
        }
        return columns;
    }

    /**
     * Verifies that two matrices are rectangular and have identical dimensions.
     *
     * @return their common column count
     */
    public static int requireSameShape(float[][] a, float[][] b) {
        int aColumns = requireRectangular(a, "a");
        int bColumns = requireRectangular(b, "b");
        if (a.length != b.length || aColumns != bColumns) {
            throw new IllegalArgumentException(
                    "Matrix dimensions must match, got "
                            + a.length + "x" + aColumns + " and "
                            + b.length + "x" + bColumns);
        }
        return aColumns;
    }

    /**
     * Verifies that a matrix is non-empty, rectangular, and square.
     *
     * @return its dimension
     */
    public static int requireSquare(float[][] matrix, String name) {
        int columns = requireRectangular(matrix, name);
        if (matrix.length != columns) {
            throw new IllegalArgumentException(
                    name + " must be square, got " + matrix.length + "x" + columns);
        }
        return matrix.length;
    }

    /**
     * Verifies that a matrix is square, finite, and symmetric within a small
     * scale-relative floating-point tolerance.
     *
     * @return its dimension
     */
    public static int requireSymmetric(float[][] matrix, String name) {
        int n = requireSquare(matrix, name);
        for (int i = 0; i < n; i++) {
            if (!Float.isFinite(matrix[i][i])) {
                throw new IllegalArgumentException(
                        name + " must contain only finite values; found "
                                + matrix[i][i] + " at (" + i + "," + i + ")");
            }
            for (int j = 0; j < i; j++) {
                float lower = matrix[i][j];
                float upper = matrix[j][i];
                if (!Float.isFinite(lower) || !Float.isFinite(upper)) {
                    throw new IllegalArgumentException(
                            name + " must contain only finite values at symmetric positions ("
                                    + i + "," + j + ") and (" + j + "," + i + ")");
                }
                float scale = Math.max(1f, Math.max(Math.abs(lower), Math.abs(upper)));
                if (Math.abs(lower - upper) > SYMMETRY_RELATIVE_TOLERANCE * scale) {
                    throw new IllegalArgumentException(
                            name + " must be symmetric: values at (" + i + "," + j
                                    + ") and (" + j + "," + i + ") differ ("
                                    + lower + " vs " + upper + ")");
                }
            }
        }
        return n;
    }
}
