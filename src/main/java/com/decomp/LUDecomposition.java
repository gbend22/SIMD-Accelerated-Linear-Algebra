package com.decomp;

/**
 * Immutable result of an LU decomposition with partial pivoting of an
 * {@code n x n} matrix {@code A}, factoring the row-permuted matrix as
 * {@code P * A = L * U}.
 *
 * <p>{@code L} is unit lower-triangular and {@code U} is upper-triangular, both
 * held as row-major {@code n x n} {@code float[][]} arrays. The row permutation
 * {@code P} is stored as a pivot array: {@code pivot[i]} is the original row
 * index moved into row {@code i}. The pivot sign ({@code +1} or {@code -1})
 * records the parity of the row swaps and feeds the determinant.
 *
 * <p>All factors are defensively copied on construction and on every accessor,
 * so instances are immutable and safe to share.
 */
public final class LUDecomposition {

    private final float[][] l;
    private final float[][] u;
    private final int[] pivot;
    private final int pivotSign;

    /**
     * Wraps the factors of an LU decomposition, copying the arrays so later
     * changes to the caller's data cannot affect this instance.
     *
     * @param l         the unit lower-triangular factor
     * @param u         the upper-triangular factor
     * @param pivot     the row permutation, where {@code pivot[i]} is the source row for row {@code i}
     * @param pivotSign the sign ({@code +1} or {@code -1}) from the parity of row swaps
     */
    public LUDecomposition(float[][] l, float[][] u, int[] pivot, int pivotSign) {
        this.l = copy(l);
        this.u = copy(u);
        this.pivot = pivot.clone();
        this.pivotSign = pivotSign;
    }

    /**
     * Returns the dimension {@code n} of the decomposed matrix.
     *
     * @return the number of rows and columns
     */
    public int size() {
        return u.length;
    }

    /**
     * Returns a fresh copy of the unit lower-triangular factor {@code L}.
     *
     * @return a new copy of {@code L}
     */
    public float[][] getL() {
        return copy(l);
    }

    /**
     * Returns a fresh copy of the upper-triangular factor {@code U}.
     *
     * @return a new copy of {@code U}
     */
    public float[][] getU() {
        return copy(u);
    }

    /**
     * Returns a fresh copy of the pivot array, where {@code pivot[i]} is the
     * original row index permuted into row {@code i}.
     *
     * @return a new copy of the pivot array
     */
    public int[] getPivot() {
        return pivot.clone();
    }

    /**
     * Returns the pivot sign ({@code +1} or {@code -1}), the parity of the row
     * swaps applied during decomposition.
     *
     * @return the pivot sign
     */
    public int getPivotSign() {
        return pivotSign;
    }

    private static float[][] copy(float[][] m) {
        float[][] out = new float[m.length][];
        for (int i = 0; i < m.length; i++) {
            out[i] = m[i].clone();
        }
        return out;
    }
}
