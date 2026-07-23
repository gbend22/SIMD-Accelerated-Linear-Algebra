package com.decomp;

/**
 * Immutable result of a QR decomposition of an {@code m x n} matrix {@code A}
 * (with {@code m >= n}), factoring it as {@code A = Q * R}.
 *
 * <p>{@code Q} is an {@code m x m} orthogonal matrix ({@code Qᵀ * Q = I}) and
 * {@code R} is an {@code m x n} upper-triangular matrix. Both are held as
 * row-major {@code float[][]} arrays.
 *
 * <p>The factors are defensively copied on construction and on every accessor,
 * so instances are immutable and safe to share.
 */
public final class QRDecomposition {

    private final float[][] q;
    private final float[][] r;

    /**
     * Wraps the two factors of a QR decomposition, copying both so later changes
     * to the caller's arrays cannot affect this instance.
     *
     * @param q the {@code m x m} orthogonal factor
     * @param r the {@code m x n} upper-triangular factor
     */
    public QRDecomposition(float[][] q, float[][] r) {
        this.q = copy(q);
        this.r = copy(r);
    }

    /**
     * Returns the row count {@code m} of the decomposed matrix.
     *
     * @return the number of rows
     */
    public int rows() {
        return q.length;
    }

    /**
     * Returns the column count {@code n} of the decomposed matrix.
     *
     * @return the number of columns, or {@code 0} if {@code R} has no rows
     */
    public int columns() {
        return r.length == 0 ? 0 : r[0].length;
    }

    /**
     * Returns a fresh copy of the {@code m x m} orthogonal factor {@code Q}.
     *
     * @return a new copy of {@code Q}
     */
    public float[][] getQ() {
        return copy(q);
    }

    /**
     * Returns a fresh copy of the {@code m x n} upper-triangular factor {@code R}.
     *
     * @return a new copy of {@code R}
     */
    public float[][] getR() {
        return copy(r);
    }

    private static float[][] copy(float[][] m) {
        float[][] out = new float[m.length][];
        for (int i = 0; i < m.length; i++) {
            out[i] = m[i].clone();
        }
        return out;
    }
}
