package com.decomp;

/**
 * Immutable result of a Cholesky decomposition of a symmetric positive-definite
 * {@code n x n} matrix {@code A}, factoring it as {@code A = L * Lᵀ}.
 *
 * <p>{@code L} is the {@code n x n} lower-triangular factor, held as a row-major
 * {@code float[][]} array with zeros above the diagonal.
 *
 * <p>The factor is defensively copied on construction and on every accessor, so
 * instances are immutable and safe to share.
 */
public final class CholeskyDecomposition {

    private final float[][] l;

    /**
     * Wraps the lower-triangular factor of a Cholesky decomposition, copying it so
     * later changes to the caller's array cannot affect this instance.
     *
     * @param l the {@code n x n} lower-triangular factor
     */
    public CholeskyDecomposition(float[][] l) {
        this.l = copy(l);
    }

    /**
     * Returns the dimension {@code n} of the decomposed matrix.
     *
     * @return the number of rows and columns
     */
    public int size() {
        return l.length;
    }

    /**
     * Returns a fresh copy of the {@code n x n} lower-triangular factor {@code L}.
     *
     * @return a new copy of {@code L}
     */
    public float[][] getL() {
        return copy(l);
    }

    private static float[][] copy(float[][] m) {
        float[][] out = new float[m.length][];
        for (int i = 0; i < m.length; i++) {
            out[i] = m[i].clone();
        }
        return out;
    }
}
