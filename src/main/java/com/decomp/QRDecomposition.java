package com.decomp;

public final class QRDecomposition {

    private final float[][] q;
    private final float[][] r;

    public QRDecomposition(float[][] q, float[][] r) {
        this.q = copy(q);
        this.r = copy(r);
    }

    public int rows() {
        return q.length;
    }

    public int columns() {
        return r.length == 0 ? 0 : r[0].length;
    }

    public float[][] getQ() {
        return copy(q);
    }

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
