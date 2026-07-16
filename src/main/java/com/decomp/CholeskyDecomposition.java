package com.decomp;

public final class CholeskyDecomposition {

    private final float[][] l;

    public CholeskyDecomposition(float[][] l) {
        this.l = copy(l);
    }

    public int size() {
        return l.length;
    }

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
