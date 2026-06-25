package com.decomp;

public final class LUDecomposition {

    private final float[][] l;
    private final float[][] u;
    private final int[] pivot;
    private final int pivotSign;

    public LUDecomposition(float[][] l, float[][] u, int[] pivot, int pivotSign) {
        this.l = copy(l);
        this.u = copy(u);
        this.pivot = pivot.clone();
        this.pivotSign = pivotSign;
    }

    public int size() {
        return u.length;
    }

    public float[][] getL() {
        return copy(l);
    }

    public float[][] getU() {
        return copy(u);
    }

    public int[] getPivot() {
        return pivot.clone();
    }

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
