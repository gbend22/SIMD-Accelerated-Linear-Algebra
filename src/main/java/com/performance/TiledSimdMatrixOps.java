package com.performance;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Register-tiled SIMD matrix multiply. Processes four rows of {@code A} at a time and
 * accumulates each output tile with broadcast-and-FMA over contiguous columns of
 * {@code B}, so no transpose or column gathering is needed. One of several
 * matrix-multiply strategies explored in {@code com.performance}; internal, not part of
 * the public API.
 */
public class TiledSimdMatrixOps {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int MR = 4;

    public float[][] multiply(float[][] a, float[][] b) {
        if (a[0].length != b.length) {
            throw new IllegalArgumentException("Matrix dimensions do not allow multiplication");
        }

        int rows = a.length;
        int cols = b[0].length;
        int inner = b.length;

        int vlen = SPECIES.length();
        int colBound = SPECIES.loopBound(cols);

        float[][] result = new float[rows][cols];

        int i = 0;
        for (; i + MR <= rows; i += MR) {
            float[] a0 = a[i];
            float[] a1 = a[i + 1];
            float[] a2 = a[i + 2];
            float[] a3 = a[i + 3];
            float[] c0 = result[i];
            float[] c1 = result[i + 1];
            float[] c2 = result[i + 2];
            float[] c3 = result[i + 3];

            int j = 0;
            for (; j < colBound; j += vlen) {
                FloatVector acc0 = FloatVector.zero(SPECIES);
                FloatVector acc1 = FloatVector.zero(SPECIES);
                FloatVector acc2 = FloatVector.zero(SPECIES);
                FloatVector acc3 = FloatVector.zero(SPECIES);
                for (int k = 0; k < inner; k++) {
                    FloatVector vb = FloatVector.fromArray(SPECIES, b[k], j);
                    acc0 = FloatVector.broadcast(SPECIES, a0[k]).fma(vb, acc0);
                    acc1 = FloatVector.broadcast(SPECIES, a1[k]).fma(vb, acc1);
                    acc2 = FloatVector.broadcast(SPECIES, a2[k]).fma(vb, acc2);
                    acc3 = FloatVector.broadcast(SPECIES, a3[k]).fma(vb, acc3);
                }
                acc0.intoArray(c0, j);
                acc1.intoArray(c1, j);
                acc2.intoArray(c2, j);
                acc3.intoArray(c3, j);
            }
            for (; j < cols; j++) {
                float s0 = 0f;
                float s1 = 0f;
                float s2 = 0f;
                float s3 = 0f;
                for (int k = 0; k < inner; k++) {
                    float bv = b[k][j];
                    s0 += a0[k] * bv;
                    s1 += a1[k] * bv;
                    s2 += a2[k] * bv;
                    s3 += a3[k] * bv;
                }
                c0[j] = s0;
                c1[j] = s1;
                c2[j] = s2;
                c3[j] = s3;
            }
        }

        for (; i < rows; i++) {
            float[] ai = a[i];
            float[] ci = result[i];
            int j = 0;
            for (; j < colBound; j += vlen) {
                FloatVector acc = FloatVector.zero(SPECIES);
                for (int k = 0; k < inner; k++) {
                    acc = FloatVector.broadcast(SPECIES, ai[k]).fma(FloatVector.fromArray(SPECIES, b[k], j), acc);
                }
                acc.intoArray(ci, j);
            }
            for (; j < cols; j++) {
                float s = 0f;
                for (int k = 0; k < inner; k++) {
                    s += ai[k] * b[k][j];
                }
                ci[j] = s;
            }
        }

        return result;
    }
}
