package com.performance;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Register-tiled SIMD matrix multiply with an added layer of cache blocking. The product
 * is built over {@code KC x NC} panels of {@code B} so each panel stays resident in cache
 * while it is reused across the rows of {@code A}, with partial results accumulated into
 * {@code C}. One of several matrix-multiply strategies explored in {@code com.performance};
 * internal, not part of the public API.
 */
public class CacheBlockedTiledSimdMatrixOps {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int MR = 4;
    private static final int KC = 128;
    private static final int NC = 128;

    public float[][] multiply(float[][] a, float[][] b) {
        if (a[0].length != b.length) {
            throw new IllegalArgumentException("Matrix dimensions do not allow multiplication");
        }

        int rows = a.length;
        int cols = b[0].length;
        int inner = b.length;
        int vlen = SPECIES.length();

        float[][] result = new float[rows][cols];

        for (int jc = 0; jc < cols; jc += NC) {
            int jcEnd = Math.min(jc + NC, cols);
            int colBound = jc + SPECIES.loopBound(jcEnd - jc);
            for (int pc = 0; pc < inner; pc += KC) {
                int pcEnd = Math.min(pc + KC, inner);

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

                    int j = jc;
                    for (; j < colBound; j += vlen) {
                        FloatVector acc0 = FloatVector.fromArray(SPECIES, c0, j);
                        FloatVector acc1 = FloatVector.fromArray(SPECIES, c1, j);
                        FloatVector acc2 = FloatVector.fromArray(SPECIES, c2, j);
                        FloatVector acc3 = FloatVector.fromArray(SPECIES, c3, j);
                        for (int k = pc; k < pcEnd; k++) {
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
                    for (; j < jcEnd; j++) {
                        float s0 = c0[j];
                        float s1 = c1[j];
                        float s2 = c2[j];
                        float s3 = c3[j];
                        for (int k = pc; k < pcEnd; k++) {
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
                    float[] ar = a[i];
                    float[] cr = result[i];
                    int j = jc;
                    for (; j < colBound; j += vlen) {
                        FloatVector acc = FloatVector.fromArray(SPECIES, cr, j);
                        for (int k = pc; k < pcEnd; k++) {
                            acc = FloatVector.broadcast(SPECIES, ar[k]).fma(FloatVector.fromArray(SPECIES, b[k], j), acc);
                        }
                        acc.intoArray(cr, j);
                    }
                    for (; j < jcEnd; j++) {
                        float s = cr[j];
                        for (int k = pc; k < pcEnd; k++) {
                            s += ar[k] * b[k][j];
                        }
                        cr[j] = s;
                    }
                }
            }
        }

        return result;
    }
}
