package com.performance;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import java.util.stream.IntStream;

/**
 * Register-tiled SIMD matrix multiply that spreads the row tiles across threads. Small
 * problems run on a single thread; once the work (rows &times; columns &times; depth)
 * exceeds a fixed threshold the tiles are distributed with a parallel stream. One of
 * several matrix-multiply strategies explored in {@code com.performance}; internal, not
 * part of the public API.
 */
public class ParallelTiledSimdMatrixOps {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int MR = 4;
    private static final long PARALLEL_THRESHOLD = 10_000_000L;

    public float[][] multiply(float[][] a, float[][] b) {
        if (a[0].length != b.length) {
            throw new IllegalArgumentException("Matrix dimensions do not allow multiplication");
        }

        int rows = a.length;
        int cols = b[0].length;
        int inner = b.length;

        float[][] result = new float[rows][cols];
        int blocks = (rows + MR - 1) / MR;

        if ((long) rows * cols * inner < PARALLEL_THRESHOLD) {
            for (int block = 0; block < blocks; block++) {
                multiplyBlock(a, b, result, block, rows, cols, inner);
            }
        } else {
            IntStream.range(0, blocks)
                    .parallel()
                    .forEach(block -> multiplyBlock(a, b, result, block, rows, cols, inner));
        }

        return result;
    }

    private static void multiplyBlock(float[][] a, float[][] b, float[][] result, int block, int rows, int cols, int inner) {
        int i = block * MR;
        int end = Math.min(i + MR, rows);
        int vlen = SPECIES.length();
        int colBound = SPECIES.loopBound(cols);

        if (end - i == MR) {
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
        } else {
            for (int r = i; r < end; r++) {
                float[] ar = a[r];
                float[] cr = result[r];
                int j = 0;
                for (; j < colBound; j += vlen) {
                    FloatVector acc = FloatVector.zero(SPECIES);
                    for (int k = 0; k < inner; k++) {
                        acc = FloatVector.broadcast(SPECIES, ar[k]).fma(FloatVector.fromArray(SPECIES, b[k], j), acc);
                    }
                    acc.intoArray(cr, j);
                }
                for (; j < cols; j++) {
                    float s = 0f;
                    for (int k = 0; k < inner; k++) {
                        s += ar[k] * b[k][j];
                    }
                    cr[j] = s;
                }
            }
        }
    }
}
