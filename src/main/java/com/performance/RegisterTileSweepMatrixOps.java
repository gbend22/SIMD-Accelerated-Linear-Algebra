package com.performance;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

public class RegisterTileSweepMatrixOps {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    public float[][] multiply(float[][] a, float[][] b, int mr) {
        if (a[0].length != b.length) {
            throw new IllegalArgumentException("Matrix dimensions do not allow multiplication");
        }

        int rows = a.length;
        int cols = b[0].length;
        int inner = b.length;
        int vlen = SPECIES.length();

        return switch (mr) {
            case 1 -> multiplyMr1(a, b, rows, cols, inner, vlen);
            case 2 -> multiplyMr2(a, b, rows, cols, inner, vlen);
            case 4 -> multiplyMr4(a, b, rows, cols, inner, vlen);
            case 6 -> multiplyMr6(a, b, rows, cols, inner, vlen);
            case 8 -> multiplyMr8(a, b, rows, cols, inner, vlen);
            default -> throw new IllegalArgumentException("Unsupported mr: " + mr);
        };
    }

    private static float[][] multiplyMr1(float[][] a, float[][] b, int rows, int cols, int inner, int vlen) {
        float[][] result = new float[rows][cols];
        multiplyRows1(a, b, result, 0, rows, cols, inner, vlen);
        return result;
    }

    private static float[][] multiplyMr2(float[][] a, float[][] b, int rows, int cols, int inner, int vlen) {
        float[][] result = new float[rows][cols];
        int colBound = SPECIES.loopBound(cols);

        int i = 0;
        for (; i + 2 <= rows; i += 2) {
            float[] a0 = a[i];
            float[] a1 = a[i + 1];
            float[] c0 = result[i];
            float[] c1 = result[i + 1];

            int j = 0;
            for (; j < colBound; j += vlen) {
                FloatVector acc0 = FloatVector.zero(SPECIES);
                FloatVector acc1 = FloatVector.zero(SPECIES);
                for (int k = 0; k < inner; k++) {
                    FloatVector vb = FloatVector.fromArray(SPECIES, b[k], j);
                    acc0 = FloatVector.broadcast(SPECIES, a0[k]).fma(vb, acc0);
                    acc1 = FloatVector.broadcast(SPECIES, a1[k]).fma(vb, acc1);
                }
                acc0.intoArray(c0, j);
                acc1.intoArray(c1, j);
            }
            for (; j < cols; j++) {
                float s0 = 0f;
                float s1 = 0f;
                for (int k = 0; k < inner; k++) {
                    float bv = b[k][j];
                    s0 += a0[k] * bv;
                    s1 += a1[k] * bv;
                }
                c0[j] = s0;
                c1[j] = s1;
            }
        }
        multiplyRows1(a, b, result, i, rows, cols, inner, vlen);
        return result;
    }

    private static float[][] multiplyMr4(float[][] a, float[][] b, int rows, int cols, int inner, int vlen) {
        float[][] result = new float[rows][cols];
        int colBound = SPECIES.loopBound(cols);

        int i = 0;
        for (; i + 4 <= rows; i += 4) {
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
        multiplyRows1(a, b, result, i, rows, cols, inner, vlen);
        return result;
    }

    private static float[][] multiplyMr6(float[][] a, float[][] b, int rows, int cols, int inner, int vlen) {
        float[][] result = new float[rows][cols];
        int colBound = SPECIES.loopBound(cols);

        int i = 0;
        for (; i + 6 <= rows; i += 6) {
            float[] a0 = a[i];
            float[] a1 = a[i + 1];
            float[] a2 = a[i + 2];
            float[] a3 = a[i + 3];
            float[] a4 = a[i + 4];
            float[] a5 = a[i + 5];
            float[] c0 = result[i];
            float[] c1 = result[i + 1];
            float[] c2 = result[i + 2];
            float[] c3 = result[i + 3];
            float[] c4 = result[i + 4];
            float[] c5 = result[i + 5];

            int j = 0;
            for (; j < colBound; j += vlen) {
                FloatVector acc0 = FloatVector.zero(SPECIES);
                FloatVector acc1 = FloatVector.zero(SPECIES);
                FloatVector acc2 = FloatVector.zero(SPECIES);
                FloatVector acc3 = FloatVector.zero(SPECIES);
                FloatVector acc4 = FloatVector.zero(SPECIES);
                FloatVector acc5 = FloatVector.zero(SPECIES);
                for (int k = 0; k < inner; k++) {
                    FloatVector vb = FloatVector.fromArray(SPECIES, b[k], j);
                    acc0 = FloatVector.broadcast(SPECIES, a0[k]).fma(vb, acc0);
                    acc1 = FloatVector.broadcast(SPECIES, a1[k]).fma(vb, acc1);
                    acc2 = FloatVector.broadcast(SPECIES, a2[k]).fma(vb, acc2);
                    acc3 = FloatVector.broadcast(SPECIES, a3[k]).fma(vb, acc3);
                    acc4 = FloatVector.broadcast(SPECIES, a4[k]).fma(vb, acc4);
                    acc5 = FloatVector.broadcast(SPECIES, a5[k]).fma(vb, acc5);
                }
                acc0.intoArray(c0, j);
                acc1.intoArray(c1, j);
                acc2.intoArray(c2, j);
                acc3.intoArray(c3, j);
                acc4.intoArray(c4, j);
                acc5.intoArray(c5, j);
            }
            for (; j < cols; j++) {
                float s0 = 0f;
                float s1 = 0f;
                float s2 = 0f;
                float s3 = 0f;
                float s4 = 0f;
                float s5 = 0f;
                for (int k = 0; k < inner; k++) {
                    float bv = b[k][j];
                    s0 += a0[k] * bv;
                    s1 += a1[k] * bv;
                    s2 += a2[k] * bv;
                    s3 += a3[k] * bv;
                    s4 += a4[k] * bv;
                    s5 += a5[k] * bv;
                }
                c0[j] = s0;
                c1[j] = s1;
                c2[j] = s2;
                c3[j] = s3;
                c4[j] = s4;
                c5[j] = s5;
            }
        }
        multiplyRows1(a, b, result, i, rows, cols, inner, vlen);
        return result;
    }

    private static float[][] multiplyMr8(float[][] a, float[][] b, int rows, int cols, int inner, int vlen) {
        float[][] result = new float[rows][cols];
        int colBound = SPECIES.loopBound(cols);

        int i = 0;
        for (; i + 8 <= rows; i += 8) {
            float[] a0 = a[i];
            float[] a1 = a[i + 1];
            float[] a2 = a[i + 2];
            float[] a3 = a[i + 3];
            float[] a4 = a[i + 4];
            float[] a5 = a[i + 5];
            float[] a6 = a[i + 6];
            float[] a7 = a[i + 7];
            float[] c0 = result[i];
            float[] c1 = result[i + 1];
            float[] c2 = result[i + 2];
            float[] c3 = result[i + 3];
            float[] c4 = result[i + 4];
            float[] c5 = result[i + 5];
            float[] c6 = result[i + 6];
            float[] c7 = result[i + 7];

            int j = 0;
            for (; j < colBound; j += vlen) {
                FloatVector acc0 = FloatVector.zero(SPECIES);
                FloatVector acc1 = FloatVector.zero(SPECIES);
                FloatVector acc2 = FloatVector.zero(SPECIES);
                FloatVector acc3 = FloatVector.zero(SPECIES);
                FloatVector acc4 = FloatVector.zero(SPECIES);
                FloatVector acc5 = FloatVector.zero(SPECIES);
                FloatVector acc6 = FloatVector.zero(SPECIES);
                FloatVector acc7 = FloatVector.zero(SPECIES);
                for (int k = 0; k < inner; k++) {
                    FloatVector vb = FloatVector.fromArray(SPECIES, b[k], j);
                    acc0 = FloatVector.broadcast(SPECIES, a0[k]).fma(vb, acc0);
                    acc1 = FloatVector.broadcast(SPECIES, a1[k]).fma(vb, acc1);
                    acc2 = FloatVector.broadcast(SPECIES, a2[k]).fma(vb, acc2);
                    acc3 = FloatVector.broadcast(SPECIES, a3[k]).fma(vb, acc3);
                    acc4 = FloatVector.broadcast(SPECIES, a4[k]).fma(vb, acc4);
                    acc5 = FloatVector.broadcast(SPECIES, a5[k]).fma(vb, acc5);
                    acc6 = FloatVector.broadcast(SPECIES, a6[k]).fma(vb, acc6);
                    acc7 = FloatVector.broadcast(SPECIES, a7[k]).fma(vb, acc7);
                }
                acc0.intoArray(c0, j);
                acc1.intoArray(c1, j);
                acc2.intoArray(c2, j);
                acc3.intoArray(c3, j);
                acc4.intoArray(c4, j);
                acc5.intoArray(c5, j);
                acc6.intoArray(c6, j);
                acc7.intoArray(c7, j);
            }
            for (; j < cols; j++) {
                float s0 = 0f;
                float s1 = 0f;
                float s2 = 0f;
                float s3 = 0f;
                float s4 = 0f;
                float s5 = 0f;
                float s6 = 0f;
                float s7 = 0f;
                for (int k = 0; k < inner; k++) {
                    float bv = b[k][j];
                    s0 += a0[k] * bv;
                    s1 += a1[k] * bv;
                    s2 += a2[k] * bv;
                    s3 += a3[k] * bv;
                    s4 += a4[k] * bv;
                    s5 += a5[k] * bv;
                    s6 += a6[k] * bv;
                    s7 += a7[k] * bv;
                }
                c0[j] = s0;
                c1[j] = s1;
                c2[j] = s2;
                c3[j] = s3;
                c4[j] = s4;
                c5[j] = s5;
                c6[j] = s6;
                c7[j] = s7;
            }
        }
        multiplyRows1(a, b, result, i, rows, cols, inner, vlen);
        return result;
    }

    private static void multiplyRows1(float[][] a, float[][] b, float[][] result,
                                      int rowStart, int rows, int cols, int inner, int vlen) {
        int colBound = SPECIES.loopBound(cols);
        for (int i = rowStart; i < rows; i++) {
            float[] ar = a[i];
            float[] cr = result[i];
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
