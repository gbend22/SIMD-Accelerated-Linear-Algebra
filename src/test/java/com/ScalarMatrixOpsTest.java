package com;

import com.scalar.ScalarMatrixOps;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ScalarMatrixOpsTest {

    @Test
    void multiplyMatrixVector_2x3() {

        float[][] matrix = {
                {1, 2, 3},
                {4, 5, 6}
        };

        float[] vector = {10, 20, 30};

        float[] expected = {
                140,
                320
        };

        assertArrayEquals(
                expected,
                ScalarMatrixOps.multiplyMatrixVector(matrix, vector)
        );
    }

    @Test
    void add_2x2() {

        float[][] a = {
                {1, 2},
                {3, 4}
        };

        float[][] b = {
                {10, 20},
                {30, 40}
        };

        float[][] result = ScalarMatrixOps.add(a, b);

        assertArrayEquals(new float[]{11, 22}, result[0]);
        assertArrayEquals(new float[]{33, 44}, result[1]);
    }

    @Test
    void transpose_2x3() {

        float[][] matrix = {
                {1, 2, 3},
                {4, 5, 6}
        };

        float[][] result = ScalarMatrixOps.transpose(matrix);

        assertArrayEquals(new float[]{1, 4}, result[0]);
        assertArrayEquals(new float[]{2, 5}, result[1]);
        assertArrayEquals(new float[]{3, 6}, result[2]);
    }
}