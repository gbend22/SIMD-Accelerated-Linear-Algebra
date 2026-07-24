package com.baseline;

import com.scalar.ScalarMatrixOps;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class NaiveSimdMatrixOpsTest {

    private final NaiveSimdMatrixOps naive = new NaiveSimdMatrixOps();
    private final ScalarMatrixOps scalar = new ScalarMatrixOps();

    @Test
    void multiply_matchesScalarForRectangularMatricesAndVectorTail() {
        Random random = new Random(42);
        float[][] a = randomMatrix(7, 19, random);
        float[][] b = randomMatrix(19, 11, random);

        float[][] expected = scalar.multiply(a, b);
        float[][] actual = naive.multiply(a, b);

        for (int row = 0; row < expected.length; row++) {
            assertArrayEquals(expected[row], actual[row], 1e-4f);
        }
    }

    @Test
    void multiply_rejectsRaggedAndIncompatibleMatrices() {
        assertThrows(IllegalArgumentException.class,
                () -> naive.multiply(new float[][]{{1, 2}, {3}}, new float[][]{{1}, {2}}));
        assertThrows(IllegalArgumentException.class,
                () -> naive.multiply(new float[][]{{1, 2}}, new float[][]{{1}, {2}, {3}}));
    }

    private static float[][] randomMatrix(int rows, int columns, Random random) {
        float[][] matrix = new float[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                matrix[i][j] = random.nextFloat() * 2f - 1f;
            }
        }
        return matrix;
    }
}
