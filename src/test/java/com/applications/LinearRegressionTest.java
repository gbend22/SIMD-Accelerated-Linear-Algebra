package com.applications;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class LinearRegressionTest {

    private static final float DELTA = 1e-2f;

    @Test
    void fit_recoversKnownLine() {
        float[][] x = {{0}, {1}, {2}, {3}, {4}};
        float[] y = {1, 4, 7, 10, 13};

        LinearRegression model = new LinearRegression();
        model.fit(x, y);

        assertArrayEquals(new float[]{3f}, model.coefficients(), DELTA);
        assertEquals(1f, model.intercept(), DELTA);
    }

    @Test
    void fit_recoversKnownPlane() {
        float[][] x = {{1, 1}, {2, 1}, {1, 2}, {3, 2}, {2, 3}, {4, 0}};
        float[] y = {10, 12, 13, 17, 18, 13};

        LinearRegression model = new LinearRegression();
        model.fit(x, y);

        assertArrayEquals(new float[]{2f, 3f}, model.coefficients(), DELTA);
        assertEquals(5f, model.intercept(), DELTA);
    }

    @Test
    void predict_matchesUnderlyingFunction() {
        float[][] x = {{1, 1}, {2, 1}, {1, 2}, {3, 2}, {2, 3}, {4, 0}};
        float[] y = {10, 12, 13, 17, 18, 13};

        LinearRegression model = new LinearRegression();
        model.fit(x, y);

        assertEquals(2f * 5f + 3f * 4f + 5f, model.predict(new float[]{5, 4}), DELTA);
    }

    @Test
    void predict_batchMatchesSingle() {
        float[][] x = {{0}, {1}, {2}, {3}, {4}};
        float[] y = {1, 4, 7, 10, 13};

        LinearRegression model = new LinearRegression();
        model.fit(x, y);

        float[][] query = {{5}, {6}, {7}};
        float[] batch = model.predict(query);
        for (int i = 0; i < query.length; i++) {
            assertEquals(model.predict(query[i]), batch[i], DELTA);
        }
    }

    @Test
    void fit_noInterceptFitsThroughOrigin() {
        float[][] x = {{1}, {2}, {3}, {4}};
        float[] y = {2, 4, 6, 8};

        LinearRegression model = new LinearRegression(false);
        model.fit(x, y);

        assertArrayEquals(new float[]{2f}, model.coefficients(), DELTA);
        assertEquals(0f, model.intercept(), DELTA);
    }

    @Test
    void fit_mismatchedSampleCount_throws() {
        float[][] x = {{1}, {2}};
        float[] y = {1, 2, 3};

        LinearRegression model = new LinearRegression();
        assertThrows(IllegalArgumentException.class, () -> model.fit(x, y));
    }

    @Test
    void fit_emptyTrainingSet_throws() {
        float[][] x = new float[0][0];
        float[] y = new float[0];

        LinearRegression model = new LinearRegression();
        assertThrows(IllegalArgumentException.class, () -> model.fit(x, y));
    }

    @Test
    void fit_singularDesign_throws() {
        float[][] x = {{1, 2}, {2, 4}, {3, 6}};
        float[] y = {1, 2, 3};

        LinearRegression model = new LinearRegression();
        assertThrows(ArithmeticException.class, () -> model.fit(x, y));
    }

    @Test
    void predict_beforeFit_throws() {
        LinearRegression model = new LinearRegression();
        assertThrows(IllegalStateException.class, () -> model.predict(new float[]{1, 2}));
    }
}
