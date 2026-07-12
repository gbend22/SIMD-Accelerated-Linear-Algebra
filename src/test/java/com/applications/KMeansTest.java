package com.applications;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class KMeansTest {

    private static final float DELTA = 1e-3f;

    private static float[][] twoClusters() {
        return new float[][]{
                {0, 0}, {1, 0}, {0, 1}, {1, 1},
                {10, 10}, {11, 10}, {10, 11}, {11, 11}
        };
    }

    private static boolean hasCentroidNear(float[][] centroids, float[] target) {
        for (float[] centroid : centroids) {
            if (Math.abs(centroid[0] - target[0]) < DELTA
                    && Math.abs(centroid[1] - target[1]) < DELTA) {
                return true;
            }
        }
        return false;
    }

    @Test
    void fit_separatedClusters_groupSharesLabel() {
        float[][] x = twoClusters();

        KMeans model = new KMeans(2);
        model.fit(x);

        int first = model.predict(x[0]);
        assertEquals(first, model.predict(x[1]));
        assertEquals(first, model.predict(x[2]));
        assertEquals(first, model.predict(x[3]));

        int second = model.predict(x[4]);
        assertEquals(second, model.predict(x[5]));
        assertEquals(second, model.predict(x[6]));
        assertEquals(second, model.predict(x[7]));

        assertNotEquals(first, second);
    }

    @Test
    void centroids_areNearTrueClusterMeans() {
        float[][] x = twoClusters();

        KMeans model = new KMeans(2);
        model.fit(x);

        float[][] centroids = model.centroids();
        assertTrue(hasCentroidNear(centroids, new float[]{0.5f, 0.5f}));
        assertTrue(hasCentroidNear(centroids, new float[]{10.5f, 10.5f}));
    }

    @Test
    void predict_assignsNewPointToNearestCluster() {
        float[][] x = twoClusters();

        KMeans model = new KMeans(2);
        model.fit(x);

        assertEquals(model.predict(x[0]), model.predict(new float[]{0.2f, 0.3f}));
        assertEquals(model.predict(x[4]), model.predict(new float[]{10.2f, 9.8f}));
    }

    @Test
    void predict_batchMatchesSingle() {
        float[][] x = twoClusters();

        KMeans model = new KMeans(2);
        model.fit(x);

        float[][] queries = {{0.5f, 0.5f}, {10.5f, 10.5f}, {2, 2}};
        int[] batch = model.predict(queries);
        for (int i = 0; i < queries.length; i++) {
            assertEquals(model.predict(queries[i]), batch[i]);
        }
    }

    @Test
    void fit_convergesBeforeMaxIterations() {
        float[][] x = twoClusters();

        KMeans model = new KMeans(2);
        model.fit(x);

        assertTrue(model.iterations() >= 1);
        assertTrue(model.iterations() < 100);
    }

    @Test
    void fit_isReproducibleWithFixedSeed() {
        float[][] x = twoClusters();

        KMeans first = new KMeans(2);
        first.fit(x);
        KMeans second = new KMeans(2);
        second.fit(x);

        float[][] a = first.centroids();
        float[][] b = second.centroids();
        assertEquals(a.length, b.length);
        for (int c = 0; c < a.length; c++) {
            assertArrayEquals(a[c], b[c], 0f);
        }
    }

    @Test
    void k_returnsConfiguredValue() {
        assertEquals(3, new KMeans(3).k());
    }

    @Test
    void constructor_kBelowOne_throws() {
        assertThrows(IllegalArgumentException.class, () -> new KMeans(0));
    }

    @Test
    void constructor_maxIterationsBelowOne_throws() {
        assertThrows(IllegalArgumentException.class, () -> new KMeans(2, 0, 42L));
    }

    @Test
    void fit_emptyTrainingSet_throws() {
        KMeans model = new KMeans(1);
        assertThrows(IllegalArgumentException.class, () -> model.fit(new float[0][0]));
    }

    @Test
    void fit_kLargerThanTrainingSet_throws() {
        float[][] x = {{0, 0}, {1, 1}};

        KMeans model = new KMeans(3);
        assertThrows(IllegalArgumentException.class, () -> model.fit(x));
    }

    @Test
    void fit_raggedRows_throws() {
        float[][] x = {{0, 0}, {1}};

        KMeans model = new KMeans(1);
        assertThrows(IllegalArgumentException.class, () -> model.fit(x));
    }

    @Test
    void predict_beforeFit_throws() {
        KMeans model = new KMeans(2);
        assertThrows(IllegalStateException.class, () -> model.predict(new float[]{1, 2}));
    }

    @Test
    void predict_featureCountMismatch_throws() {
        float[][] x = {{0, 0}, {1, 1}, {2, 2}};

        KMeans model = new KMeans(2);
        model.fit(x);

        assertThrows(IllegalArgumentException.class, () -> model.predict(new float[]{1, 2, 3}));
    }

    @Test
    void centroids_beforeFit_throws() {
        KMeans model = new KMeans(2);
        assertThrows(IllegalStateException.class, model::centroids);
    }
}
