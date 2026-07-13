package com.applications;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class KNearestNeighborsTest {

    @Test
    void predict_k1_returnsLabelOfNearestPoint() {
        float[][] x = {{0, 0}, {0, 2}, {2, 0}, {2, 2}};
        int[] y = {0, 1, 1, 1};

        KNearestNeighbors model = new KNearestNeighbors(1);
        model.fit(x, y);

        assertEquals(0, model.predict(new float[]{0.5f, 0.5f}));
    }

    @Test
    void predict_k3_majorityVoteOverridesNearestPoint() {
        float[][] x = {{0, 0}, {0, 2}, {2, 0}, {2, 2}};
        int[] y = {0, 1, 1, 1};

        KNearestNeighbors model = new KNearestNeighbors(3);
        model.fit(x, y);

        assertEquals(1, model.predict(new float[]{0.5f, 0.5f}));
    }

    @Test
    void predict_separatedClusters() {
        float[][] x = {{0, 0}, {1, 0}, {0, 1}, {10, 10}, {9, 10}, {10, 9}};
        int[] y = {0, 0, 0, 1, 1, 1};

        KNearestNeighbors model = new KNearestNeighbors(3);
        model.fit(x, y);

        assertEquals(0, model.predict(new float[]{0.5f, 0.5f}));
        assertEquals(1, model.predict(new float[]{9.5f, 9.5f}));
    }

    @Test
    void predict_tieBreaksToSmallerLabel() {
        float[][] x = {{0, 0}, {0, 2}, {2, 0}, {2, 2}};
        int[] y = {0, 1, 1, 1};

        KNearestNeighbors model = new KNearestNeighbors(2);
        model.fit(x, y);

        assertEquals(0, model.predict(new float[]{0.5f, 0.5f}));
    }

    @Test
    void predict_batchMatchesSingle() {
        float[][] x = {{0, 0}, {1, 0}, {0, 1}, {10, 10}, {9, 10}, {10, 9}};
        int[] y = {0, 0, 0, 1, 1, 1};

        KNearestNeighbors model = new KNearestNeighbors(3);
        model.fit(x, y);

        float[][] queries = {{0.5f, 0.5f}, {9.5f, 9.5f}, {1, 1}};
        int[] batch = model.predict(queries);
        for (int i = 0; i < queries.length; i++) {
            assertEquals(model.predict(queries[i]), batch[i]);
        }
    }

    @Test
    void accessors_reportConfiguredKAndTrainingSize() {
        float[][] x = {{0, 0}, {1, 0}, {0, 1}};
        int[] y = {0, 0, 1};

        KNearestNeighbors model = new KNearestNeighbors(2);
        model.fit(x, y);

        assertEquals(2, model.k());
        assertEquals(3, model.trainingSize());
    }

    @Test
    void constructor_kBelowOne_throws() {
        assertThrows(IllegalArgumentException.class, () -> new KNearestNeighbors(0));
    }

    @Test
    void fit_mismatchedSampleCount_throws() {
        float[][] x = {{0, 0}, {1, 0}};
        int[] y = {0, 0, 1};

        KNearestNeighbors model = new KNearestNeighbors(1);
        assertThrows(IllegalArgumentException.class, () -> model.fit(x, y));
    }

    @Test
    void fit_emptyTrainingSet_throws() {
        float[][] x = new float[0][0];
        int[] y = new int[0];

        KNearestNeighbors model = new KNearestNeighbors(1);
        assertThrows(IllegalArgumentException.class, () -> model.fit(x, y));
    }

    @Test
    void fit_kLargerThanTrainingSet_throws() {
        float[][] x = {{0, 0}, {1, 0}};
        int[] y = {0, 1};

        KNearestNeighbors model = new KNearestNeighbors(3);
        assertThrows(IllegalArgumentException.class, () -> model.fit(x, y));
    }

    @Test
    void predict_featureCountMismatch_throws() {
        float[][] x = {{0, 0}, {1, 0}, {0, 1}};
        int[] y = {0, 0, 1};

        KNearestNeighbors model = new KNearestNeighbors(1);
        model.fit(x, y);

        assertThrows(IllegalArgumentException.class, () -> model.predict(new float[]{1, 2, 3}));
    }

    @Test
    void predict_beforeFit_throws() {
        KNearestNeighbors model = new KNearestNeighbors(1);
        assertThrows(IllegalStateException.class, () -> model.predict(new float[]{1, 2}));
    }

    @Test
    void fromTrainingData_buildsFittedModel() {
        float[][] x = {{0, 0}, {1, 0}, {0, 1}, {10, 10}, {9, 10}, {10, 9}};
        int[] y = {0, 0, 0, 1, 1, 1};

        KNearestNeighbors model = KNearestNeighbors.fromTrainingData(3, x, y);

        assertEquals(3, model.k());
        assertEquals(6, model.trainingSize());
        assertEquals(0, model.predict(new float[]{0.5f, 0.5f}));
        assertEquals(1, model.predict(new float[]{9.5f, 9.5f}));
    }

    @Test
    void fromTrainingData_invalidInput_throws() {
        float[][] x = {{0, 0}, {1, 0}};
        int[] y = {0, 0, 1};

        assertThrows(IllegalArgumentException.class,
                () -> KNearestNeighbors.fromTrainingData(1, x, y));
    }
}
