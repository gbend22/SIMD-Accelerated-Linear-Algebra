package com.applications;

public class KNearestNeighbors {

    private final int k;

    private float[][] trainingFeatures;
    private int[] trainingLabels;
    private boolean fitted;

    public KNearestNeighbors(int k) {
        if (k < 1) {
            throw new IllegalArgumentException("k must be at least 1, got " + k);
        }
        this.k = k;
    }

    public void fit(float[][] x, int[] y) {
        if (x.length == 0) {
            throw new IllegalArgumentException("Training set must not be empty");
        }
        if (x.length != y.length) {
            throw new IllegalArgumentException(
                    "Sample count mismatch: " + x.length + " rows but " + y.length + " labels");
        }
        if (k > x.length) {
            throw new IllegalArgumentException(
                    "k (" + k + ") must not exceed the training set size (" + x.length + ")");
        }

        int features = x[0].length;
        for (float[] row : x) {
            if (row.length != features) {
                throw new IllegalArgumentException("All samples must have the same number of features");
            }
        }

        trainingFeatures = new float[x.length][];
        for (int i = 0; i < x.length; i++) {
            trainingFeatures[i] = x[i].clone();
        }
        trainingLabels = y.clone();
        fitted = true;
    }

    private void requireFitted() {
        if (!fitted) {
            throw new IllegalStateException("Model has not been fitted");
        }
    }
}
