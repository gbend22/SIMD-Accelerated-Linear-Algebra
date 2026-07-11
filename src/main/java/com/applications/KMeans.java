package com.applications;

import java.util.Random;

public class KMeans {

    private static final int DEFAULT_MAX_ITERATIONS = 100;
    private static final long DEFAULT_SEED = 42L;

    private final int k;
    private final int maxIterations;
    private final long seed;

    private float[][] centroids;
    private int[] assignments;
    private int iterations;
    private boolean fitted;

    public KMeans(int k) {
        this(k, DEFAULT_MAX_ITERATIONS, DEFAULT_SEED);
    }

    public KMeans(int k, int maxIterations, long seed) {
        if (k < 1) {
            throw new IllegalArgumentException("k must be at least 1, got " + k);
        }
        if (maxIterations < 1) {
            throw new IllegalArgumentException("maxIterations must be at least 1, got " + maxIterations);
        }
        this.k = k;
        this.maxIterations = maxIterations;
        this.seed = seed;
    }

    public void fit(float[][] x) {
        if (x.length == 0) {
            throw new IllegalArgumentException("Training set must not be empty");
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

        centroids = initializeCentroids(x);
        assignments = new int[x.length];
        iterations = 0;
        fitted = true;
    }

    private float[][] initializeCentroids(float[][] x) {
        Random random = new Random(seed);
        int n = x.length;
        boolean[] used = new boolean[n];
        float[][] result = new float[k][];
        for (int c = 0; c < k; c++) {
            int idx;
            do {
                idx = random.nextInt(n);
            } while (used[idx]);
            used[idx] = true;
            result[c] = x[idx].clone();
        }
        return result;
    }

    private void requireFitted() {
        if (!fitted) {
            throw new IllegalStateException("Model has not been fitted");
        }
    }
}
