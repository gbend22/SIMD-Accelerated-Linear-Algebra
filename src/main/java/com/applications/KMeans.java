package com.applications;

import com.vector.VectorOps;

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

    public static KMeans fromCentroids(float[][] centroids) {
        if (centroids.length == 0) {
            throw new IllegalArgumentException("Centroids must not be empty");
        }
        int features = centroids[0].length;
        for (float[] centroid : centroids) {
            if (centroid.length != features) {
                throw new IllegalArgumentException("All centroids must have the same number of features");
            }
        }
        KMeans model = new KMeans(centroids.length);
        float[][] copy = new float[centroids.length][];
        for (int c = 0; c < centroids.length; c++) {
            copy[c] = centroids[c].clone();
        }
        model.centroids = copy;
        model.iterations = 0;
        model.fitted = true;
        return model;
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

        for (iterations = 0; iterations < maxIterations; iterations++) {
            boolean changed = assignPoints(x);
            if (!changed) {
                break;
            }
            updateCentroids(x);
        }
        fitted = true;
    }

    public int predict(float[] sample) {
        requireFitted();
        if (sample.length != centroids[0].length) {
            throw new IllegalArgumentException(
                    "Feature count mismatch: model has " + centroids[0].length
                            + " but sample has " + sample.length);
        }
        return nearestCentroid(sample);
    }

    public int[] predict(float[][] x) {
        requireFitted();
        int[] predictions = new int[x.length];
        for (int i = 0; i < x.length; i++) {
            predictions[i] = predict(x[i]);
        }
        return predictions;
    }

    private boolean assignPoints(float[][] x) {
        boolean changed = false;
        for (int i = 0; i < x.length; i++) {
            int nearest = nearestCentroid(x[i]);
            if (nearest != assignments[i]) {
                assignments[i] = nearest;
                changed = true;
            }
        }
        return changed;
    }

    private int nearestCentroid(float[] sample) {
        int best = 0;
        float bestDistance = VectorOps.euclideanDistance(centroids[0], sample);
        for (int c = 1; c < centroids.length; c++) {
            float distance = VectorOps.euclideanDistance(centroids[c], sample);
            if (distance < bestDistance) {
                bestDistance = distance;
                best = c;
            }
        }
        return best;
    }

    private void updateCentroids(float[][] x) {
        int features = x[0].length;
        float[][] sums = new float[k][features];
        int[] counts = new int[k];
        for (int i = 0; i < x.length; i++) {
            int c = assignments[i];
            sums[c] = VectorOps.add(sums[c], x[i]);
            counts[c]++;
        }
        for (int c = 0; c < k; c++) {
            if (counts[c] > 0) {
                centroids[c] = VectorOps.scale(sums[c], 1f / counts[c]);
            }
        }
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

    public float[][] centroids() {
        requireFitted();
        float[][] copy = new float[centroids.length][];
        for (int c = 0; c < centroids.length; c++) {
            copy[c] = centroids[c].clone();
        }
        return copy;
    }

    public int k() {
        return k;
    }

    public int iterations() {
        requireFitted();
        return iterations;
    }

    private void requireFitted() {
        if (!fitted) {
            throw new IllegalStateException("Model has not been fitted");
        }
    }
}
