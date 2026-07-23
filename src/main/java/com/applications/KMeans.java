package com.applications;

import com.vector.VectorOps;

import java.util.Random;

/**
 * K-means clustering built on the JavaSIMDLinalg vector operations. Partitions samples
 * into {@code k} clusters with Lloyd's algorithm: assign each sample to its nearest
 * centroid (by Euclidean distance), move each centroid to the mean of its members, and
 * repeat until the assignments stop changing or an iteration cap is reached. Initial
 * centroids are distinct random samples chosen from a fixed seed, so a given input clusters
 * reproducibly.
 *
 * <p>Instances are stateful: call {@link #fit(float[][])} (or build a pre-fitted model with
 * {@link #fromCentroids(float[][])}) before calling {@link #predict(float[])} or the
 * accessors. Not thread-safe.
 */
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

    /**
     * Creates a model that finds {@code k} clusters, using the default iteration cap and
     * seed.
     *
     * @param k the number of clusters to form
     * @throws IllegalArgumentException if {@code k} is less than {@code 1}
     */
    public KMeans(int k) {
        this(k, DEFAULT_MAX_ITERATIONS, DEFAULT_SEED);
    }

    /**
     * Creates a model with full control over the clustering parameters.
     *
     * @param k             the number of clusters to form
     * @param maxIterations the maximum number of assign/update iterations
     * @param seed          the seed for the random initial-centroid selection
     * @throws IllegalArgumentException if {@code k} or {@code maxIterations} is less than
     *         {@code 1}
     */
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

    /**
     * Creates a pre-fitted model directly from known centroids, skipping training. Useful
     * for reusing centroids learned elsewhere. The resulting model reports {@code 0}
     * iterations.
     *
     * @param centroids the cluster centroids, one row per cluster, all of equal length
     * @return a fitted model ready for {@link #predict(float[])}
     * @throws IllegalArgumentException if {@code centroids} is empty or its rows differ in
     *         length
     */
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

    /**
     * Fits the model to a training set, learning the cluster centroids in place.
     *
     * @param x the training samples, one row per sample, all with the same number of
     *          features
     * @throws IllegalArgumentException if {@code x} is empty, {@code k} exceeds the number
     *         of samples, or the rows differ in length
     */
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

    /**
     * Assigns a single sample to the index of its nearest centroid.
     *
     * @param sample the feature vector to classify
     * @return the index of the closest cluster, in {@code [0, k)}
     * @throws IllegalStateException    if the model has not been fitted
     * @throws IllegalArgumentException if {@code sample} has the wrong number of features
     */
    public int predict(float[] sample) {
        requireFitted();
        if (sample.length != centroids[0].length) {
            throw new IllegalArgumentException(
                    "Feature count mismatch: model has " + centroids[0].length
                            + " but sample has " + sample.length);
        }
        return nearestCentroid(sample);
    }

    /**
     * Assigns each sample in a batch to its nearest centroid.
     *
     * @param x the samples to classify, one row per sample
     * @return an array of cluster indices, one per input row
     * @throws IllegalStateException    if the model has not been fitted
     * @throws IllegalArgumentException if any sample has the wrong number of features
     */
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

    /**
     * Returns a fresh copy of the learned centroids.
     *
     * @return a new array holding the cluster centroids, one row per cluster
     * @throws IllegalStateException if the model has not been fitted
     */
    public float[][] centroids() {
        requireFitted();
        float[][] copy = new float[centroids.length][];
        for (int c = 0; c < centroids.length; c++) {
            copy[c] = centroids[c].clone();
        }
        return copy;
    }

    /**
     * Returns the number of clusters this model forms.
     *
     * @return the cluster count {@code k}
     */
    public int k() {
        return k;
    }

    /**
     * Returns the number of iterations performed by the last {@link #fit(float[][])}.
     *
     * @return the iteration count ({@code 0} for a model built from known centroids)
     * @throws IllegalStateException if the model has not been fitted
     */
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
