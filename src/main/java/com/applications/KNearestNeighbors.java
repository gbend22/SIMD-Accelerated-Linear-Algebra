package com.applications;

import com.vector.VectorOps;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * K-nearest-neighbours classifier built on the JavaSIMDLinalg vector operations. Training
 * simply stores the labelled samples; a prediction ranks the stored samples by Euclidean
 * distance to the query and takes a majority vote among the {@code k} closest, breaking
 * ties towards the smaller label.
 *
 * <p>Instances are stateful: call {@link #fit(float[][], int[])} (or build a fitted model
 * with {@link #fromTrainingData(int, float[][], int[])}) before calling
 * {@link #predict(float[])}. Not thread-safe.
 */
public class KNearestNeighbors {

    private final int k;

    private float[][] trainingFeatures;
    private int[] trainingLabels;
    private boolean fitted;

    /**
     * Creates a classifier that votes among the {@code k} nearest neighbours.
     *
     * @param k the number of neighbours to consult per prediction
     * @throws IllegalArgumentException if {@code k} is less than {@code 1}
     */
    public KNearestNeighbors(int k) {
        if (k < 1) {
            throw new IllegalArgumentException("k must be at least 1, got " + k);
        }
        this.k = k;
    }

    /**
     * Creates a classifier and immediately fits it to the given labelled data.
     *
     * @param k the number of neighbours to consult per prediction
     * @param x the training samples, one row per sample
     * @param y the label for each training sample
     * @return a fitted classifier ready for {@link #predict(float[])}
     * @throws IllegalArgumentException if {@code k} is less than {@code 1}, the data is
     *         empty, the sample and label counts differ, {@code k} exceeds the sample
     *         count, or the rows differ in length
     */
    public static KNearestNeighbors fromTrainingData(int k, float[][] x, int[] y) {
        KNearestNeighbors model = new KNearestNeighbors(k);
        model.fit(x, y);
        return model;
    }

    /**
     * Stores the labelled training set used for subsequent predictions.
     *
     * @param x the training samples, one row per sample, all with the same number of
     *          features
     * @param y the label for each training sample
     * @throws IllegalArgumentException if {@code x} is empty, the sample and label counts
     *         differ, {@code k} exceeds the sample count, or the rows differ in length
     */
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

    /**
     * Classifies a single sample by majority vote among its {@code k} nearest neighbours.
     * Ties in the vote are broken in favour of the smaller label.
     *
     * @param sample the feature vector to classify
     * @return the predicted label
     * @throws IllegalStateException    if the model has not been fitted
     * @throws IllegalArgumentException if {@code sample} has the wrong number of features
     */
    public int predict(float[] sample) {
        requireFitted();
        if (sample.length != trainingFeatures[0].length) {
            throw new IllegalArgumentException(
                    "Feature count mismatch: model has " + trainingFeatures[0].length
                            + " but sample has " + sample.length);
        }

        int n = trainingFeatures.length;
        float[] distances = new float[n];
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            distances[i] = VectorOps.euclideanDistance(trainingFeatures[i], sample);
            order[i] = i;
        }
        Arrays.sort(order, (p, q) -> Float.compare(distances[p], distances[q]));

        Map<Integer, Integer> votes = new HashMap<>();
        for (int rank = 0; rank < k; rank++) {
            int label = trainingLabels[order[rank]];
            votes.merge(label, 1, Integer::sum);
        }

        int bestLabel = Integer.MAX_VALUE;
        int bestCount = -1;
        for (Map.Entry<Integer, Integer> vote : votes.entrySet()) {
            int label = vote.getKey();
            int count = vote.getValue();
            if (count > bestCount || (count == bestCount && label < bestLabel)) {
                bestCount = count;
                bestLabel = label;
            }
        }
        return bestLabel;
    }

    /**
     * Classifies each sample in a batch.
     *
     * @param x the samples to classify, one row per sample
     * @return an array of predicted labels, one per input row
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

    /**
     * Returns the number of neighbours consulted per prediction.
     *
     * @return the neighbour count {@code k}
     */
    public int k() {
        return k;
    }

    /**
     * Returns the number of stored training samples.
     *
     * @return the training set size
     * @throws IllegalStateException if the model has not been fitted
     */
    public int trainingSize() {
        requireFitted();
        return trainingLabels.length;
    }

    private void requireFitted() {
        if (!fitted) {
            throw new IllegalStateException("Model has not been fitted");
        }
    }
}
