package com.applications;

import com.vector.VectorOps;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

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

    public int[] predict(float[][] x) {
        requireFitted();
        int[] predictions = new int[x.length];
        for (int i = 0; i < x.length; i++) {
            predictions[i] = predict(x[i]);
        }
        return predictions;
    }

    private void requireFitted() {
        if (!fitted) {
            throw new IllegalStateException("Model has not been fitted");
        }
    }
}
