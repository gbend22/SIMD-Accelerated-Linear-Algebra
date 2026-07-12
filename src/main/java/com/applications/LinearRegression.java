package com.applications;

import com.matrix.MatrixOps;
import com.vector.VectorOps;

import java.util.Arrays;

public class LinearRegression {

    private final boolean fitIntercept;

    private float[] coefficients;
    private float intercept;
    private boolean fitted;

    public LinearRegression() {
        this(true);
    }

    public LinearRegression(boolean fitIntercept) {
        this.fitIntercept = fitIntercept;
    }

    public static LinearRegression fromParameters(float[] coefficients, float intercept) {
        if (coefficients.length == 0) {
            throw new IllegalArgumentException("Coefficients must not be empty");
        }
        LinearRegression model = new LinearRegression();
        model.coefficients = coefficients.clone();
        model.intercept = intercept;
        model.fitted = true;
        return model;
    }

    public void fit(float[][] x, float[] y) {
        if (x.length == 0) {
            throw new IllegalArgumentException("Training set must not be empty");
        }
        if (x.length != y.length) {
            throw new IllegalArgumentException(
                    "Sample count mismatch: " + x.length + " rows but " + y.length + " targets");
        }

        int features = x[0].length;
        for (float[] row : x) {
            if (row.length != features) {
                throw new IllegalArgumentException("All samples must have the same number of features");
            }
        }

        float[][] design = fitIntercept ? augment(x) : x;

        float[][] dt = MatrixOps.transpose(design);
        float[][] normal = MatrixOps.multiply(dt, design);
        float[] rhs = MatrixOps.multiply(dt, y);

        float[] beta = MatrixOps.solve(normal, rhs);

        if (fitIntercept) {
            intercept = beta[0];
            coefficients = Arrays.copyOfRange(beta, 1, beta.length);
        } else {
            intercept = 0f;
            coefficients = beta;
        }
        fitted = true;
    }

    public float predict(float[] sample) {
        requireFitted();
        if (sample.length != coefficients.length) {
            throw new IllegalArgumentException(
                    "Feature count mismatch: model has " + coefficients.length
                            + " but sample has " + sample.length);
        }
        return intercept + VectorOps.dot(coefficients, sample);
    }

    public float[] predict(float[][] x) {
        requireFitted();
        float[] predictions = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            predictions[i] = predict(x[i]);
        }
        return predictions;
    }

    public float[] coefficients() {
        requireFitted();
        return coefficients.clone();
    }

    public float intercept() {
        requireFitted();
        return intercept;
    }

    private void requireFitted() {
        if (!fitted) {
            throw new IllegalStateException("Model has not been fitted");
        }
    }

    private static float[][] augment(float[][] x) {
        int n = x.length;
        int features = x[0].length;
        float[][] design = new float[n][features + 1];
        for (int i = 0; i < n; i++) {
            design[i][0] = 1f;
            System.arraycopy(x[i], 0, design[i], 1, features);
        }
        return design;
    }
}
