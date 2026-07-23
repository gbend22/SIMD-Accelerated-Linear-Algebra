package com.applications;

import com.matrix.MatrixOps;
import com.vector.VectorOps;

import java.util.Arrays;

/**
 * Ordinary least-squares linear regression built on the JavaSIMDLinalg matrix operations.
 * Fitting solves the normal equations {@code (XᵀX) β = Xᵀy} for the coefficients using the
 * library's matrix multiply and linear solver; by default an intercept is included by
 * augmenting the design matrix with a constant column.
 *
 * <p>Instances are stateful: call {@link #fit(float[][], float[])} (or build a model with
 * {@link #fromParameters(float[], float)}) before calling {@link #predict(float[])} or the
 * accessors. Not thread-safe.
 */
public class LinearRegression {

    private final boolean fitIntercept;

    private float[] coefficients;
    private float intercept;
    private boolean fitted;

    /**
     * Creates a model that fits an intercept term.
     */
    public LinearRegression() {
        this(true);
    }

    /**
     * Creates a model, optionally without an intercept term.
     *
     * @param fitIntercept whether to fit an intercept; if {@code false} the fitted
     *                     relationship passes through the origin
     */
    public LinearRegression(boolean fitIntercept) {
        this.fitIntercept = fitIntercept;
    }

    /**
     * Creates a pre-fitted model from known coefficients and intercept, skipping training.
     *
     * @param coefficients the per-feature coefficients
     * @param intercept    the intercept term
     * @return a fitted model ready for {@link #predict(float[])}
     * @throws IllegalArgumentException if {@code coefficients} is empty
     */
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

    /**
     * Fits the model to a training set by solving the normal equations.
     *
     * @param x the training samples, one row per sample, all with the same number of
     *          features
     * @param y the target value for each training sample
     * @throws IllegalArgumentException if {@code x} is empty, the sample and target counts
     *         differ, or the rows differ in length
     * @throws ArithmeticException      if the normal equations are singular (for example
     *         when features are perfectly collinear)
     */
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

    /**
     * Predicts the target value for a single sample as
     * {@code intercept + dot(coefficients, sample)}.
     *
     * @param sample the feature vector
     * @return the predicted value
     * @throws IllegalStateException    if the model has not been fitted
     * @throws IllegalArgumentException if {@code sample} has the wrong number of features
     */
    public float predict(float[] sample) {
        requireFitted();
        if (sample.length != coefficients.length) {
            throw new IllegalArgumentException(
                    "Feature count mismatch: model has " + coefficients.length
                            + " but sample has " + sample.length);
        }
        return intercept + VectorOps.dot(coefficients, sample);
    }

    /**
     * Predicts the target value for each sample in a batch.
     *
     * @param x the samples, one row per sample
     * @return an array of predictions, one per input row
     * @throws IllegalStateException    if the model has not been fitted
     * @throws IllegalArgumentException if any sample has the wrong number of features
     */
    public float[] predict(float[][] x) {
        requireFitted();
        float[] predictions = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            predictions[i] = predict(x[i]);
        }
        return predictions;
    }

    /**
     * Returns a fresh copy of the fitted per-feature coefficients.
     *
     * @return a new array of coefficients
     * @throws IllegalStateException if the model has not been fitted
     */
    public float[] coefficients() {
        requireFitted();
        return coefficients.clone();
    }

    /**
     * Returns the fitted intercept term ({@code 0} when the model was created without one).
     *
     * @return the intercept
     * @throws IllegalStateException if the model has not been fitted
     */
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
