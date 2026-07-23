package com.vector;

import com.core.Dispatcher;

/**
 * Public entry point for single-precision ({@code float}) vector operations.
 *
 * <p>This is the façade of the JavaSIMDLinalg library. Each method delegates to a
 * backend that is selected once at JVM start-up: a SIMD backend built on the Java
 * Vector API when the running CPU exposes a usable vector width, or a plain scalar
 * backend otherwise. Backend selection is transparent; callers always use the
 * same methods and receive numerically equivalent results regardless of which backend
 * is active.
 *
 * <p>All operations work on caller-supplied {@code float[]} arrays. Methods that return
 * a vector allocate and return a new array; the only in-place method is {@link #fill}.
 * Arguments are expected to be non-{@code null}; passing {@code null} results in a
 * {@link NullPointerException}.
 *
 * <p>This class is stateless and therefore thread-safe. It cannot be instantiated.
 *
 * @since 1.0
 */
public class VectorOps {

    private VectorOps() {}

    /**
     * Computes the dot (inner) product of two vectors,
     * {@code a[0]*b[0] + a[1]*b[1] + ... + a[n-1]*b[n-1]}.
     *
     * @param a the first vector
     * @param b the second vector, of the same length as {@code a}
     * @return the scalar dot product
     * @throws IllegalArgumentException if the vectors differ in length
     */
    public static float dot(float[] a, float[] b) {
        return Dispatcher.dot(a, b);
    }

    /**
     * Computes the Euclidean (L2) norm of a vector, {@code sqrt(sum of squares)}.
     *
     * @param a the vector
     * @return the L2 norm; {@code 0} for an empty vector
     */
    public static float norm(float[] a) {
        return Dispatcher.norm(a);
    }

    /**
     * Computes the cosine similarity between two vectors,
     * {@code dot(a, b) / (norm(a) * norm(b))}.
     *
     * @param a the first vector
     * @param b the second vector, of the same length as {@code a}
     * @return the cosine similarity, in the range {@code [-1, 1]} for finite inputs
     * @throws IllegalArgumentException if the vectors differ in length,
     *         or if either vector has zero norm
     */
    public static float cosineSimilarity(float[] a, float[] b) {
        return Dispatcher.cosineSimilarity(a, b);
    }

    /**
     * Computes the element-wise sum of two vectors.
     *
     * @param a the first vector
     * @param b the second vector, of the same length as {@code a}
     * @return a new vector where each element is {@code a[i] + b[i]}
     * @throws IllegalArgumentException if the vectors differ in length
     */
    public static float[] add(float[] a, float[] b) {
        return Dispatcher.add(a, b);
    }

    /**
     * Computes the element-wise difference of two vectors.
     *
     * @param a the minuend vector
     * @param b the subtrahend vector, of the same length as {@code a}
     * @return a new vector where each element is {@code a[i] - b[i]}
     * @throws IllegalArgumentException if the vectors differ in length
     */
    public static float[] subtract(float[] a, float[] b) {
        return Dispatcher.subtract(a, b);
    }

    /**
     * Computes the element-wise (Hadamard) product of two vectors.
     *
     * @param a the first vector
     * @param b the second vector, of the same length as {@code a}
     * @return a new vector where each element is {@code a[i] * b[i]}
     * @throws IllegalArgumentException if the vectors differ in length
     */
    public static float[] multiply(float[] a, float[] b) {
        return Dispatcher.multiply(a, b);
    }

    /**
     * Computes the element-wise quotient of two vectors. Division by zero follows
     * IEEE-754 semantics, yielding {@code Infinity} or {@code NaN} rather than throwing.
     *
     * @param a the dividend vector
     * @param b the divisor vector, of the same length as {@code a}
     * @return a new vector where each element is {@code a[i] / b[i]}
     * @throws IllegalArgumentException if the vectors differ in length
     */
    public static float[] divide(float[] a, float[] b) {
        return Dispatcher.divide(a, b);
    }

    /**
     * Computes the sum of all elements of a vector.
     *
     * @param a the vector
     * @return the sum of the elements; {@code 0} for an empty vector
     */
    public static float sum(float[] a) {
        return Dispatcher.sum(a);
    }

    /**
     * Returns the smallest element of a vector.
     *
     * @param a the vector
     * @return the minimum element
     * @throws IllegalArgumentException if the vector is empty
     */
    public static float min(float[] a) {
        return Dispatcher.min(a);
    }

    /**
     * Returns the largest element of a vector.
     *
     * @param a the vector
     * @return the maximum element
     * @throws IllegalArgumentException if the vector is empty
     */
    public static float max(float[] a) {
        return Dispatcher.max(a);
    }

    /**
     * Multiplies every element of a vector by a scalar.
     *
     * @param a      the vector
     * @param scalar the multiplier
     * @return a new vector where each element is {@code a[i] * scalar}
     */
    public static float[] scale(float[] a, float scalar) {
        return Dispatcher.scale(a, scalar);
    }

    /**
     * Returns an independent copy of a vector.
     *
     * @param a the vector to copy
     * @return a new array with the same length and contents as {@code a}
     */
    public static float[] copy(float[] a) {
        return Dispatcher.copy(a);
    }

    /**
     * Sets every element of the given vector to a fixed value, in place.
     *
     * @param a     the vector to overwrite
     * @param value the value written to every element
     */
    public static void fill(float[] a, float value) {
        Dispatcher.fill(a, value);
    }

    /**
     * Returns a unit-length version of a vector, {@code a[i] / norm(a)}.
     *
     * @param a the vector to normalize
     * @return a new vector with L2 norm {@code 1}
     * @throws IllegalArgumentException if the vector has zero norm
     */
    public static float[] normalize(float[] a) {
        return Dispatcher.normalize(a);
    }

    /**
     * Computes the Euclidean distance between two vectors,
     * {@code sqrt(sum of (a[i] - b[i])^2)}.
     *
     * @param a the first vector
     * @param b the second vector, of the same length as {@code a}
     * @return the Euclidean distance
     * @throws IllegalArgumentException if the vectors differ in length
     */
    public static float euclideanDistance(float[] a, float[] b) {
        return Dispatcher.euclideanDistance(a, b);
    }

    /**
     * Computes the element-wise fused multiply-add {@code a[i] * b[i] + c[i]} using
     * {@link Math#fma}, performing a single rounding step per element.
     *
     * @param a the first factor vector
     * @param b the second factor vector, of the same length as {@code a}
     * @param c the addend vector, of the same length as {@code a}
     * @return a new vector where each element is {@code a[i] * b[i] + c[i]}
     * @throws IllegalArgumentException if the vectors differ in length
     */
    public static float[] fma(float[] a, float[] b, float[] c) {
        return Dispatcher.fma(a, b, c);
    }

    /**
     * Returns the index of the largest element of a vector. If several elements share
     * the maximum value, the index of the first one is returned.
     *
     * @param a the vector
     * @return the index of the maximum element
     * @throws IllegalArgumentException if the vector is empty
     */
    public static int argmax(float[] a) {
        return Dispatcher.argmax(a);
    }

    /**
     * Computes the softmax of a vector. The implementation is numerically stable: the
     * maximum element is subtracted before exponentiation. The returned values are
     * non-negative and sum to {@code 1}.
     *
     * @param a the input vector
     * @return a new vector containing the softmax distribution
     * @throws IllegalArgumentException if the vector is empty
     */
    public static float[] softmax(float[] a) {
        return Dispatcher.softmax(a);
    }
}
