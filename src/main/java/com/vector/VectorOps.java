package com.vector;

import com.core.Dispatcher;

public class VectorOps {

    private VectorOps() {}

    public static float dot(float[] a, float[] b) {
        return Dispatcher.dot(a, b);
    }

    public static float norm(float[] a) {
        return Dispatcher.norm(a);
    }

    public static float cosineSimilarity(float[] a, float[] b) {
        return Dispatcher.cosineSimilarity(a, b);
    }

    public static float[] add(float[] a, float[] b) {
        return Dispatcher.add(a, b);
    }

    public static float[] subtract(float[] a, float[] b) {
        return Dispatcher.subtract(a, b);
    }

    public static float[] multiply(float[] a, float[] b) {
        return Dispatcher.multiply(a, b);
    }

    public static float[] divide(float[] a, float[] b) {
        return Dispatcher.divide(a, b);
    }

    public static float sum(float[] a) {
        return Dispatcher.sum(a);
    }

    public static float min(float[] a) {
        return Dispatcher.min(a);
    }

    public static float max(float[] a) {
        return Dispatcher.max(a);
    }

    public static float[] scale(float[] a, float scalar) {
        return Dispatcher.scale(a, scalar);
    }

    public static float[] copy(float[] a) {
        return Dispatcher.copy(a);
    }

    public static void fill(float[] a, float value) {
        Dispatcher.fill(a, value);
    }

    public static float[] normalize(float[] a) {
        return Dispatcher.normalize(a);
    }

    public static float euclideanDistance(float[] a, float[] b) {
        return Dispatcher.euclideanDistance(a, b);
    }

    public static float[] fma(float[] a, float[] b, float[] c) {
        return Dispatcher.fma(a, b, c);
    }

    public static int argmax(float[] a) {
        return Dispatcher.argmax(a);
    }

    public static float[] softmax(float[] a) {
        return Dispatcher.softmax(a);
    }
}
