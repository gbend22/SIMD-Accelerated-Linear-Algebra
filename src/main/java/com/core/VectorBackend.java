package com.core;

public interface VectorBackend {
    float dot(float[] a, float[] b);
    float norm(float[] a);
    float cosineSimilarity(float[] a, float[] b);
    float[] add(float[] a, float[] b);
    float[] subtract(float[] a, float[] b);
    float[] multiply(float[] a, float[] b);
    float[] divide(float[] a, float[] b);
    float sum(float[] a);
    float min(float[] a);
    float max(float[] a);
    float[] scale(float[] a, float scalar);
    float[] copy(float[] a);
    void fill(float[] a, float value);
    float[] normalize(float[] a);
    float euclideanDistance(float[] a, float[] b);
    float[] fma(float[] a, float[] b, float[] c);
    int argmax(float[] a);
    float[] softmax(float[] a);
}
