package com;

import com.simd.SimdVectorOps;

import java.util.concurrent.ForkJoinPool;

/** Environment reporting and an optional fail-fast vector-width assertion for JMH. */
public final class BenchmarkEnvironment {

    private static final String EXPECTED_LANES_PROPERTY = "simdlinalg.expectedLanes";

    private BenchmarkEnvironment() {}

    /**
     * Fails before measurement if {@code -Dsimdlinalg.expectedLanes=N} does not match
     * {@link jdk.incubator.vector.FloatVector#SPECIES_PREFERRED} on the forked JVM.
     */
    public static void verifyExpectedVectorWidth() {
        String configured = System.getProperty(EXPECTED_LANES_PROPERTY);
        if (configured == null || configured.isBlank()) {
            return;
        }

        final int expected;
        try {
            expected = Integer.parseInt(configured);
        } catch (NumberFormatException exception) {
            throw new IllegalArgumentException(
                    EXPECTED_LANES_PROPERTY + " must be an integer, got " + configured,
                    exception);
        }

        int actual = SimdVectorOps.simdWidth();
        if (actual != expected) {
            throw new IllegalStateException(
                    "Expected " + expected + " SIMD float lanes but the forked JVM selected "
                            + actual + ". Check CPU support and -XX:UseAVX.");
        }
    }

    /** Prints the metadata that should accompany every reported benchmark run. */
    public static void main(String[] args) {
        verifyExpectedVectorWidth();
        System.out.println("java.version=" + System.getProperty("java.version"));
        System.out.println("java.vm.name=" + System.getProperty("java.vm.name"));
        System.out.println("java.vm.version=" + System.getProperty("java.vm.version"));
        System.out.println("os.name=" + System.getProperty("os.name"));
        System.out.println("os.version=" + System.getProperty("os.version"));
        System.out.println("os.arch=" + System.getProperty("os.arch"));
        System.out.println("processor.identifier="
                + System.getenv().getOrDefault("PROCESSOR_IDENTIFIER", "unavailable"));
        System.out.println("available.processors=" + Runtime.getRuntime().availableProcessors());
        System.out.println("common.pool.parallelism=" + ForkJoinPool.commonPool().getParallelism());
        System.out.println("simd.float.lanes=" + SimdVectorOps.simdWidth());
    }
}
