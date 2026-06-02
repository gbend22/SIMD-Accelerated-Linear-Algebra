package com;

import com.scalar.ScalarVectorOps;
import com.simd.SimdVectorOps;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.Random;
import java.util.concurrent.TimeUnit;


@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Fork(
        value = 2,
        jvmArgs = {
                "--add-modules", "jdk.incubator.vector",
                "-XX:+UseVectorCmov"
        }
)
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
public class VectorOpsBenchmark {

    @Param({"16", "64", "256", "1024", "4096", "16384"})
    public int size;

    private final ScalarVectorOps scalar = new ScalarVectorOps();
    private final SimdVectorOps simd = new SimdVectorOps();

    private float[] a;
    private float[] b;

    @Setup(Level.Trial)
    public void setup() {
        Random rng = new Random(42);
        a = new float[size];
        b = new float[size];
        for (int i = 0; i < size; i++) {
            a[i] = rng.nextFloat() * 2f - 1f;
            b[i] = rng.nextFloat() * 2f - 1f;
        }
    }

    @Benchmark
    public float scalar_dot() {
        return scalar.dot(a, b);
    }

    @Benchmark
    public float simd_dot() {
        return simd.dot(a, b);
    }

    @Benchmark
    public float scalar_norm() {
        return scalar.norm(a);
    }

    @Benchmark
    public float simd_norm() {
        return simd.norm(a);
    }

    @Benchmark
    public float scalar_cosine() {
        return scalar.cosineSimilarity(a, b);
    }

    @Benchmark
    public float simd_cosine() {
        return simd.cosineSimilarity(a, b);
    }

    @Benchmark
    public float[] scalar_add() {
        return scalar.add(a, b);
    }

    @Benchmark
    public float[] simd_add() {
        return simd.add(a, b);
    }

    @Benchmark
    public float scalar_sum() {
        return scalar.sum(a);
    }

    @Benchmark
    public float simd_sum() {
        return simd.sum(a);
    }

    @Benchmark
    public float scalar_min() {
        return scalar.min(a);
    }

    @Benchmark
    public float simd_min() {
        return simd.min(a);
    }

    @Benchmark
    public float scalar_max() {
        return scalar.max(a);
    }

    @Benchmark
    public float simd_max() {
        return simd.max(a);
    }

    @Benchmark
    public float[] scalar_scale() {
        return scalar.scale(a, 2.5f);
    }

    @Benchmark
    public float[] simd_scale() {
        return simd.scale(a, 2.5f);
    }

    @Benchmark
    public float[] scalar_normalize() {
        return scalar.normalize(a);
    }

    @Benchmark
    public float[] simd_normalize() {
        return simd.normalize(a);
    }

    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
                .include(VectorOpsBenchmark.class.getSimpleName())
                .forks(1)
                .warmupIterations(3)
                .measurementIterations(3)
                .build();
        new Runner(opt).run();
    }
}
