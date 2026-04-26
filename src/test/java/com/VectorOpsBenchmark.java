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

    @Param({"64", "256", "1024", "4096"})
    public int size;

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
        return ScalarVectorOps.dot(a, b);
    }

    @Benchmark
    public float simd_dot() {
        return SimdVectorOps.dot(a, b);
    }

    @Benchmark
    public float scalar_norm() {
        return ScalarVectorOps.norm(a);
    }

    @Benchmark
    public float simd_norm() {
        return SimdVectorOps.norm(a);
    }

    @Benchmark
    public float scalar_cosine() {
        return ScalarVectorOps.cosineSimilarity(a, b);
    }

    @Benchmark
    public float simd_cosine() {
        return SimdVectorOps.cosineSimilarity(a, b);
    }

    @Benchmark
    public float[] scalar_add() {
        return ScalarVectorOps.add(a, b);
    }

    @Benchmark
    public float[] simd_add() {
        return SimdVectorOps.add(a, b);
    }

    @Benchmark
    public float scalar_sum() {
        return ScalarVectorOps.sum(a);
    }

    @Benchmark
    public float simd_sum() {
        return SimdVectorOps.sum(a);
    }

    @Benchmark
    public float scalar_min() {
        return ScalarVectorOps.min(a);
    }

    @Benchmark
    public float simd_min() {
        return SimdVectorOps.min(a);
    }

    @Benchmark
    public float scalar_max() {
        return ScalarVectorOps.max(a);
    }

    @Benchmark
    public float simd_max() {
        return SimdVectorOps.max(a);
    }

    @Benchmark
    public float[] scalar_scale() {
        return ScalarVectorOps.scale(a, 2.5f);
    }

    @Benchmark
    public float[] simd_scale() {
        return SimdVectorOps.scale(a, 2.5f);
    }

    @Benchmark
    public float[] scalar_normalize() {
        return ScalarVectorOps.normalize(a);
    }

    @Benchmark
    public float[] simd_normalize() {
        return SimdVectorOps.normalize(a);
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
