package com;

import com.scalar.ScalarDecompositionOps;
import com.simd.SimdDecompositionOps;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.Random;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
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
public class DecompositionBenchmark {

    @Param({"64", "128", "256", "512"})
    public int size;

    private final ScalarDecompositionOps scalar = new ScalarDecompositionOps();
    private final SimdDecompositionOps simd = new SimdDecompositionOps();

    private float[][] a;
    private float[] b;

    @Setup(Level.Trial)
    public void setup() {
        BenchmarkEnvironment.verifyExpectedVectorWidth();

        Random rng = new Random(42);

        a = new float[size][size];
        b = new float[size];

        for (int r = 0; r < size; r++) {

            b[r] = rng.nextFloat() * 2f - 1f;

            for (int c = 0; c < size; c++) {
                a[r][c] = rng.nextFloat() * 2f - 1f;
            }

            a[r][r] += size;
        }
    }

    @Benchmark
    public void scalar_lu(Blackhole bh) {
        bh.consume(scalar.lu(a));
    }

    @Benchmark
    public void simd_lu(Blackhole bh) {
        bh.consume(simd.lu(a));
    }

    @Benchmark
    public void scalar_solve(Blackhole bh) {
        bh.consume(scalar.solve(a, b));
    }

    @Benchmark
    public void simd_solve(Blackhole bh) {
        bh.consume(simd.solve(a, b));
    }

    @Benchmark
    public void scalar_inverse(Blackhole bh) {
        bh.consume(scalar.inverse(a));
    }

    @Benchmark
    public void simd_inverse(Blackhole bh) {
        bh.consume(simd.inverse(a));
    }

    public static void main(String[] args)
            throws RunnerException {

        Options opt = new OptionsBuilder()
                .include(DecompositionBenchmark.class.getSimpleName())
                .build();

        new Runner(opt).run();
    }
}
