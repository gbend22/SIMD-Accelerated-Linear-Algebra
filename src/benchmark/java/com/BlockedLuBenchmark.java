package com;

import com.performance.BlockedLuSweep;
import com.simd.SimdDecompositionOps;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
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
public class BlockedLuBenchmark {

    @Param({"128", "256", "512", "1024"})
    public int size;

    @Param({"8", "16", "32", "64", "128"})
    public int blockSize;

    private final BlockedLuSweep blocked = new BlockedLuSweep();
    private final SimdDecompositionOps simd = new SimdDecompositionOps();

    private float[][] a;

    @Setup(Level.Trial)
    public void setup() {

        Random rng = new Random(42);

        a = new float[size][size];

        for (int r = 0; r < size; r++) {
            for (int c = 0; c < size; c++) {
                a[r][c] = rng.nextFloat() * 2f - 1f;
            }
            a[r][r] += size;
        }
    }

    @Benchmark
    public void blocked_lu(Blackhole bh) {
        bh.consume(blocked.lu(a, blockSize));
    }

    @Benchmark
    public void unblocked_simd_lu(Blackhole bh) {
        bh.consume(simd.lu(a));
    }

    public static void main(String[] args)
            throws RunnerException {

        Options opt = new OptionsBuilder()
                .include(BlockedLuBenchmark.class.getSimpleName())
                .build();

        new Runner(opt).run();
    }
}
