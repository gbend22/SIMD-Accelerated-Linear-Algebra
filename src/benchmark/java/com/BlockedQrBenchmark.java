package com;

import com.performance.BlockedQrSweep;
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
public class BlockedQrBenchmark {

    @Param({"128", "256", "512", "1024"})
    public int size;

    @Param({"8", "16", "32", "64", "128"})
    public int blockSize;

    private final BlockedQrSweep blocked = new BlockedQrSweep();
    private final SimdDecompositionOps unblocked = SimdDecompositionOps.unblocked();

    private float[][] a;

    @Setup(Level.Trial)
    public void setup() {
        BenchmarkEnvironment.verifyExpectedVectorWidth();

        Random rng = new Random(42);

        a = new float[size][size];

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                a[i][j] = rng.nextFloat() * 2f - 1f;
            }
        }
    }

    @Benchmark
    public void blocked_qr(Blackhole bh) {
        bh.consume(blocked.qr(a, blockSize));
    }

    @Benchmark
    public void unblocked_simd_qr(Blackhole bh) {
        bh.consume(unblocked.qr(a));
    }

    public static void main(String[] args)
            throws RunnerException {

        Options opt = new OptionsBuilder()
                .include(BlockedQrBenchmark.class.getSimpleName())
                .build();

        new Runner(opt).run();
    }
}
