package com;

import com.performance.RegisterTileSweepMatrixOps;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.Random;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.Throughput)
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
public class RegisterTileSweepBenchmark {

    @Param({"512", "1024"})
    public int size;

    @Param({"1", "2", "4", "6", "8"})
    public int mr;

    private final RegisterTileSweepMatrixOps sweep = new RegisterTileSweepMatrixOps();

    private float[][] a;
    private float[][] b;

    @Setup(Level.Trial)
    public void setup() {
        BenchmarkEnvironment.verifyExpectedVectorWidth();

        Random rng = new Random(42);

        a = new float[size][size];
        b = new float[size][size];

        for (int r = 0; r < size; r++) {
            for (int c = 0; c < size; c++) {
                a[r][c] = rng.nextFloat() * 2f - 1f;
                b[r][c] = rng.nextFloat() * 2f - 1f;
            }
        }
    }

    @Benchmark
    public void registerTile_multiply(Blackhole bh) {
        bh.consume(sweep.multiply(a, b, mr));
    }

    public static void main(String[] args)
            throws RunnerException {

        Options opt = new OptionsBuilder()
                .include(RegisterTileSweepBenchmark.class.getSimpleName())
                .build();

        new Runner(opt).run();
    }
}
