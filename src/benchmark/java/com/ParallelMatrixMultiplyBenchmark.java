package com;

import com.baseline.NaiveSimdMatrixOps;
import com.performance.CacheBlockedTiledSimdMatrixOps;
import com.performance.ParallelCacheBlockedTiledSimdMatrixOps;
import com.performance.ParallelSimdMatrixOps;
import com.performance.ParallelTiledSimdMatrixOps;
import com.performance.TiledSimdMatrixOps;
import com.scalar.ScalarMatrixOps;
import com.simd.SimdMatrixOps;

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
public class ParallelMatrixMultiplyBenchmark {

    @Param({"128", "256", "512", "1024"})
    public int size;

    private final ScalarMatrixOps scalar = new ScalarMatrixOps();
    private final NaiveSimdMatrixOps naive = new NaiveSimdMatrixOps();
    private final SimdMatrixOps productionRegisterTiled = new SimdMatrixOps();
    private final TiledSimdMatrixOps tiled = new TiledSimdMatrixOps();
    private final CacheBlockedTiledSimdMatrixOps cacheBlocked = new CacheBlockedTiledSimdMatrixOps();
    private final ParallelSimdMatrixOps parallel = new ParallelSimdMatrixOps();
    private final ParallelTiledSimdMatrixOps parallelTiled = new ParallelTiledSimdMatrixOps();
    private final ParallelCacheBlockedTiledSimdMatrixOps parallelCacheBlocked = new ParallelCacheBlockedTiledSimdMatrixOps();

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
    public void scalar_multiply(Blackhole bh) {
        bh.consume(scalar.multiply(a, b));
    }

    @Benchmark
    public void naiveSimd_multiply(Blackhole bh) {
        bh.consume(naive.multiply(a, b));
    }

    @Benchmark
    public void productionRegisterTiled_multiply(Blackhole bh) {
        bh.consume(productionRegisterTiled.multiply(a, b));
    }

    @Benchmark
    public void registerTiledMr4_multiply(Blackhole bh) {
        bh.consume(tiled.multiply(a, b));
    }

    @Benchmark
    public void cacheBlockedTiled_multiply(Blackhole bh) {
        bh.consume(cacheBlocked.multiply(a, b));
    }

    @Benchmark
    public void parallelSimd_multiply(Blackhole bh) {
        bh.consume(parallel.multiply(a, b));
    }

    @Benchmark
    public void parallelTiledSimd_multiply(Blackhole bh) {
        bh.consume(parallelTiled.multiply(a, b));
    }

    @Benchmark
    public void parallelCacheBlockedTiled_multiply(Blackhole bh) {
        bh.consume(parallelCacheBlocked.multiply(a, b));
    }

    public static void main(String[] args)
            throws RunnerException {

        Options opt = new OptionsBuilder()
                .include(ParallelMatrixMultiplyBenchmark.class.getSimpleName())
                .build();

        new Runner(opt).run();
    }
}
