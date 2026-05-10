package com;

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

@BenchmarkMode(Mode.AverageTime)
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
public class MatrixOpsBenchmark {

    @Param({"64", "128", "256", "512"})
    public int size;

    private float[][] a;
    private float[][] b;
    private float[] vector;

    @Setup(Level.Trial)
    public void setup() {

        Random rng = new Random(42);

        a = new float[size][size];
        b = new float[size][size];
        vector = new float[size];

        for (int r = 0; r < size; r++) {

            vector[r] = rng.nextFloat() * 2f - 1f;

            for (int c = 0; c < size; c++) {

                a[r][c] = rng.nextFloat() * 2f - 1f;
                b[r][c] = rng.nextFloat() * 2f - 1f;
            }
        }
    }

    @Benchmark
    public void scalar_matrixVectorMultiply(Blackhole bh) {

        bh.consume(
                ScalarMatrixOps.multiplyMatrixVector(
                        a,
                        vector
                )
        );
    }

    @Benchmark
    public void simd_matrixVectorMultiply(Blackhole bh) {

        bh.consume(
                SimdMatrixOps.multiplyMatrixVector(
                        a,
                        vector
                )
        );
    }

    @Benchmark
    public void scalar_add(Blackhole bh) {

        bh.consume(
                ScalarMatrixOps.add(a, b)
        );
    }

    @Benchmark
    public void simd_add(Blackhole bh) {

        bh.consume(
                SimdMatrixOps.add(a, b)
        );
    }

    public static void main(String[] args)
            throws RunnerException {

        Options opt = new OptionsBuilder()
                .include(MatrixOpsBenchmark.class.getSimpleName())
                .build();

        new Runner(opt).run();
    }
}