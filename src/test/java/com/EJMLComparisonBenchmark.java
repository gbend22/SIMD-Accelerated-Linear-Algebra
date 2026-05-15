package com;

import com.simd.SimdVectorOps;

import org.ejml.simple.SimpleMatrix;

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
public class EJMLComparisonBenchmark {

    @Param({"64", "256", "1024", "4096"})
    public int size;

    private float[] a;
    private float[] b;

    private SimpleMatrix ejmlA;
    private SimpleMatrix ejmlB;

    @Setup(Level.Trial)
    public void setup() {
        Random rng = new Random(42);

        a = new float[size];
        b = new float[size];

        double[][] da = new double[size][1];
        double[][] db = new double[size][1];

        for (int i = 0; i < size; i++) {
            a[i] = rng.nextFloat() * 2f - 1f;
            b[i] = rng.nextFloat() * 2f - 1f;

            da[i][0] = a[i];
            db[i][0] = b[i];
        }

        ejmlA = new SimpleMatrix(da);
        ejmlB = new SimpleMatrix(db);
    }

    @Benchmark
    public float simd_dot() {
        return SimdVectorOps.dot(a, b);
    }

    @Benchmark
    public double ejml_dot() {
        return ejmlA.dot(ejmlB);
    }

    @Benchmark
    public float simd_norm() {
        return SimdVectorOps.norm(a);
    }

    @Benchmark
    public double ejml_norm() {
        return ejmlA.normF();
    }

    @Benchmark
    public void simd_add(Blackhole bh) {
        bh.consume(SimdVectorOps.add(a, b));
    }

    @Benchmark
    public void ejml_add(Blackhole bh) {
        bh.consume(ejmlA.plus(ejmlB));
    }

    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
                .include(EJMLComparisonBenchmark.class.getSimpleName())
                .build();

        new Runner(opt).run();
    }
}
