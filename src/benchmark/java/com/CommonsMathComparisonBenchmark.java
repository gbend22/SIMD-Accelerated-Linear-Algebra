package com;

import com.simd.SimdVectorOps;
import org.apache.commons.math3.linear.ArrayRealVector;
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
                "--add-modules", "jdk.incubator.vector"
        }
)
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
public class CommonsMathComparisonBenchmark {

    @Param({"16", "64", "256", "1024", "4096", "16384"})
    public int size;

    private final SimdVectorOps simd = new SimdVectorOps();

    private float[] a;
    private float[] b;

    private ArrayRealVector va;
    private ArrayRealVector vb;

    @Setup(Level.Trial)
    public void setup() {
        Random rng = new Random(42);

        a = new float[size];
        b = new float[size];

        double[] da = new double[size];
        double[] db = new double[size];

        for (int i = 0; i < size; i++) {
            float x = rng.nextFloat() * 2f - 1f;
            float y = rng.nextFloat() * 2f - 1f;

            a[i] = x;
            b[i] = y;

            da[i] = x;
            db[i] = y;
        }

        va = new ArrayRealVector(da);
        vb = new ArrayRealVector(db);
    }

    @Benchmark
    public float simd_dot() {
        return simd.dot(a, b);
    }

    @Benchmark
    public double commons_dot() {
        return va.dotProduct(vb);
    }

    @Benchmark
    public float simd_norm() {
        return simd.norm(a);
    }

    @Benchmark
    public double commons_norm() {
        return va.getNorm();
    }

    @Benchmark
    public float[] simd_add() {
        return simd.add(a, b);
    }

    @Benchmark
    public double[] commons_add() {
        return va.add(vb).toArray();
    }

    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
                .include(CommonsMathComparisonBenchmark.class.getSimpleName())
                .forks(1)
                .build();

        new Runner(opt).run();
    }
}
