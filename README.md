# JavaSIMDLinalg

> A SIMD-accelerated linear algebra library for Java and a reproducible study of explicit
> vectorization on the JVM.

JavaSIMDLinalg is both a usable software product and the implementation artifact of a
bachelor's research project. It combines a product-oriented linear algebra library with a
substantial experimental study of SIMD performance, memory-aware optimization, and
algorithm design on the Java platform.

From the product perspective, the project provides a cohesive public API for dense
single-precision linear algebra, automatic backend selection, documented validation and
error behavior, Maven packaging, API documentation, and end-to-end application examples.
From the research perspective, it provides independently testable scalar baselines,
multiple optimized kernels, controlled AVX2 and AVX-512 experiments, raw JMH data, and an
analysis that includes both successful and unsuccessful optimizations.

The two aims reinforce each other. A fast kernel without a dependable interface is not a
complete software product, while an optimized library without controlled measurements
cannot support a rigorous performance argument. JavaSIMDLinalg is designed to demonstrate
both engineering completeness and evidence-based evaluation.

The current `1.0-SNAPSHOT` version denotes pre-release maturity, not a throwaway prototype.
It can be built, tested, installed into a local Maven repository, used through its public
facades, and evaluated through the included benchmark artifact. Publication to Maven
Central remains a release and distribution step.

## Project objectives

### Product engineering objectives

- provide a small and consistent API through `VectorOps` and `MatrixOps`;
- hide scalar and SIMD implementation details behind transparent runtime dispatch;
- validate inputs and report invalid shapes, singular systems, and invalid decompositions
  predictably;
- keep the public facades stateless and return immutable decomposition result objects;
- package the library, source code, Javadocs, and benchmark harness as separate artifacts;
- demonstrate reuse through k-means, k-nearest neighbors, and linear regression;
- maintain high automated test coverage across scalar, SIMD, blocked, and parallel code.

### Research objectives

The experimental component addresses three questions:

1. How much speedup can explicit Java Vector API kernels provide over straightforward
   scalar loops for dense `float` operations?
2. How do wider SIMD vectors, register tiling, cache blocking, and internal parallelism
   change performance as the input size grows?
3. Can optimized backends remain numerically consistent with a scalar reference while
   presenting one product-facing API?

### Technical contributions

- scalar and SIMD backends behind a shared public facade;
- width-portable kernels based on `FloatVector.SPECIES_PREFERRED`;
- scalar tail handling for input sizes that are not multiples of the SIMD width;
- register-tiled, cache-blocked, and parallel matrix multiplication variants;
- LU with partial pivoting, Householder QR, and Cholesky decomposition;
- blocked LU and QR implementations with forced-unblocked experimental controls;
- centralized validation for matrix shape, symmetry, and finite values;
- 577 automated tests and a reproducible JMH evaluation on AVX2 and AVX-512.

## Capabilities

| Category | Operations |
|---|---|
| Vector reductions | dot product, sum, minimum, maximum, L2 norm, argmax |
| Vector transforms | add, subtract, multiply, divide, scale, normalize, softmax, FMA |
| Vector utilities | cosine similarity, Euclidean distance, copy, fill |
| Matrix operations | matrix-vector multiply, matrix-matrix multiply, add, transpose |
| Decompositions | LU with partial pivoting, Householder QR, Cholesky |
| Linear algebra | solve `Ax = b`, determinant, inverse |
| Example applications | k-means, k-nearest neighbors, linear regression |

All public numerical operations use `float`. Matrices are represented by rectangular
row-major `float[][]` arrays. Methods return newly allocated results except
`VectorOps.fill`, which operates in place.

## Requirements

- Java 21 or newer
- Maven
- `jdk.incubator.vector` enabled at compile time and runtime

The source and bytecode target Java 21. The archived performance evaluation was run with
JDK 24.0.1. AVX2 or AVX-512 is not required to build or test the project, but matching
hardware is required to reproduce the corresponding benchmark profile.

## Build and installation

Run the complete build from the repository root:

```text
mvn clean package
```

The build runs the tests and produces four independently useful artifacts:

```text
target/simd-linalg-1.0-SNAPSHOT.jar
target/simd-linalg-1.0-SNAPSHOT-sources.jar
target/simd-linalg-1.0-SNAPSHOT-javadoc.jar
target/simd-linalg-1.0-SNAPSHOT-benchmarks.jar
```

The main JAR is the reusable library. The source and Javadoc JARs support inspection and
API integration, while the executable benchmark JAR keeps performance evaluation separate
from ordinary library use.

To install the library in the local Maven repository:

```text
mvn install
```

The local dependency coordinates are:

```xml
<dependency>
    <groupId>com.simdlinalg</groupId>
    <artifactId>simd-linalg</artifactId>
    <version>1.0-SNAPSHOT</version>
</dependency>
```

Applications using the library must enable the incubator module:

```text
java --add-modules jdk.incubator.vector -cp <classpath> <main-class>
```

## Usage

The public entry points are `com.vector.VectorOps` and `com.matrix.MatrixOps`.

### Vector operations

```java
import com.vector.VectorOps;

float[] a = {1f, 2f, 3f, 4f};
float[] b = {5f, 6f, 7f, 8f};

float dot = VectorOps.dot(a, b);
float norm = VectorOps.norm(a);
float cosine = VectorOps.cosineSimilarity(a, b);
float[] sum = VectorOps.add(a, b);
float[] unit = VectorOps.normalize(a);
```

For this example, `dot` is `70`, `norm` is approximately `5.477`, and `sum` contains
`{6, 8, 10, 12}`.

### Matrix operations and solvers

```java
import com.decomp.LUDecomposition;
import com.decomp.QRDecomposition;
import com.matrix.MatrixOps;

float[][] matrix = {
    {4f, 3f},
    {6f, 3f}
};
float[] vector = {7f, 5f};

float[] product = MatrixOps.multiply(matrix, vector);
float[][] transpose = MatrixOps.transpose(matrix);
LUDecomposition lu = MatrixOps.lu(matrix);
float determinant = MatrixOps.determinant(matrix);
float[] solution = MatrixOps.solve(matrix, vector);
float[][] inverse = MatrixOps.inverse(matrix);

float[][] tall = {
    {1f, 2f},
    {3f, 4f},
    {5f, 6f}
};
QRDecomposition qr = MatrixOps.qr(tall);
```

Vector operands must have compatible lengths. Matrices must be non-null, non-empty, and
rectangular. Decompositions validate their shape requirements. Cholesky additionally
requires finite, symmetric, positive-definite input. Invalid structure produces
`IllegalArgumentException`; singular or non-positive-definite problems produce
`ArithmeticException` where appropriate.

## Architecture

```text
Application code
      |
      v
VectorOps / MatrixOps                 public, stateless facades
      |
      v
Dispatcher                            one backend selection at class initialization
      |
      +-- ScalarVectorOps             reference vector implementation
      +-- ScalarMatrixOps             reference matrix implementation
      +-- ScalarDecompositionOps      reference decomposition implementation
      +-- SimdVectorOps               Vector API kernels
      +-- SimdMatrixOps               SIMD and register-tiled matrix kernels
      +-- SimdDecompositionOps        SIMD, blocked LU, and blocked QR kernels
```

`Dispatcher` reads the preferred `float` vector width once. It selects the SIMD backends
when at least four lanes are reported; otherwise it selects the scalar backends. The
scalar implementations are also used independently as correctness oracles and benchmark
baselines.

### SIMD strategy

The SIMD code is written once against `FloatVector.SPECIES_PREFERRED`. The same kernels
therefore use 8 `float` lanes when HotSpot is capped at AVX2 and 16 lanes when AVX-512 is
selected. Each loop processes full vectors first and completes the remaining elements
with a scalar tail loop.

Reduction kernels accumulate partial values in vector lanes before reducing them to a
scalar. Element-wise kernels load, transform, and store complete vectors. Matrix-vector
multiplication vectorizes each row dot product.

### Matrix multiplication

The production `SimdMatrixOps` multiplication kernel uses an eight-row register tile. The
benchmark harness additionally isolates:

- a genuinely naive SIMD baseline;
- four-row and eight-row register tiles;
- cache-blocked register tiling;
- naive internal parallelism;
- parallel register tiling;
- parallel cache-blocked register tiling.

These variants separate the contribution of SIMD width from data reuse and parallel
execution. Experimental parallel kernels are not silently substituted into the public
API.

### Decompositions

The SIMD backend implements LU, QR, and Cholesky directly. Production LU uses a block size
of 64 from size 256, while production QR uses a block size of 32 from size 128. The
benchmark suite can force the unblocked SIMD implementations, making the effect of
blocking measurable. Cholesky remains unblocked in the production backend because the
measured blocked variants were slower on the evaluation machine.

## Applications and product demonstrations

Three complete models exercise the public API end to end:

- `KMeans` assigns samples using Euclidean distance and recomputes centroids with vector
  addition and scaling.
- `KNearestNeighbors` ranks training samples by Euclidean distance and performs majority
  voting among the nearest samples.
- `LinearRegression` solves the normal equations using matrix transpose, multiplication,
  and LU-based solving.

Example:

```java
import com.applications.LinearRegression;

static float trainAndPredict(float[][] features, float[] targets, float[] sample) {
    LinearRegression model = new LinearRegression();
    model.fit(features, targets);
    return model.predict(sample);
}
```

These classes serve two roles: they are ready-to-run usage examples for library users and
integration tests showing that the facade, dispatcher, numerical backends, and solvers
compose into higher-level workloads. Their scope is intentionally compact, but they
establish that JavaSIMDLinalg is usable as a component rather than only as a collection of
isolated microbenchmark kernels.

## Correctness and testing

The current test suite contains 577 JUnit 5 tests. A fresh run completed with 577 passed,
0 failed, 0 errors, and 0 skipped. The current JaCoCo report gives 96.5% line coverage and
96.0% instruction coverage for the library classes. Benchmark and benchmark-only baseline
classes are excluded from the coverage calculation.

```text
mvn test
```

The HTML coverage report is generated at:

```text
target/site/jacoco/index.html
```

The tests cover:

- public API contracts and invalid inputs;
- scalar and SIMD backend parity;
- vector tails at non-multiple lengths;
- `NaN`, infinity, zero vectors, and large magnitudes;
- rectangular, ragged, square, symmetric, and singular matrices;
- LU, QR, and Cholesky reconstruction and structural properties;
- blocked and unblocked decomposition parity;
- scale-relative solve and inverse residuals at multiple matrix scales;
- optimized and parallel matrix multiplication variants;
- the three example applications.

Floating-point reductions are checked with tolerances rather than bit-for-bit equality.
SIMD changes the order of addition, and floating-point addition is not associative.
Decomposition tests use reconstruction or residual criteria appropriate for
single-precision arithmetic.

## Performance evaluation

The repository contains 780 complete JMH result records: 390 for AVX2 and 390 for
AVX-512. The results below are a single-machine experimental snapshot. They demonstrate
behavior on the recorded system and should not be interpreted as universal JVM or CPU
performance claims.

### Experimental environment

| Item | Recorded value |
|---|---|
| CPU | Intel Core i5-1035G1 |
| Logical processors visible to JVM | 8 |
| Common pool parallelism | 7 |
| Operating system | Windows 11, amd64 |
| JVM | Java HotSpot 64-Bit Server VM 24.0.1+9-30 |
| JMH | 1.37 |
| AVX2 profile | `-XX:UseAVX=2`, 8 `float` lanes |
| AVX-512 profile | `-XX:UseAVX=3`, 16 `float` lanes |
| Forks | 2 |
| Warmup | 5 iterations, 1 second each |
| Measurement | 5 iterations, 1 second each |
| JMH threads | 1 |
| Input generation | deterministic pseudo-random values, seed 42 |

The matrix multiplication benchmark has one JMH thread, but its explicitly parallel
variants use the Java common fork-join pool. Decomposition results use average time in
milliseconds per operation. Core vector, matrix, comparison, and matrix multiplication
results use throughput. The exact unit is stored in every JSON record.

Speedups below are ratios of JMH point estimates. Values above `1.00x` favor the optimized
implementation. The raw files contain `scoreError`, `scoreConfidence`, percentiles, and
all fork-level measurements. Comparisons with materially overlapping confidence intervals
are treated as inconclusive.

### Raw results

| Suite | AVX2, 8 lanes | AVX-512, 16 lanes |
|---|---|---|
| Core operations | [JSON](benchmark-results/jdk24/avx2/results-jdk24-avx2-core.json) | [JSON](benchmark-results/jdk24/avx512/results-jdk24-avx512-core.json) |
| EJML and Commons Math | [JSON](benchmark-results/jdk24/avx2/results-jdk24-avx2-comparisons.json) | [JSON](benchmark-results/jdk24/avx512/results-jdk24-avx512-comparisons.json) |
| Matrix multiplication | [JSON](benchmark-results/jdk24/avx2/results-jdk24-avx2-matmul.json) | [JSON](benchmark-results/jdk24/avx512/results-jdk24-avx512-matmul.json) |
| Decompositions | [JSON](benchmark-results/jdk24/avx2/results-jdk24-avx2-decompositions.json) | [JSON](benchmark-results/jdk24/avx512/results-jdk24-avx512-decompositions.json) |

### Core SIMD results

The following table uses representative large inputs. `n` is the vector length or the
square matrix dimension. The last column compares the absolute SIMD throughput between
AVX-512 and AVX2, rather than comparing their separate speedup ratios.

| Operation | n | AVX2 vs scalar | AVX-512 vs scalar | AVX-512 vs AVX2 SIMD throughput |
|---|---:|---:|---:|---:|
| Dot product | 4096 | 7.98x | 16.29x | 1.95x |
| L2 norm | 4096 | 8.56x | 18.39x | 2.05x |
| Cosine similarity | 4096 | 8.20x | 17.07x | 1.94x |
| Sum | 4096 | 8.39x | 18.56x | 2.07x |
| Normalize | 16384 | 3.04x | 3.93x | 1.24x |
| Vector add | 16384 | 0.81x | 0.88x | 1.10x |
| Matrix-vector multiply | 512 | 6.55x | 7.92x | 1.11x |
| Production matrix multiply | 512 | 17.07x | 32.22x | 1.78x |
| Matrix add | 512 | 0.99x | 1.01x | 0.99x |

Compute-heavy reductions scale close to the twofold lane increase. Allocating,
memory-bound operations do not: vector addition and scaling are slower than the scalar
reference at the largest tested vector size, and matrix addition is effectively tied.
This distinction is central to the result. SIMD width alone does not guarantee a speedup.

### Matrix multiplication results

This table compares the experimental parallel cache-blocked register-tiled kernel with the
scalar matrix multiplication baseline.

| Square matrix size | AVX2 speedup | AVX-512 speedup | AVX-512 vs AVX2 optimized throughput |
|---:|---:|---:|---:|
| 128 | 20.48x | 32.90x | 1.50x |
| 256 | 47.17x | 64.92x | 1.35x |
| 512 | 60.46x | 86.12x | 1.41x |
| 1024 | 57.95x | 98.97x | 1.45x |

These values combine SIMD, improved cache locality, register reuse, and seven-way common
pool parallelism. They must not be presented as SIMD-only gains. In the single-threaded
production register-tiled kernel, AVX-512 improved absolute throughput over AVX2 by about
1.78x to 1.91x across the larger matrix sizes.

The register-tile sweep selected eight rows as the strongest tested configuration. At
size 1024, the eight-row kernel was 4.10x faster than the one-row SIMD baseline under
AVX-512.

### Decomposition results

Production decomposition results at size 512 are shown below. For these average-time
benchmarks, speedup is scalar time divided by SIMD time, and the final column is AVX2 SIMD
time divided by AVX-512 SIMD time.

| Operation | AVX2 vs scalar | AVX-512 vs scalar | AVX-512 vs AVX2 SIMD time |
|---|---:|---:|---:|
| Matrix inverse | 4.81x | 6.39x | 1.32x |
| LU decomposition | 1.32x | 1.53x | 1.17x |
| Linear solve | 1.34x | 1.61x | 1.22x |

At size 1024, the configured block sizes improved LU by 1.64x on AVX2 and 1.95x on
AVX-512 relative to unblocked SIMD. Configured QR blocking improved performance by 1.41x
and 1.79x, respectively. Experimental Cholesky blocking reached only 0.80x on AVX2 and
0.70x on AVX-512, so the unblocked Cholesky implementation remains preferable on this
machine.

### External library comparisons

At vector length 16384, the measured JavaSIMDLinalg SIMD throughput divided by the
reference-library throughput was:

| Reference | Operation | AVX2 ratio | AVX-512 ratio |
|---|---|---:|---:|
| Apache Commons Math | add | 3.88x | 3.87x |
| Apache Commons Math | dot | 7.95x | 13.43x |
| Apache Commons Math | norm | 7.95x | 15.07x |
| EJML | add | 1.64x | 1.73x |
| EJML | dot | 7.34x | 12.98x |
| EJML | norm | 11.08x | 22.15x |

These are contextual comparisons, not a definitive ranking. JavaSIMDLinalg uses
single-precision `float[]`; the comparison benchmarks use Commons Math `ArrayRealVector`
and EJML `SimpleMatrix`, both backed by double-precision data and different allocation and
abstraction costs. The workloads therefore do not have identical representation or
precision.

### Benchmark conclusions

The main conclusions from this benchmark snapshot are:

- compute-bound vector reductions benefit strongly from wider SIMD;
- allocation and memory bandwidth limit simple element-wise operations;
- matrix multiplication benefits more from combining SIMD with register reuse, cache
  locality, and parallelism than from SIMD width alone;
- decomposition speedups are smaller because pivoting, dependencies, scalar control flow,
  and allocation limit vectorization;
- blocking is algorithm-specific and can reduce performance when reuse does not offset
  its overhead.

## Reproducing the benchmarks

Build the executable benchmark JAR first:

```text
mvn clean package
```

Verify the vector width before collecting measurements.

AVX2:

```text
java "-XX:UseAVX=2" "-Dsimdlinalg.expectedLanes=8" --add-modules jdk.incubator.vector -cp target/simd-linalg-1.0-SNAPSHOT-benchmarks.jar com.BenchmarkEnvironment
```

AVX-512:

```text
java "-XX:UseAVX=3" "-Dsimdlinalg.expectedLanes=16" --add-modules jdk.incubator.vector -cp target/simd-linalg-1.0-SNAPSHOT-benchmarks.jar com.BenchmarkEnvironment
```

Each benchmark setup repeats this lane assertion inside the forked JVM. Use the following
JMH command template:

```text
java --add-modules jdk.incubator.vector -jar target/simd-linalg-1.0-SNAPSHOT-benchmarks.jar "<include-regex>" -rf json -rff "<output-file>" -jvmArgsAppend "--add-modules jdk.incubator.vector -XX:UseAVX=<2-or-3> -Dsimdlinalg.expectedLanes=<8-or-16>"
```

The archived data was split into these suites:

| Suite | Include regular expression |
|---|---|
| Core | `MatrixOpsBenchmark|VectorOpsBenchmark` |
| Comparisons | `CommonsMathComparisonBenchmark|EJMLComparisonBenchmark` |
| Matrix multiplication | `ParallelMatrixMultiplyBenchmark|RegisterTileSweepBenchmark` |
| Decompositions | `BlockedCholeskyBenchmark|BlockedLuBenchmark|BlockedQrBenchmark|DecompositionBenchmark` |

For a controlled comparison, close unrelated applications, keep the power and thermal
configuration stable, and run AVX2 and AVX-512 under comparable conditions. Preserve the
complete JSON instead of copying only the final score.

## Repository layout

```text
src/main/java/com/
  applications/    example models built on the public API
  core/            dispatcher, validation, and backend interfaces
  decomp/          immutable decomposition results
  matrix/          public matrix facade
  performance/     tiled, blocked, and parallel kernels
  scalar/          scalar reference backends
  simd/            Java Vector API backends
  vector/          public vector facade

src/test/java/      JUnit 5 tests
src/benchmark/java/ JMH benchmarks and benchmark-only baselines
benchmark-results/  raw JDK 24 AVX2 and AVX-512 JSON measurements
```

## Product maturity, scope, and limitations

Within its documented scope, JavaSIMDLinalg is buildable, tested, packaged, and reusable as
a local Maven dependency. The snapshot version allows the API and performance thresholds
to evolve before a formal release. The following constraints define the current product
boundary and the limits of the experimental conclusions:

- The project supports dense, single-precision data only.
- The `float[][]` matrix representation is simple but has more indirection than a flat
  contiguous buffer.
- The Java Vector API is incubating and requires an explicit module flag.
- Backend selection is based on the preferred vector width, not on runtime autotuning.
- Block sizes and routing thresholds were tuned on one processor and may not transfer to
  other microarchitectures.
- The benchmark snapshot covers one Windows machine. RAM configuration, dynamic CPU
  frequency, temperature, and system background activity were not recorded in the JSON.
- External library comparisons differ in precision and representation.
- The example regression model uses normal equations and is educational, not a
  numerically optimal general-purpose regression implementation.

The product roadmap and research follow-up include Maven Central publication, multi-machine
replication, automatic threshold tuning, flat or foreign-memory matrix storage, additional
element types, sparse matrices, and a controlled rerun of the anomalous AVX2 cache-blocked
measurement.

## License

Licensed under the [Apache License 2.0](LICENSE).

Copyright (c) 2026 Giorgi Bendianishvili.

## Acknowledgements

The implementation uses the [Java Vector API](https://openjdk.org/jeps/448). Performance
measurements use [JMH](https://github.com/openjdk/jmh). Contextual comparisons use
[EJML](https://ejml.org/) and
[Apache Commons Math](https://commons.apache.org/proper/commons-math/).