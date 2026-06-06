# JavaSIMDLinalg

**A SIMD-accelerated linear algebra library for Java, built on the Java Vector API (Project Panama).**

JavaSIMDLinalg provides a small, clean API for the most common single-precision
(`float`) linear algebra operations, backed by explicit CPU SIMD implementations using
the [Java Vector API](https://openjdk.org/jeps/426) (`jdk.incubator.vector`). Where the
running CPU supports wide vector registers (AVX2 / AVX-512), the library processes 8 or
16 `float` values per instruction; where it does not, it falls back transparently to
plain scalar loops. The backend is chosen once at start-up - calling code never changes.

The goal is to close a real gap in the Java ecosystem: unlike Python (NumPy), Rust
(`ndarray`, `faer`), or C++ (Eigen, BLAS), the JVM has no focused, native, explicit-SIMD
linear algebra library. JavaSIMDLinalg is that library, with every SIMD kernel verified
against a scalar reference and benchmarked with JMH.

---

## Features

| Category        | Operations |
|-----------------|------------|
| Vector          | dot product, L2 norm, cosine similarity, Euclidean distance |
| Element-wise    | add, subtract, multiply, divide, fused multiply-add (FMA) |
| Reductions      | sum, min, max, argmax, softmax |
| Utility         | scale, normalize, copy, fill |
| Matrix          | matrix–vector multiply, matrix–matrix multiply, add, transpose |

All operations have **both** a scalar and a SIMD implementation, selected automatically
at runtime.

---

## Requirements

- **Java 21+** (developed and benchmarked on Temurin/OpenJDK 21 LTS)
- The Vector API is an *incubator* module, so it must be enabled at both compile and run
  time with `--add-modules jdk.incubator.vector`.

---

## Installation

> The library targets publication to Maven Central (coordinates below). Until it is
> published, build it locally with `mvn install`.

```xml
<dependency>
    <groupId>com.simdlinalg</groupId>
    <artifactId>simd-linalg</artifactId>
    <version>1.0-SNAPSHOT</version>
</dependency>
```

Because the Vector API is incubating, applications that use this library must pass the
module flag when they run:

```
java --add-modules jdk.incubator.vector -cp ... your.Main
```

---

## Usage

All operations are exposed as static methods on two façade classes:
`com.vector.VectorOps` and `com.matrix.MatrixOps`.

```java
import com.vector.VectorOps;
import com.matrix.MatrixOps;

float[] a = {1f, 2f, 3f, 4f};
float[] b = {5f, 6f, 7f, 8f};

float d   = VectorOps.dot(a, b);              // 70.0
float n   = VectorOps.norm(a);                // 5.477...
float cos = VectorOps.cosineSimilarity(a, b); // 0.968...
float[] s = VectorOps.add(a, b);              // {6, 8, 10, 12}
float[] u = VectorOps.normalize(a);           // unit-length vector

float[][] m = {{1f, 2f}, {3f, 4f}};
float[]   v = {1f, 1f};
float[]   mv = MatrixOps.multiply(m, v);      // {3, 7}
float[][] mm = MatrixOps.multiply(m, m);      // matrix product
float[][] mt = MatrixOps.transpose(m);        // {{1,3},{2,4}}
```

Operations that combine two vectors require equal lengths and throw
`IllegalArgumentException` otherwise; `cosineSimilarity` and `normalize` reject
zero-norm vectors. See the Javadoc on `VectorOps` / `MatrixOps` for the exact contract
of each method.

---

## How it works

The library uses a three-layer design:

```
VectorOps / MatrixOps   (public façade - what you call)
        │
   Dispatcher           (selects a backend once, at JVM start-up)
        │
 ┌──────┴───────┐
 SIMD backend   Scalar backend
 (Vector API)   (plain Java loops)
```

At class-load time the `Dispatcher` queries `FloatVector.SPECIES_PREFERRED` for the
widest vector species the CPU and JVM support. If a usable width is available it binds
the SIMD backends; otherwise it binds the scalar backends. Each SIMD kernel follows the
standard pattern: process the array in full vector-width chunks with `FloatVector`
operations, then finish the leftover *tail* elements with a scalar loop.

---

## Performance

Benchmarks were produced with **JMH 1.37** (2 forks, 5 warm-up + 5 measurement
iterations each) on **OpenJDK 21**, on a CPU exposing **AVX-512 (16 `float` lanes)**.
Speedups are SIMD throughput relative to the scalar baseline; library comparisons are
this library's `float` SIMD against the named `double`-precision library.

**Representative results** (throughput speedup, higher is better):

| Operation                | vs scalar | vs EJML | vs Commons Math |
|--------------------------|-----------|---------|-----------------|
| dot product (n=1024)     | 17.1×     | 16.9×   | 16.1×           |
| L2 norm (n=1024)         | 20.2×     | 30.8×   | 19.0×           |
| cosine similarity (n=1024)| 18.1×    | -       | -               |
| sum (n=1024)             | 15.0×     | -       | -               |
| matrix multiply (512×512)| 15.7×     | -       | -               |
| matrix–vector (256×256)  | 9.5×      | -       | -               |
| element-wise add (n=1024)| 0.9×      | 2.4×    | 4.3×            |

**Key findings:**

- **Compute-bound operations win big.** Reduction-style kernels (dot, norm, cosine,
  sum) reach **10–20× over scalar** for vectors of 256 elements or more, comfortably
  exceeding the project's 4× target.
- **There is a breakeven point.** At very small sizes (n≈16) the fixed overhead of the
  Vector API cancels its benefit, and SIMD is roughly on par with scalar. SIMD pulls
  decisively ahead from about n≈128 onward.
- **Memory-bound operations don't speed up.** Element-wise `add`/`scale` and matrix
  `add` allocate a fresh result array and are limited by memory bandwidth, not
  arithmetic, so SIMD gives little or no gain (~1×) - as expected.
- **The library beats general-purpose `double` libraries** (EJML, Apache Commons Math)
  on these operations, partly through SIMD and partly because single precision moves
  half the bytes. This precision/bandwidth difference is a deliberate caveat, not an
  apples-to-apples comparison.

To reproduce:

```
mvn clean package -DskipTests
java --add-modules jdk.incubator.vector -jar target/simd-linalg-1.0-SNAPSHOT-benchmarks.jar -rf json -rff results.json
```

---

## Correctness & testing

Every SIMD kernel is paired with a JUnit 5 test that checks it against the scalar
reference across many sizes - including sizes that are not multiples of the vector width,
to exercise the tail-handling path. Edge cases are covered explicitly: `NaN`/infinity
propagation, zero vectors, mismatched lengths, and very large magnitudes.

```
mvn test          # run the test suite
mvn test jacoco:report   # + line-coverage report under target/site/jacoco
```

SIMD and scalar results are compared within a small tolerance rather than bit-for-bit:
floating-point addition is not associative, so summing in vector-lane order produces
results that differ from a sequential scalar sum by a few ULP at large sizes. This is
expected and documented behaviour for any SIMD reduction.

---

## Scope and limitations

In scope: single-precision (`float`) dense vector and matrix operations listed above.

Out of scope (natural extensions for future work):

- complex numbers, sparse matrices
- decompositions (LU, QR, SVD)
- `double`/integer element types
- GPU acceleration and distributed computation

---

## License

Licensed under the [Apache License 2.0](LICENSE). You are free to use, modify, and
distribute this library, including in commercial and proprietary software, subject to
the terms of the license. The Apache 2.0 license also provides an explicit patent grant,
which suits a SIMD/Vector API library intended for Maven Central publication.

Copyright © 2026 Giorgi Bendianishvili.

---

## Acknowledgements

Built on [JEP 426: Vector API](https://openjdk.org/jeps/426). Benchmarked with
[JMH](https://github.com/openjdk/jmh). Compared against
[EJML](https://ejml.org/) and [Apache Commons Math](https://commons.apache.org/proper/commons-math/).
