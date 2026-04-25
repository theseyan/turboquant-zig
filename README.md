# TurboQuant

A Zig implementation of Google's TurboQuant vector compression library based on the paper "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate".

Based on [botirk38/turboquant](https://github.com/botirk38/turboquant) with many improvements.

## Features

- **Paper-aligned codec** - random rotation + Lloyd-Max scalar quantization + QJL residual
- **Configurable bitwidth** - set `bits_per_dim` per engine instance
- **Fast dot product** - Estimate inner products without full decode
- **QJL** - Quantized Johnson-Lindenstrauss for residual inner product estimation
- **Structured projection** - Power-of-two dimensions use a storage-free Rademacher-Hadamard QJL projection
- **SIMD optimized** - Uses Zig's portable `@Vector` for ARM64 NEON
- **Engine-based API** - Precompute state for repeated operations

## Quick Start

```zig
const turboquant = @import("turboquant");

// Create an engine for repeated operations
var engine = try turboquant.Engine.init(allocator, .{
    .dim = 1024,
    .seed = 12345,
    .bits_per_dim = 4,
});
defer engine.deinit(allocator);

// Encode a vector
const compressed = try engine.encode(allocator, my_vector);
defer allocator.free(compressed);

// Decode it back
const decoded = try engine.decode(allocator, compressed);
defer allocator.free(decoded);

// Fast dot product without full decode
const score = engine.dot(query_vector, compressed);

// Precompute query transforms when scoring many compressed vectors
var prepared = try engine.prepareQuery(allocator, query_vector);
defer prepared.deinit(allocator);
const batch_score = engine.dotPrepared(prepared, compressed);
```

## API

```zig
pub const EngineConfig = struct {
    dim: usize,
    seed: u32,
    bits_per_dim: u8 = 4,
};

pub const Engine = struct {
    pub fn init(allocator: std.mem.Allocator, config: EngineConfig) !Engine
    pub fn deinit(e: *Engine, allocator: std.mem.Allocator) void
    pub fn encode(e: *Engine, allocator: std.mem.Allocator, x: []const f32) ![]u8
    pub fn decode(e: *Engine, allocator: std.mem.Allocator, compressed: []const u8) ![]f32
    pub fn dot(e: *Engine, q: []const f32, compressed: []const u8) f32
    pub fn prepareQuery(e: *const Engine, allocator: std.mem.Allocator, q: []const f32) !PreparedQuery
    pub fn dotPrepared(e: *const Engine, query: PreparedQuery, compressed: []const u8) f32
};
```

## Performance

![Performance](docs/assets/performance.png)

At dim=1024, the default power-of-two path avoids storing the QJL dense random matrix and uses a Rademacher-Hadamard projection instead. This preserves the residual sign-projection API while reducing QJL projection cost from dense matrix-vector work to a structured transform.

### Structured QJL

For power-of-two dimensions, the QJL residual projection uses a seeded Rademacher-Hadamard transform instead of a stored dense Gaussian matrix. Non-power-of-two dimensions keep the dense Gaussian fallback, so dimensions such as 384 and 1000 produce identical quality metrics to the dense implementation.

Measured against the dense baseline with `N=1000`, `k=10`, `queries=50`, and `bits_per_dim=4`:

| Config | Dense recall@10 | Structured recall@10 | Dense top1-in-top10 | Structured top1-in-top10 |
|---|---:|---:|---:|---:|
| 128d uniform | 0.774 | 0.850 | 0.980 | 1.000 |
| 256d uniform | 0.730 | 0.826 | 0.960 | 1.000 |
| 512d uniform | 0.726 | 0.848 | 1.000 | 1.000 |
| 1024d uniform | 0.756 | 0.840 | 1.000 | 1.000 |
| 512d gaussian | 0.412 | 0.664 | 0.860 | 0.980 |
| 1024d gaussian | 0.352 | 0.508 | 0.760 | 0.820 |
| 512d sparse | 0.790 | 0.858 | 1.000 | 1.000 |
| 1024d sparse | 0.772 | 0.836 | 1.000 | 1.000 |

Dedicated microbenchmarks show the main runtime gains in encode, decode, and one-off `dot`, where QJL projection work is on the hot path. Prepared per-vector scoring is roughly flat because `prepareQuery` already precomputes the query-side QJL projection.

| Op | 128d dense -> structured | 512d dense -> structured | 1024d dense -> structured |
|---|---:|---:|---:|
| encode | 6534 -> 5882 ns | 47932 -> 34454 ns | 178499 -> 126821 ns |
| decode | 1748 -> 1139 ns | 27262 -> 15968 ns | 119162 -> 58114 ns |
| dot | 1433 -> 852 ns | 28152 -> 14817 ns | 106341 -> 56442 ns |

### Batch Search

Use `dot` for one-off scores. For retrieval workloads, prepare the query once and reuse it:

```zig
var prepared = try engine.prepareQuery(allocator, query_vector);
defer prepared.deinit(allocator);

while (try cursor.next()) |compressed_vector| {
    const score = engine.dotPrepared(prepared, compressed_vector);
    // keep top-k results
}
```

This avoids recomputing the query rotation and QJL projection for every stored vector. It is the intended path for RAG candidate scoring, and reranking batches returned by another index. The regular `dot` API remains useful for scoring a single compressed vector.

Benchmark support:

```bash
zig build bench-engine -Doptimize=ReleaseFast -- dot 1024 10000 4
zig build bench-engine -Doptimize=ReleaseFast -- prepared-dot 1024 10000 4
```

## Compression

![Compression Ratio](docs/assets/compression-ratio.png)

![Bits per Dimension](docs/assets/bits-per-dimension.png)

## Quality

![MSE Distortion](docs/assets/mse-distortion.png)

TurboQuant reconstructs in the original basis and uses a separate QJL residual stage for dot estimation.

For power-of-two dimensions, the residual stage uses a seeded Rademacher-Hadamard projection instead of a stored dense Gaussian matrix. Non-power-of-two dimensions keep the dense projection fallback. Validate any projection change with:

```bash
zig build quality -Doptimize=ReleaseFast -- 1024 1000 10 50 4
zig build quality -Doptimize=ReleaseFast -- 1024 1000 10 50 4 gaussian
zig build quality -Doptimize=ReleaseFast -- 1024 1000 10 50 4 sparse
```

![Recall@k](docs/assets/recall-at-k.png)

Run `zig build quality -- <dim> [N] [k] [num_queries] [bits_per_dim] [uniform|gaussian|sparse]` to measure top-k intersection recall and top-1-in-top-k for the current build.

![Component Analysis](docs/assets/component-analysis.png)

The residual stage corrects the scalar-quantizer bias for inner-product estimation.

![Dot Product Error](docs/assets/dot-product-error.png)

Inner-product error decreases with dimension, with QJL keeping the estimator unbiased.

## Building

```bash
cd turboquant
zig build test

# Run quality benchmarks
zig build quality -- <dim> [N] [k] [num_queries] [bits_per_dim] [uniform|gaussian|sparse]
```

## Binary Format

```
Header (22 bytes):
- version: u8
- dim: u32
- bits_per_dim: u8
- scalar_bytes: u32
- qjl_bytes: u32
- vector_norm: f32
- gamma: f32

Payload:
- scalar quantizer indices (variable)
- qjl encoded data (variable)
```

## License

MIT
