# TurboQuant

[![Star History Chart](https://api.star-history.com/image?repos=botirk38/turboquant&type=Date)](https://star-history.com/#botirk38/turboquant)

A Zig implementation of Google's TurboQuant vector compression library based on the paper "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate".

## Features

- **Paper-aligned codec** - random rotation + Lloyd-Max scalar quantization + QJL residual
- **Configurable bitwidth** - set `bits_per_dim` per engine instance
- **Fast dot product** - Estimate inner products without full decode
- **QJL** - Quantized Johnson-Lindenstrauss for unbiased inner product estimation
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

At dim=1024: encode 2105µs, decode 1032µs, dot 997µs

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

This avoids recomputing the query rotation and QJL projection for every stored vector. It is the intended path for RAG candidate scoring, LMDB/RocksDB-backed scans, and reranking batches returned by another index. The regular `dot` API remains useful for scoring a single compressed vector.

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

![Recall@k](docs/assets/recall-at-k.png)

Run `zig build quality -- <dim> [N] [k] [num_queries] [bits_per_dim]` to measure top-k intersection recall and top-1-in-top-k for the current build.

![Component Analysis](docs/assets/component-analysis.png)

The residual stage corrects the scalar-quantizer bias for inner-product estimation.

![Dot Product Error](docs/assets/dot-product-error.png)

Inner-product error decreases with dimension, with QJL keeping the estimator unbiased.

## Building

```bash
cd turboquant
zig build test

# Run quality benchmarks
zig build quality -- <dim> [N] [k] [num_queries] [bits_per_dim]
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
