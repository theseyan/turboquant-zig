# TurboQuant

A Zig implementation of Google's TurboQuant vector compression library.

## Installation

Add to your `build.zig.zon`:

```zig
.{
    .name = "your-project",
    .dependencies = .{
        .turboquant = .{
            .url = "https://github.com/theseyan/turboquant-zig/archive/refs/tags/v0.2.0.tar.gz",
            .hash = "...",
        },
    },
}
```

To get the hash, run `zig fetch --save https://github.com/theseyan/turboquant-zig/archive/refs/tags/v0.2.0.tar.gz` after adding the URL, and Zig will provide the correct hash.

Or use the latest version from the main branch:

```zig
.{
    .name = "your-project",
    .dependencies = .{
        .turboquant = .{
            .url = "https://github.com/theseyan/turboquant-zig/archive/refs/heads/main.tar.gz",
            .hash = "...",
        },
    },
}
```

## Usage

```zig
const turboquant = @import("turboquant");

// Create an engine for repeated operations
var engine = try turboquant.Engine.init(allocator, .{
    .dim = 1024,
    .seed = 12345,
    .bits_per_dim = 4,
});
defer engine.deinit(allocator);

// Encode
const compressed = try engine.encode(allocator, my_vector);
defer allocator.free(compressed);

// Decode
const decoded = try engine.decode(allocator, compressed);
defer allocator.free(decoded);

// Fast dot without decode
const score = engine.dot(query_vector, compressed);

// Reuse query transforms across many compressed vectors
var prepared = try engine.prepareQuery(allocator, query_vector);
defer prepared.deinit(allocator);
const batch_score = engine.dotPrepared(prepared, compressed);
```

## API

- `Engine.init(allocator, .{ .dim, .seed, .bits_per_dim })` - Create engine
- `engine.deinit(allocator)` - Destroy engine
- `engine.encode(allocator, vector)` - Compress vector
- `engine.decode(allocator, compressed)` - Decompress
- `engine.dot(query, compressed)` - Dot product without full decode
- `engine.prepareQuery(allocator, query)` / `engine.dotPrepared(prepared, compressed)` - Reuse query transforms for batch scoring

## Performance

![Performance](docs/assets/performance.png)

At dim=1024, the default power-of-two path avoids storing the QJL dense random matrix and uses a Rademacher-Hadamard projection instead. This preserves the residual sign-projection API while reducing QJL projection cost from dense matrix-vector work to a structured transform.

### Structured QJL

For power-of-two dimensions, the QJL residual projection uses a seeded Rademacher-Hadamard transform instead of a stored dense Gaussian matrix. Non-power-of-two dimensions keep the dense Gaussian fallback.

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

This avoids recomputing the query rotation and QJL projection for every stored vector. It is the intended path for RAG candidate scoring, LMDB/RocksDB-backed scans, and reranking batches returned by another index.

For power-of-two dimensions, the residual stage uses a seeded Rademacher-Hadamard projection instead of a stored dense Gaussian matrix. Non-power-of-two dimensions keep the dense projection fallback. Validate any projection change with:

```bash
zig build quality -Doptimize=ReleaseFast -- 1024 1000 10 50 4
zig build quality -Doptimize=ReleaseFast -- 1024 1000 10 50 4 gaussian
zig build quality -Doptimize=ReleaseFast -- 1024 1000 10 50 4 sparse
```

Benchmark support:

```bash
zig build bench-engine -Doptimize=ReleaseFast -- dot 1024 10000 4
zig build bench-engine -Doptimize=ReleaseFast -- prepared-dot 1024 10000 4
```

## Compression

![Compression Ratio](docs/assets/compression-ratio.png)

![Bits per Dimension](docs/assets/bits-per-dimension.png)

- ~6x compression ratio at dim=1024
- Configurable payload bitrate with 1-bit QJL residual correction

## Building

```bash
cd turboquant
zig build test
zig build quality -- <dim> [N] [k] [num_queries] [bits_per_dim] [uniform|gaussian|sparse]
```

## License

MIT
