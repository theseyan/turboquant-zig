# TurboQuant

A Zig implementation of Google's TurboQuant vector compression library.

## Installation

Add to your `build.zig.zon`:

```zig
.{
    .name = "your-project",
    .dependencies = .{
        .turboquant = .{
            .url = "https://github.com/botirk38/turboquant/archive/refs/tags/v0.1.0.tar.gz",
            .hash = "sha256-...",
        },
    },
}
```

To get the hash, run `zig fetch --save https://github.com/botirk38/turboquant/archive/refs/tags/v0.1.0.tar.gz` after adding the URL, and Zig will provide the correct hash.

Or use the latest version from the main branch:

```zig
.{
    .name = "your-project",
    .dependencies = .{
        .turboquant = .{
            .url = "https://github.com/botirk38/turboquant/archive/refs/heads/master.tar.gz",
            .hash = "sha256-...",
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

This avoids recomputing the query rotation and QJL projection for every stored vector. It is the intended path for RAG candidate scoring, LMDB/RocksDB-backed scans, and reranking batches returned by another index.

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
zig build quality -- <dim> [N] [k] [num_queries] [bits_per_dim]
```

## License

MIT
