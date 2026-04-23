const std = @import("std");
const turboquant = @import("turboquant.zig");

const BenchError = error{
    MissingArgs,
    InvalidOp,
    InvalidDim,
    InvalidIterations,
    InvalidBitsPerDim,
};

const Operation = enum {
    encode,
    decode,
    dot,
    prepared_dot,
    compression,
    all,
};

const Config = struct {
    op: Operation,
    dim: usize,
    iterations: usize,
    bits_per_dim: u8,
    seed: u32,
};

fn parseArgs(args: []const [:0]u8) BenchError!Config {
    if (args.len < 2) return BenchError.MissingArgs;

    const op_str = std.mem.sliceTo(args[1], 0);
    const op: Operation = if (std.mem.eql(u8, op_str, "encode")) .encode else if (std.mem.eql(u8, op_str, "decode")) .decode else if (std.mem.eql(u8, op_str, "dot")) .dot else if (std.mem.eql(u8, op_str, "prepared-dot")) .prepared_dot else if (std.mem.eql(u8, op_str, "compression")) .compression else if (std.mem.eql(u8, op_str, "all")) .all else return BenchError.InvalidOp;

    var dim: usize = 128;
    var iterations: usize = 100;
    var bits_per_dim: u8 = 4;

    if (args.len > 2) {
        dim = std.fmt.parseInt(usize, std.mem.sliceTo(args[2], 0), 10) catch return BenchError.InvalidDim;
        if (dim == 0) return BenchError.InvalidDim;
    }

    if (args.len > 3) {
        iterations = std.fmt.parseInt(usize, std.mem.sliceTo(args[3], 0), 10) catch return BenchError.InvalidIterations;
        if (iterations == 0) return BenchError.InvalidIterations;
    }

    if (args.len > 4) {
        bits_per_dim = std.fmt.parseInt(u8, std.mem.sliceTo(args[4], 0), 10) catch return BenchError.InvalidBitsPerDim;
        if (bits_per_dim == 0 or bits_per_dim > 8) return BenchError.InvalidBitsPerDim;
    }

    return .{
        .op = op,
        .dim = dim,
        .iterations = iterations,
        .bits_per_dim = bits_per_dim,
        .seed = 12345,
    };
}

fn generateVector(allocator: std.mem.Allocator, dim: usize, seed: u32) ![]f32 {
    const data = try allocator.alloc(f32, dim);
    errdefer allocator.free(data);

    var rng = std.Random.DefaultPrng.init(seed);
    const random = rng.random();
    for (data) |*value| value.* = random.float(f32) * 10 - 5;
    return data;
}

fn runEncode(allocator: std.mem.Allocator, dim: usize, iterations: usize, bits_per_dim: u8, seed: u32) !u64 {
    var engine = try turboquant.Engine.init(allocator, .{ .dim = dim, .seed = seed, .bits_per_dim = bits_per_dim });
    defer engine.deinit(allocator);

    const data = try generateVector(allocator, dim, seed);
    defer allocator.free(data);

    const out = try allocator.alloc(u8, engine.compressedLen());
    defer allocator.free(out);

    var total_ns: u64 = 0;
    for (0..iterations) |_| {
        var timer = try std.time.Timer.start();
        try engine.encodeInto(out, data);
        total_ns += timer.read();
    }
    return total_ns / iterations;
}

fn runDecode(allocator: std.mem.Allocator, dim: usize, iterations: usize, bits_per_dim: u8, seed: u32) !u64 {
    var engine = try turboquant.Engine.init(allocator, .{ .dim = dim, .seed = seed, .bits_per_dim = bits_per_dim });
    defer engine.deinit(allocator);

    const data = try generateVector(allocator, dim, seed);
    defer allocator.free(data);

    const compressed = try allocator.alloc(u8, engine.compressedLen());
    defer allocator.free(compressed);
    try engine.encodeInto(compressed, data);

    const decoded = try allocator.alloc(f32, dim);
    defer allocator.free(decoded);

    var total_ns: u64 = 0;
    for (0..iterations) |_| {
        var timer = try std.time.Timer.start();
        try engine.decodeInto(decoded, compressed);
        total_ns += timer.read();
    }
    return total_ns / iterations;
}

fn runDot(allocator: std.mem.Allocator, dim: usize, iterations: usize, bits_per_dim: u8, seed: u32) !u64 {
    var engine = try turboquant.Engine.init(allocator, .{ .dim = dim, .seed = seed, .bits_per_dim = bits_per_dim });
    defer engine.deinit(allocator);

    const data = try generateVector(allocator, dim, seed);
    defer allocator.free(data);

    const query = try generateVector(allocator, dim, seed + 1);
    defer allocator.free(query);

    const compressed = try allocator.alloc(u8, engine.compressedLen());
    defer allocator.free(compressed);
    try engine.encodeInto(compressed, data);

    var total_ns: u64 = 0;
    for (0..iterations) |_| {
        var timer = try std.time.Timer.start();
        _ = engine.dot(query, compressed);
        total_ns += timer.read();
    }
    return total_ns / iterations;
}

fn runPreparedDot(allocator: std.mem.Allocator, dim: usize, iterations: usize, bits_per_dim: u8, seed: u32) !u64 {
    var engine = try turboquant.Engine.init(allocator, .{ .dim = dim, .seed = seed, .bits_per_dim = bits_per_dim });
    defer engine.deinit(allocator);

    const data = try generateVector(allocator, dim, seed);
    defer allocator.free(data);

    const query = try generateVector(allocator, dim, seed + 1);
    defer allocator.free(query);

    const compressed = try allocator.alloc(u8, engine.compressedLen());
    defer allocator.free(compressed);
    try engine.encodeInto(compressed, data);

    var prepared = try engine.prepareQuery(allocator, query);
    defer prepared.deinit(allocator);

    var total_ns: u64 = 0;
    for (0..iterations) |_| {
        var timer = try std.time.Timer.start();
        _ = engine.dotPrepared(prepared, compressed);
        total_ns += timer.read();
    }
    return total_ns / iterations;
}

fn runCompression(allocator: std.mem.Allocator, dim: usize, bits_per_dim: u8, seed: u32) !void {
    var engine = try turboquant.Engine.init(allocator, .{ .dim = dim, .seed = seed, .bits_per_dim = bits_per_dim });
    defer engine.deinit(allocator);

    const data = try generateVector(allocator, dim, seed);
    defer allocator.free(data);

    const compressed = try allocator.alloc(u8, engine.compressedLen());
    defer allocator.free(compressed);
    try engine.encodeInto(compressed, data);

    const raw_bytes = dim * 4;
    const ratio = @as(f64, @floatFromInt(raw_bytes)) / @as(f64, @floatFromInt(compressed.len));
    const bits = @as(f64, @floatFromInt(compressed.len * 8)) / @as(f64, @floatFromInt(dim));
    std.debug.print("{d} | {d} | {d:.2}x | {d:.2}\n", .{ dim, compressed.len, ratio, bits });
}

pub fn main() void {
    const allocator = std.heap.page_allocator;
    const args = std.process.argsAlloc(allocator) catch {
        std.debug.print("error: out of memory parsing args\n", .{});
        return;
    };
    defer std.process.argsFree(allocator, args);

    const config = parseArgs(args) catch |err| {
        switch (err) {
            BenchError.MissingArgs => {
                std.debug.print("Usage: bench <op> [dim] [iterations] [bits_per_dim]\n", .{});
                std.debug.print("  op: encode, decode, dot, prepared-dot, compression, all\n", .{});
            },
            BenchError.InvalidOp => std.debug.print("error: invalid operation\n", .{}),
            BenchError.InvalidDim => std.debug.print("error: invalid dimension\n", .{}),
            BenchError.InvalidIterations => std.debug.print("error: invalid iterations\n", .{}),
            BenchError.InvalidBitsPerDim => std.debug.print("error: invalid bits_per_dim\n", .{}),
        }
        return;
    };

    switch (config.op) {
        .encode => {
            const ns = runEncode(allocator, config.dim, config.iterations, config.bits_per_dim, config.seed) catch {
                std.debug.print("error: benchmark failed\n", .{});
                return;
            };
            std.debug.print("encode/dim={d}/bpd={d}: {d} ns/op\n", .{ config.dim, config.bits_per_dim, ns });
        },
        .decode => {
            const ns = runDecode(allocator, config.dim, config.iterations, config.bits_per_dim, config.seed) catch {
                std.debug.print("error: benchmark failed\n", .{});
                return;
            };
            std.debug.print("decode/dim={d}/bpd={d}: {d} ns/op\n", .{ config.dim, config.bits_per_dim, ns });
        },
        .dot => {
            const ns = runDot(allocator, config.dim, config.iterations, config.bits_per_dim, config.seed) catch {
                std.debug.print("error: benchmark failed\n", .{});
                return;
            };
            std.debug.print("dot/dim={d}/bpd={d}: {d} ns/op\n", .{ config.dim, config.bits_per_dim, ns });
        },
        .prepared_dot => {
            const ns = runPreparedDot(allocator, config.dim, config.iterations, config.bits_per_dim, config.seed) catch {
                std.debug.print("error: benchmark failed\n", .{});
                return;
            };
            std.debug.print("prepared-dot/dim={d}/bpd={d}: {d} ns/op\n", .{ config.dim, config.bits_per_dim, ns });
        },
        .compression => {
            const dims = [_]usize{ 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 };
            std.debug.print("\n=== COMPRESSION RATIOS ===\n", .{});
            std.debug.print("{s:>4} | {s:>6} | {s:>6} | {s:>8}\n", .{ "dim", "compressed", "ratio", "bits/dim" });
            std.debug.print("------|----------|----------|----------\n", .{});
            for (dims) |dim| {
                runCompression(allocator, dim, config.bits_per_dim, config.seed) catch {
                    std.debug.print("error: compression benchmark failed\n", .{});
                    return;
                };
            }
        },
        .all => {
            const dims = [_]usize{ 8, 16, 32, 64, 128, 256, 512, 1024 };

            std.debug.print("=== ENCODE BENCHMARK ===\n", .{});
            std.debug.print("{s:>4} | {s:>12}\n", .{ "dim", "ns/op" });
            std.debug.print("------|------------\n", .{});
            for (dims) |dim| {
                const ns = runEncode(allocator, dim, config.iterations, config.bits_per_dim, config.seed) catch continue;
                std.debug.print("{d:>4} | {d:>12}\n", .{ dim, ns });
            }

            std.debug.print("\n=== DECODE BENCHMARK ===\n", .{});
            std.debug.print("{s:>4} | {s:>12}\n", .{ "dim", "ns/op" });
            std.debug.print("------|------------\n", .{});
            for (dims) |dim| {
                const ns = runDecode(allocator, dim, config.iterations, config.bits_per_dim, config.seed) catch continue;
                std.debug.print("{d:>4} | {d:>12}\n", .{ dim, ns });
            }

            std.debug.print("\n=== DOT BENCHMARK ===\n", .{});
            std.debug.print("{s:>4} | {s:>12}\n", .{ "dim", "ns/op" });
            std.debug.print("------|------------\n", .{});
            for (dims) |dim| {
                const ns = runDot(allocator, dim, config.iterations, config.bits_per_dim, config.seed) catch continue;
                std.debug.print("{d:>4} | {d:>12}\n", .{ dim, ns });
            }

            std.debug.print("\n=== PREPARED DOT BENCHMARK ===\n", .{});
            std.debug.print("{s:>4} | {s:>12}\n", .{ "dim", "ns/op" });
            std.debug.print("------|------------\n", .{});
            for (dims) |dim| {
                const ns = runPreparedDot(allocator, dim, config.iterations, config.bits_per_dim, config.seed) catch continue;
                std.debug.print("{d:>4} | {d:>12}\n", .{ dim, ns });
            }

            std.debug.print("\n=== COMPRESSION RATIOS ===\n", .{});
            std.debug.print("{s:>4} | {s:>6} | {s:>6} | {s:>8}\n", .{ "dim", "compressed", "ratio", "bits/dim" });
            std.debug.print("------|----------|----------|----------\n", .{});
            for (dims) |dim| {
                runCompression(allocator, dim, config.bits_per_dim, config.seed) catch continue;
            }
        },
    }
}
