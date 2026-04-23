const std = @import("std");
const turboquant = @import("turboquant");
const math = turboquant.math;

const QualityError = error{
    MissingArgs,
    InvalidDim,
    InvalidParam,
};

const Config = struct {
    dim: usize,
    n: usize,
    k: usize,
    num_queries: usize,
    bits_per_dim: u8,
    seed: u32,
};

const IndexScore = struct {
    idx: usize,
    score: f32,
};

fn parseArgs(args: []const [:0]const u8) QualityError!Config {
    if (args.len < 2) return QualityError.MissingArgs;

    const dim = std.fmt.parseInt(usize, std.mem.sliceTo(args[1], 0), 10) catch return QualityError.InvalidDim;
    if (dim == 0) return QualityError.InvalidDim;

    const n: usize = if (args.len > 2) std.fmt.parseInt(usize, std.mem.sliceTo(args[2], 0), 10) catch return QualityError.InvalidParam else 1000;
    const k: usize = if (args.len > 3) std.fmt.parseInt(usize, std.mem.sliceTo(args[3], 0), 10) catch return QualityError.InvalidParam else 10;
    const num_queries: usize = if (args.len > 4) std.fmt.parseInt(usize, std.mem.sliceTo(args[4], 0), 10) catch return QualityError.InvalidParam else 50;
    const bits_per_dim: u8 = if (args.len > 5) std.fmt.parseInt(u8, std.mem.sliceTo(args[5], 0), 10) catch return QualityError.InvalidParam else 4;

    if (n == 0 or k == 0 or k > n or num_queries == 0 or bits_per_dim == 0 or bits_per_dim > 8) {
        return QualityError.InvalidParam;
    }

    return .{
        .dim = dim,
        .n = n,
        .k = k,
        .num_queries = num_queries,
        .bits_per_dim = bits_per_dim,
        .seed = 42,
    };
}

fn scoreDesc(_: void, a: IndexScore, b: IndexScore) bool {
    return a.score > b.score;
}

fn recallAtK(true_scores: []const IndexScore, est_scores: []const IndexScore, k: usize) f64 {
    var hits: usize = 0;
    for (true_scores[0..k]) |true_score| {
        for (est_scores[0..k]) |est_score| {
            if (true_score.idx == est_score.idx) {
                hits += 1;
                break;
            }
        }
    }
    return @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(k));
}

fn top1InTopK(true_scores: []const IndexScore, est_scores: []const IndexScore, k: usize) f64 {
    const true_best = true_scores[0].idx;
    for (est_scores[0..k]) |est_score| {
        if (est_score.idx == true_best) return 1.0;
    }
    return 0.0;
}

fn generateUnitSphere(allocator: std.mem.Allocator, dim: usize, rng: *std.Random.DefaultPrng) ![]f32 {
    const vec = try allocator.alloc(f32, dim);
    errdefer allocator.free(vec);

    const random = rng.random();
    var norm_sq: f32 = 0;
    for (vec) |*value| {
        value.* = random.float(f32) * 2 - 1;
        norm_sq += value.* * value.*;
    }

    const inv_norm = 1.0 / @sqrt(norm_sq);
    for (vec) |*value| {
        value.* *= inv_norm;
    }

    return vec;
}

pub fn main() void {
    const allocator = std.heap.page_allocator;

    const args = std.process.argsAlloc(allocator) catch {
        std.debug.print("error: out of memory\n", .{});
        return;
    };
    defer std.process.argsFree(allocator, args);

    const config = parseArgs(args) catch |err| {
        switch (err) {
            QualityError.MissingArgs => {
                std.debug.print("Usage: quality <dim> [N] [k] [num_queries] [bits_per_dim]\n", .{});
            },
            QualityError.InvalidDim => std.debug.print("error: invalid dimension\n", .{}),
            QualityError.InvalidParam => std.debug.print("error: invalid parameter\n", .{}),
        }
        return;
    };

    var engine = turboquant.Engine.init(allocator, .{
        .dim = config.dim,
        .seed = config.seed,
        .bits_per_dim = config.bits_per_dim,
    }) catch {
        std.debug.print("error: engine init failed\n", .{});
        return;
    };
    defer engine.deinit(allocator);

    const db_vecs = allocator.alloc([]f32, config.n) catch unreachable;
    defer allocator.free(db_vecs);

    const db_compressed = allocator.alloc([]u8, config.n) catch unreachable;
    defer allocator.free(db_compressed);

    var db_rng = std.Random.DefaultPrng.init(config.seed);
    var encode_timer = std.time.Timer.start() catch unreachable;
    for (0..config.n) |i| {
        db_vecs[i] = generateUnitSphere(allocator, config.dim, &db_rng) catch unreachable;
        db_compressed[i] = engine.encode(allocator, db_vecs[i]) catch unreachable;
    }
    const encode_ns = encode_timer.read();

    const true_scores = allocator.alloc(IndexScore, config.n) catch unreachable;
    defer allocator.free(true_scores);

    const est_scores = allocator.alloc(IndexScore, config.n) catch unreachable;
    defer allocator.free(est_scores);

    var total_recall: f64 = 0;
    var total_top1_in_topk: f64 = 0;
    var total_abs_dot_error: f64 = 0;
    var query_ns_total: u64 = 0;

    var query_rng = std.Random.DefaultPrng.init(config.seed + 1000);
    for (0..config.num_queries) |_| {
        const q = generateUnitSphere(allocator, config.dim, &query_rng) catch unreachable;

        for (0..config.n) |i| {
            true_scores[i] = .{
                .idx = i,
                .score = math.dot(q, db_vecs[i]),
            };
        }

        var timer = std.time.Timer.start() catch unreachable;
        for (0..config.n) |i| {
            est_scores[i] = .{
                .idx = i,
                .score = engine.dot(q, db_compressed[i]),
            };
            total_abs_dot_error += @abs(@as(f64, true_scores[i].score) - @as(f64, est_scores[i].score));
        }
        query_ns_total += timer.read();

        std.mem.sort(IndexScore, true_scores, {}, scoreDesc);
        std.mem.sort(IndexScore, est_scores, {}, scoreDesc);

        total_recall += recallAtK(true_scores, est_scores, config.k);
        total_top1_in_topk += top1InTopK(true_scores, est_scores, config.k);
        allocator.free(q);
    }

    for (0..config.n) |i| {
        allocator.free(db_vecs[i]);
        allocator.free(db_compressed[i]);
    }

    const total_payload_bits = blk: {
        if (config.n == 0) break :blk 0.0;
        const payload_bytes = db_compressed[0].len - turboquant.format.HEADER_SIZE;
        break :blk @as(f64, @floatFromInt(payload_bytes * 8)) / @as(f64, @floatFromInt(config.dim));
    };

    const nq = @as(f64, @floatFromInt(config.num_queries));
    const recall = total_recall / nq;
    const top1_in_topk = total_top1_in_topk / nq;
    const mean_abs_dot_error = total_abs_dot_error / (nq * @as(f64, @floatFromInt(config.n)));
    const encode_ms = @as(f64, @floatFromInt(encode_ns)) / 1_000_000.0;
    const query_ms = @as(f64, @floatFromInt(query_ns_total)) / 1_000_000.0;

    std.debug.print("=== QUALITY ===\n", .{});
    std.debug.print("dim={}, N={}, queries={}, bits/dim≈{d:.3}\n", .{ config.dim, config.n, config.num_queries, total_payload_bits });
    std.debug.print("encode total: {d:.1}ms ({d:.1}us/vec)\n", .{ encode_ms, encode_ms * 1000.0 / @as(f64, @floatFromInt(config.n)) });
    std.debug.print("query total:  {d:.1}ms ({d:.1}us/query)\n", .{ query_ms, query_ms * 1000.0 / nq });
    std.debug.print("recall@{}:    {d:.4}\n", .{ config.k, recall });
    std.debug.print("top1-in-top{}: {d:.4}\n", .{ config.k, top1_in_topk });
    std.debug.print("mean |error|: {e}\n", .{@as(f32, @floatCast(mean_abs_dot_error))});
}

test "parseArgs rejects invalid benchmark params" {
    const zero_n = [_][:0]const u8{ "quality", "128", "0", "10", "50", "4" };
    try std.testing.expectError(QualityError.InvalidParam, parseArgs(&zero_n));

    const zero_k = [_][:0]const u8{ "quality", "128", "100", "0", "50", "4" };
    try std.testing.expectError(QualityError.InvalidParam, parseArgs(&zero_k));

    const too_large_k = [_][:0]const u8{ "quality", "128", "100", "101", "50", "4" };
    try std.testing.expectError(QualityError.InvalidParam, parseArgs(&too_large_k));

    const zero_queries = [_][:0]const u8{ "quality", "128", "100", "10", "0", "4" };
    try std.testing.expectError(QualityError.InvalidParam, parseArgs(&zero_queries));

    const invalid_bits = [_][:0]const u8{ "quality", "128", "100", "10", "50", "9" };
    try std.testing.expectError(QualityError.InvalidParam, parseArgs(&invalid_bits));
}

test "recallAtK measures top-k intersection" {
    const truth = [_]IndexScore{
        .{ .idx = 1, .score = 0.9 },
        .{ .idx = 2, .score = 0.8 },
        .{ .idx = 3, .score = 0.7 },
        .{ .idx = 4, .score = 0.6 },
    };
    const estimated = [_]IndexScore{
        .{ .idx = 5, .score = 0.95 },
        .{ .idx = 2, .score = 0.85 },
        .{ .idx = 3, .score = 0.75 },
        .{ .idx = 1, .score = 0.65 },
    };

    try std.testing.expectApproxEqAbs(2.0 / 3.0, recallAtK(&truth, &estimated, 3), 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), top1InTopK(&truth, &estimated, 3));
    try std.testing.expectEqual(@as(f64, 1.0), top1InTopK(&truth, &estimated, 4));
}
