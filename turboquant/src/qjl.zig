const std = @import("std");
const math = @import("math.zig");

pub const QjlError = error{
    BufferTooSmall,
    InvalidDimension,
    OutOfMemory,
};

const SQRT_PI_OVER_2: f32 = 1.2533141373155003;
const LANE_COUNT = std.simd.suggestVectorLength(f32) orelse 4;

fn randGaussian(random: std.Random) f32 {
    const v1 = @max(random.float(f32), std.math.floatMin(f32));
    const v2 = random.float(f32);
    return @sqrt(-2.0 * @log(v1)) * @cos(2.0 * std.math.pi * v2);
}

fn matVecKernel(matrix: []const f32, input: []const f32, output: []f32, dim: usize) void {
    var row: usize = 0;
    while (row + 1 < dim) : (row += 2) {
        const row0 = matrix[row * dim ..][0..dim];
        const row1 = matrix[(row + 1) * dim ..][0..dim];

        var acc00: @Vector(LANE_COUNT, f32) = @splat(0);
        var acc01: @Vector(LANE_COUNT, f32) = @splat(0);
        var acc10: @Vector(LANE_COUNT, f32) = @splat(0);
        var acc11: @Vector(LANE_COUNT, f32) = @splat(0);

        var i: usize = 0;
        while (i + 2 * LANE_COUNT <= dim) : (i += 2 * LANE_COUNT) {
            const in0: @Vector(LANE_COUNT, f32) = input[i..][0..LANE_COUNT].*;
            const in1: @Vector(LANE_COUNT, f32) = input[i + LANE_COUNT ..][0..LANE_COUNT].*;
            const r00: @Vector(LANE_COUNT, f32) = row0[i..][0..LANE_COUNT].*;
            const r01: @Vector(LANE_COUNT, f32) = row0[i + LANE_COUNT ..][0..LANE_COUNT].*;
            const r10: @Vector(LANE_COUNT, f32) = row1[i..][0..LANE_COUNT].*;
            const r11: @Vector(LANE_COUNT, f32) = row1[i + LANE_COUNT ..][0..LANE_COUNT].*;
            acc00 += r00 * in0;
            acc01 += r01 * in1;
            acc10 += r10 * in0;
            acc11 += r11 * in1;
        }

        var sum0 = @reduce(.Add, acc00 + acc01);
        var sum1 = @reduce(.Add, acc10 + acc11);

        while (i + LANE_COUNT <= dim) : (i += LANE_COUNT) {
            const inv: @Vector(LANE_COUNT, f32) = input[i..][0..LANE_COUNT].*;
            const r0v: @Vector(LANE_COUNT, f32) = row0[i..][0..LANE_COUNT].*;
            const r1v: @Vector(LANE_COUNT, f32) = row1[i..][0..LANE_COUNT].*;
            sum0 += @reduce(.Add, r0v * inv);
            sum1 += @reduce(.Add, r1v * inv);
        }
        while (i < dim) : (i += 1) {
            sum0 += row0[i] * input[i];
            sum1 += row1[i] * input[i];
        }

        output[row] = sum0;
        output[row + 1] = sum1;
    }

    if (row < dim) {
        output[row] = math.dot(matrix[row * dim ..][0..dim], input);
    }
}

pub const Workspace = struct {
    projected: []f32,

    pub fn init(allocator: std.mem.Allocator, dim: usize) QjlError!Workspace {
        return .{ .projected = try allocator.alloc(f32, dim) };
    }

    pub fn initInto(projected: []f32, dim: usize) QjlError!Workspace {
        if (projected.len < dim) return QjlError.BufferTooSmall;
        return .{ .projected = projected[0..dim] };
    }

    pub fn requiredStorageSize(dim: usize) usize {
        return dim * @sizeOf(f32);
    }

    pub fn deinit(w: *Workspace, allocator: std.mem.Allocator) void {
        allocator.free(w.projected);
        w.* = .{ .projected = &.{} };
    }
};

pub const ProjectionOperator = struct {
    dim: usize,
    seed: u32,
    matrix: []f32,
    matrix_t: []f32,

    pub fn prepare(allocator: std.mem.Allocator, dim: usize, seed: u32) QjlError!ProjectionOperator {
        const matrix = try allocator.alloc(f32, dim * dim);
        errdefer allocator.free(matrix);

        const matrix_t = try allocator.alloc(f32, dim * dim);
        errdefer allocator.free(matrix_t);

        return prepareInto(matrix, matrix_t, dim, seed);
    }

    pub fn prepareInto(
        matrix: []f32,
        matrix_t: []f32,
        dim: usize,
        seed: u32,
    ) QjlError!ProjectionOperator {
        if (matrix.len < dim * dim or matrix_t.len < dim * dim) return QjlError.BufferTooSmall;

        const matrix_view = matrix[0 .. dim * dim];
        const matrix_t_view = matrix_t[0 .. dim * dim];

        var rng = std.Random.DefaultPrng.init(seed);
        const random = rng.random();
        for (0..dim) |row| {
            for (0..dim) |col| {
                matrix_view[row * dim + col] = randGaussian(random);
            }
        }

        for (0..dim) |row| {
            for (0..dim) |col| {
                matrix_t_view[col * dim + row] = matrix_view[row * dim + col];
            }
        }

        return .{
            .dim = dim,
            .seed = seed,
            .matrix = matrix_view,
            .matrix_t = matrix_t_view,
        };
    }

    pub fn requiredStorageSize(dim: usize) usize {
        return 2 * dim * dim * @sizeOf(f32);
    }

    pub fn destroy(op: *ProjectionOperator, allocator: std.mem.Allocator) void {
        allocator.free(op.matrix);
        allocator.free(op.matrix_t);
        op.* = undefined;
    }

    pub fn matVecMul(op: *const ProjectionOperator, input: []const f32, output: []f32) void {
        const dim = op.dim;
        std.debug.assert(input.len == dim and output.len == dim);
        matVecKernel(op.matrix, input, output, dim);
    }

    pub fn matVecMulTransposed(op: *const ProjectionOperator, input: []const f32, output: []f32) void {
        const dim = op.dim;
        std.debug.assert(input.len == dim and output.len == dim);
        matVecKernel(op.matrix_t, input, output, dim);
    }
};

pub fn sqrtPiOver2() f32 {
    return SQRT_PI_OVER_2;
}

pub fn encodedLen(dim: usize) QjlError!usize {
    if (dim == 0) return QjlError.InvalidDimension;
    return (dim + 7) / 8;
}

pub fn encodeWithWorkspace(
    allocator: std.mem.Allocator,
    residual: []const f32,
    projection: *const ProjectionOperator,
    workspace: *Workspace,
) QjlError![]u8 {
    const out_len = try encodedLen(residual.len);
    const out = try allocator.alloc(u8, out_len);
    errdefer allocator.free(out);
    try encodeInto(out, residual, projection, workspace);
    return out;
}

pub fn encodeInto(
    out: []u8,
    residual: []const f32,
    projection: *const ProjectionOperator,
    workspace: *Workspace,
) QjlError!void {
    const dim = residual.len;
    if (dim == 0) return QjlError.InvalidDimension;

    const out_len = try encodedLen(dim);
    if (out.len < out_len) return QjlError.BufferTooSmall;
    @memset(out[0..out_len], 0);

    projection.matVecMul(residual, workspace.projected[0..dim]);
    for (workspace.projected[0..dim], 0..) |value, i| {
        if (value > 0) {
            out[i / 8] |= @as(u8, 1) << @intCast(i % 8);
        }
    }
}

pub fn decodeInto(
    out: []f32,
    qjl_bits: []const u8,
    gamma: f32,
    projection: *const ProjectionOperator,
) void {
    const dim = out.len;
    if (dim == 0) return;

    @memset(out, 0);
    decodeAddInto(out, qjl_bits, gamma, projection);
}

pub fn decodeAddInto(
    out: []f32,
    qjl_bits: []const u8,
    gamma: f32,
    projection: *const ProjectionOperator,
) void {
    const dim = out.len;
    if (dim == 0) return;

    const scale = SQRT_PI_OVER_2 * gamma / @as(f32, @floatFromInt(dim));
    for (0..dim) |row| {
        const sign: f32 = if (((qjl_bits[row / 8] >> @intCast(row % 8)) & 1) == 1) scale else -scale;
        addScaledInPlace(out, projection.matrix[row * dim ..][0..dim], sign);
    }
}

pub fn estimateDotWithWorkspace(
    q: []const f32,
    qjl_bits: []const u8,
    gamma: f32,
    projection: *const ProjectionOperator,
    workspace: *Workspace,
) f32 {
    const dim = q.len;
    if (dim == 0) return 0;

    projection.matVecMul(q, workspace.projected[0..dim]);
    return estimateDotFromProjection(workspace.projected[0..dim], qjl_bits, gamma);
}

pub fn estimateDotFromProjection(projected_q: []const f32, qjl_bits: []const u8, gamma: f32) f32 {
    const dim = projected_q.len;
    if (dim == 0) return 0;

    var dot_sum: f32 = 0;
    for (projected_q, 0..) |value, i| {
        dot_sum += if (((qjl_bits[i / 8] >> @intCast(i % 8)) & 1) == 1) value else -value;
    }

    const scale = SQRT_PI_OVER_2 * gamma / @as(f32, @floatFromInt(dim));
    return dot_sum * scale;
}

fn addScaledInPlace(dst: []f32, src: []const f32, sign: f32) void {
    std.debug.assert(dst.len == src.len);

    const sign_vec: @Vector(LANE_COUNT, f32) = @splat(sign);
    var i: usize = 0;
    while (i + LANE_COUNT <= dst.len) : (i += LANE_COUNT) {
        const dv: @Vector(LANE_COUNT, f32) = dst[i..][0..LANE_COUNT].*;
        const sv: @Vector(LANE_COUNT, f32) = src[i..][0..LANE_COUNT].*;
        @as(*[LANE_COUNT]f32, @ptrCast(dst[i..])).* = dv + sv * sign_vec;
    }
    while (i < dst.len) : (i += 1) {
        dst[i] += src[i] * sign;
    }
}

test "encodeInto rejects zero dimension" {
    const allocator = std.testing.allocator;

    var ws = try Workspace.init(allocator, 1);
    defer ws.deinit(allocator);

    var projection = try ProjectionOperator.prepare(allocator, 1, 12345);
    defer projection.destroy(allocator);

    var out: [1]u8 = undefined;
    const residual: [0]f32 = .{};
    const result = encodeInto(&out, &residual, &projection, &ws);
    try std.testing.expectError(QjlError.InvalidDimension, result);
}

test "projection is deterministic" {
    const allocator = std.testing.allocator;

    var p1 = try ProjectionOperator.prepare(allocator, 4, 12345);
    defer p1.destroy(allocator);

    var p2 = try ProjectionOperator.prepare(allocator, 4, 12345);
    defer p2.destroy(allocator);

    try std.testing.expectEqualSlices(f32, p1.matrix, p2.matrix);
}

test "encodeInto decodeInto roundtrip with gaussian projection" {
    const allocator = std.testing.allocator;
    const dim: usize = 16;

    var projection = try ProjectionOperator.prepare(allocator, dim, 12345);
    defer projection.destroy(allocator);

    var ws = try Workspace.init(allocator, dim);
    defer ws.deinit(allocator);

    const residual = [_]f32{ 1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0, 11.0, -12.0, 13.0, -14.0, 15.0, -16.0 };
    const gamma = math.norm(&residual);

    var encoded: [2]u8 = undefined;
    try encodeInto(&encoded, &residual, &projection, &ws);

    var decoded: [dim]f32 = undefined;
    decodeInto(&decoded, &encoded, gamma, &projection);

    for (decoded) |value| {
        try std.testing.expect(std.math.isFinite(value));
    }
}

test "estimateDotWithWorkspace matches decoded dot" {
    const allocator = std.testing.allocator;
    const dim: usize = 8;

    var projection = try ProjectionOperator.prepare(allocator, dim, 54321);
    defer projection.destroy(allocator);

    var ws = try Workspace.init(allocator, dim);
    defer ws.deinit(allocator);

    const residual = [_]f32{ 1.0, -0.5, 0.25, -1.5, 2.0, -1.0, 0.75, -0.25 };
    const q = [_]f32{ 0.5, 1.25, -0.75, 0.5, -1.0, 0.25, 1.5, -0.5 };
    const gamma = math.norm(&residual);

    var encoded: [1]u8 = undefined;
    try encodeInto(&encoded, &residual, &projection, &ws);

    var decoded: [dim]f32 = undefined;
    decodeInto(&decoded, &encoded, gamma, &projection);

    const estimated = estimateDotWithWorkspace(&q, &encoded, gamma, &projection, &ws);
    const decoded_dot = math.dot(&q, &decoded);
    try std.testing.expectApproxEqAbs(decoded_dot, estimated, 1e-4);
}

test "decodeAddInto matches decodeInto then add" {
    const allocator = std.testing.allocator;
    const dim: usize = 8;

    var projection = try ProjectionOperator.prepare(allocator, dim, 31415);
    defer projection.destroy(allocator);

    var ws = try Workspace.init(allocator, dim);
    defer ws.deinit(allocator);

    const residual = [_]f32{ 0.5, -0.25, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0 };
    const gamma = math.norm(&residual);

    var encoded: [1]u8 = undefined;
    try encodeInto(&encoded, &residual, &projection, &ws);

    var added_via_decode = [_]f32{ 1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0 };
    var decoded: [dim]f32 = undefined;
    decodeInto(&decoded, &encoded, gamma, &projection);
    math.addInPlace(&added_via_decode, &decoded);

    var added_direct = [_]f32{ 1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0 };
    decodeAddInto(&added_direct, &encoded, gamma, &projection);

    for (added_via_decode, added_direct) |expected, actual| {
        try std.testing.expectApproxEqAbs(expected, actual, 1e-6);
    }
}
