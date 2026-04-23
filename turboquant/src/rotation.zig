const std = @import("std");
const math = @import("math.zig");

const LANE_COUNT = std.simd.suggestVectorLength(f32) orelse 4;
pub const RotationError = error{
    BufferTooSmall,
    InvalidDimension,
    OutOfMemory,
};

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

pub const RotationOperator = struct {
    dim: usize,
    seed: u32,
    dense_matrix: []f32,
    dense_matrix_t: []f32,

    pub fn prepare(allocator: std.mem.Allocator, dim: usize, seed: u32) RotationError!RotationOperator {
        const matrix = try allocator.alloc(f32, dim * dim);
        errdefer allocator.free(matrix);

        const matrix_t = try allocator.alloc(f32, dim * dim);
        errdefer allocator.free(matrix_t);

        return prepareInto(matrix, matrix_t, dim, seed);
    }

    pub fn prepareInto(
        storage_a: []f32,
        storage_b: []f32,
        dim: usize,
        seed: u32,
    ) RotationError!RotationOperator {
        if (storage_a.len < dim * dim or storage_b.len < dim * dim) return RotationError.BufferTooSmall;

        const matrix_view = storage_a[0 .. dim * dim];
        const matrix_t_view = storage_b[0 .. dim * dim];

        var rng = std.Random.DefaultPrng.init(seed);
        const random = rng.random();
        for (0..dim) |row| {
            for (0..dim) |col| {
                matrix_view[row * dim + col] = randGaussian(random);
            }
        }

        orthogonalizeInPlace(matrix_view, dim);

        for (0..dim) |row| {
            for (0..dim) |col| {
                matrix_t_view[col * dim + row] = matrix_view[row * dim + col];
            }
        }

        return .{
            .dim = dim,
            .seed = seed,
            .dense_matrix = matrix_view,
            .dense_matrix_t = matrix_t_view,
        };
    }

    pub fn requiredStorageSize(dim: usize) usize {
        return 2 * dim * dim * @sizeOf(f32);
    }

    pub fn destroy(op: *RotationOperator, allocator: std.mem.Allocator) void {
        allocator.free(op.dense_matrix);
        allocator.free(op.dense_matrix_t);
        op.* = undefined;
    }

    pub fn matVecMul(op: *const RotationOperator, input: []const f32, output: []f32) void {
        const dim = op.dim;
        std.debug.assert(input.len == dim and output.len == dim);
        matVecKernel(op.dense_matrix, input, output, dim);
    }

    pub fn matVecMulTransposed(op: *const RotationOperator, input: []const f32, output: []f32) void {
        const dim = op.dim;
        std.debug.assert(input.len == dim and output.len == dim);
        matVecKernel(op.dense_matrix_t, input, output, dim);
    }

    pub fn rotate(op: *const RotationOperator, input: []const f32, output: []f32) void {
        op.matVecMul(input, output);
    }
};

fn orthogonalizeInPlace(matrix: []f32, dim: usize) void {
    for (0..dim) |i| {
        for (0..i) |j| {
            var dot: f64 = 0;
            for (0..dim) |row| {
                dot += @as(f64, matrix[row * dim + i]) * @as(f64, matrix[row * dim + j]);
            }
            for (0..dim) |row| {
                matrix[row * dim + i] -= @as(f32, @floatCast(dot * @as(f64, matrix[row * dim + j])));
            }
        }

        var col_norm_sq: f64 = 0;
        for (0..dim) |row| {
            const value = @as(f64, matrix[row * dim + i]);
            col_norm_sq += value * value;
        }

        const col_norm = @sqrt(col_norm_sq);
        if (col_norm > 0) {
            const inv = @as(f32, @floatCast(1.0 / col_norm));
            for (0..dim) |row| {
                matrix[row * dim + i] *= inv;
            }
        }
    }
}

test "prepareInto matches heap prepare" {
    const allocator = std.testing.allocator;
    const dim: usize = 8;
    const matrix_count = dim * dim;

    var heap_op = try RotationOperator.prepare(allocator, dim, 42);
    defer heap_op.destroy(allocator);

    var storage_a: [matrix_count]f32 = undefined;
    var storage_b: [matrix_count]f32 = undefined;
    const inline_op = try RotationOperator.prepareInto(&storage_a, &storage_b, dim, 42);

    try std.testing.expectEqualSlices(f32, heap_op.dense_matrix, inline_op.dense_matrix);
    try std.testing.expectEqualSlices(f32, heap_op.dense_matrix_t, inline_op.dense_matrix_t);
}

test "rotation preserves norm" {
    const allocator = std.testing.allocator;
    const dim: usize = 64;

    var op = try RotationOperator.prepare(allocator, dim, 42);
    defer op.destroy(allocator);

    var rng = std.Random.DefaultPrng.init(12345);
    const random = rng.random();

    for (0..10) |_| {
        var input: [64]f32 = undefined;
        var output: [64]f32 = undefined;

        var input_norm_sq: f32 = 0;
        for (&input) |*value| {
            value.* = random.float(f32) * 2 - 1;
            input_norm_sq += value.* * value.*;
        }

        op.matVecMul(&input, &output);

        var output_norm_sq: f32 = 0;
        for (output) |value| {
            output_norm_sq += value * value;
        }

        try std.testing.expectApproxEqAbs(@sqrt(input_norm_sq), @sqrt(output_norm_sq), @sqrt(input_norm_sq) * 1e-3);
    }
}
