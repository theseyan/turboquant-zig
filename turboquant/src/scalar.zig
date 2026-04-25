const std = @import("std");

pub const ScalarError = error{
    BufferTooSmall,
    InvalidBitWidth,
    InvalidDimension,
    OutOfMemory,
};

const MAX_BITS: u8 = 8;
const MAX_LEVELS: usize = 1 << MAX_BITS;
const SQRT_TWO_PI: f64 = 2.5066282746310002;

pub const Codebook = struct {
    bits: u8,
    centroids: []f32,
    boundaries: []f32,

    pub fn init(allocator: std.mem.Allocator, bits: u8) ScalarError!Codebook {
        if (bits > MAX_BITS) return ScalarError.InvalidBitWidth;

        const levels = @as(usize, 1) << @intCast(bits);
        const centroids = try allocator.alloc(f32, levels);
        errdefer allocator.free(centroids);

        const boundaries = try allocator.alloc(f32, levels -| 1);
        errdefer allocator.free(boundaries);

        return initInto(centroids, boundaries, bits);
    }

    pub fn initInto(centroids: []f32, boundaries: []f32, bits: u8) ScalarError!Codebook {
        if (bits > MAX_BITS) return ScalarError.InvalidBitWidth;

        const levels = @as(usize, 1) << @intCast(bits);
        if (centroids.len < levels or boundaries.len < levels -| 1) return ScalarError.BufferTooSmall;

        const centroids_view = centroids[0..levels];
        const boundaries_view = boundaries[0..levels -| 1];

        if (bits == 0) {
            centroids_view[0] = 0;
            return .{
                .bits = bits,
                .centroids = centroids_view,
                .boundaries = boundaries_view,
            };
        }

        buildStandardNormalCodebook(centroids_view, boundaries_view);
        return .{
            .bits = bits,
            .centroids = centroids_view,
            .boundaries = boundaries_view,
        };
    }

    pub fn deinit(codebook: *Codebook, allocator: std.mem.Allocator) void {
        allocator.free(codebook.centroids);
        allocator.free(codebook.boundaries);
        codebook.* = .{
            .bits = 0,
            .centroids = &.{},
            .boundaries = &.{},
        };
    }

    pub fn requiredStorageSize(bits: u8) ScalarError!usize {
        if (bits > MAX_BITS) return ScalarError.InvalidBitWidth;
        const levels = @as(usize, 1) << @intCast(bits);
        return levels * @sizeOf(f32) + (levels -| 1) * @sizeOf(f32);
    }
};

pub fn encodedLen(dim: usize, bits: u8) ScalarError!usize {
    if (bits > MAX_BITS) return ScalarError.InvalidBitWidth;
    if (dim == 0) return ScalarError.InvalidDimension;
    return (dim * bits + 7) / 8;
}

pub fn encode(
    allocator: std.mem.Allocator,
    values: []const f32,
    codebook: *const Codebook,
    coord_scale: f32,
) ScalarError![]u8 {
    const out_len = try encodedLen(values.len, codebook.bits);
    const out = try allocator.alloc(u8, out_len);
    errdefer allocator.free(out);
    try encodeInto(out, values, codebook, coord_scale);
    return out;
}

pub fn encodeInto(
    out: []u8,
    values: []const f32,
    codebook: *const Codebook,
    coord_scale: f32,
) ScalarError!void {
    if (values.len == 0) return ScalarError.InvalidDimension;

    const out_len = try encodedLen(values.len, codebook.bits);
    if (out.len < out_len) return ScalarError.BufferTooSmall;
    @memset(out[0..out_len], 0);

    if (codebook.bits == 0) return;

    if (codebook.bits == 3) {
        for (values, 0..) |value, i| {
            const idx = quantizeIndex(codebook, value, coord_scale);
            write3Bits(out, i, idx);
        }
        return;
    }

    var bit_pos: usize = 0;
    for (values) |value| {
        const idx = quantizeIndex(codebook, value, coord_scale);
        writeBits(out, &bit_pos, idx, codebook.bits);
    }
}

pub fn encodeIntoDecoded(
    out: []u8,
    decoded: []f32,
    values: []const f32,
    codebook: *const Codebook,
    coord_scale: f32,
) ScalarError!void {
    if (values.len == 0) return ScalarError.InvalidDimension;
    if (decoded.len < values.len) return ScalarError.BufferTooSmall;

    const out_len = try encodedLen(values.len, codebook.bits);
    if (out.len < out_len) return ScalarError.BufferTooSmall;
    @memset(out[0..out_len], 0);

    if (codebook.bits == 0) {
        @memset(decoded[0..values.len], 0);
        return;
    }

    if (codebook.bits == 3) {
        for (values, decoded[0..values.len], 0..) |value, *dst, i| {
            const idx = quantizeIndex(codebook, value, coord_scale);
            write3Bits(out, i, idx);
            dst.* = codebook.centroids[idx] * coord_scale;
        }
        return;
    }

    var bit_pos: usize = 0;
    for (values, decoded[0..values.len]) |value, *dst| {
        const idx = quantizeIndex(codebook, value, coord_scale);
        writeBits(out, &bit_pos, idx, codebook.bits);
        dst.* = codebook.centroids[idx] * coord_scale;
    }
}

pub fn decodeInto(
    out: []f32,
    compressed: []const u8,
    codebook: *const Codebook,
    coord_scale: f32,
) ScalarError!void {
    if (out.len == 0) return ScalarError.InvalidDimension;

    const expected_len = try encodedLen(out.len, codebook.bits);
    if (compressed.len < expected_len) return ScalarError.BufferTooSmall;

    if (codebook.bits == 0) {
        @memset(out, 0);
        return;
    }

    if (codebook.bits == 3) {
        for (out, 0..) |*value, i| {
            const idx = read3Bits(compressed, i);
            value.* = codebook.centroids[idx] * coord_scale;
        }
        return;
    }

    var bit_pos: usize = 0;
    for (out) |*value| {
        const idx = readBits(compressed, &bit_pos, codebook.bits);
        value.* = codebook.centroids[idx] * coord_scale;
    }
}

pub fn dotProduct(
    q: []const f32,
    compressed: []const u8,
    codebook: *const Codebook,
    coord_scale: f32,
) ScalarError!f32 {
    if (q.len == 0) return ScalarError.InvalidDimension;

    const expected_len = try encodedLen(q.len, codebook.bits);
    if (compressed.len < expected_len) return ScalarError.InvalidDimension;

    if (codebook.bits == 0) return 0;

    if (codebook.bits == 3) {
        var sum: f32 = 0;
        for (q, 0..) |qv, i| {
            const idx = read3Bits(compressed, i);
            sum += qv * (codebook.centroids[idx] * coord_scale);
        }
        return sum;
    }

    var sum: f32 = 0;
    var bit_pos: usize = 0;
    for (q) |qv| {
        const idx = readBits(compressed, &bit_pos, codebook.bits);
        sum += qv * (codebook.centroids[idx] * coord_scale);
    }
    return sum;
}

pub fn quantizeIndex(codebook: *const Codebook, value: f32, coord_scale: f32) usize {
    if (codebook.bits == 0) return 0;

    const normalized = @as(f64, value) / @as(f64, coord_scale);
    var lo: usize = 0;
    var hi: usize = codebook.boundaries.len;

    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (normalized <= codebook.boundaries[mid]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    return lo;
}

fn writeBits(out: []u8, bit_pos: *usize, value: usize, bits: u8) void {
    var remaining: usize = bits;
    var current = value;

    while (remaining > 0) {
        const byte_idx = bit_pos.* / 8;
        const bit_offset = bit_pos.* % 8;
        const take = @min(remaining, 8 - bit_offset);
        const mask = (@as(usize, 1) << @intCast(take)) - 1;
        out[byte_idx] |= @as(u8, @intCast((current & mask) << @intCast(bit_offset)));
        current >>= @intCast(take);
        bit_pos.* += take;
        remaining -= take;
    }
}

fn read3Bits(data: []const u8, index: usize) usize {
    const bit_pos = index * 3;
    const byte_idx = bit_pos / 8;
    const bit_offset: u3 = @intCast(bit_pos % 8);
    var word: u16 = data[byte_idx];
    if (byte_idx + 1 < data.len) {
        word |= @as(u16, data[byte_idx + 1]) << 8;
    }
    return @intCast((word >> bit_offset) & 0x7);
}

fn write3Bits(out: []u8, index: usize, value: usize) void {
    const bit_pos = index * 3;
    const byte_idx = bit_pos / 8;
    const bit_offset: u4 = @intCast(bit_pos % 8);
    const shifted = @as(u16, @intCast(value & 0x7)) << bit_offset;
    out[byte_idx] |= @intCast(shifted & 0xFF);
    if (bit_offset > 5 and byte_idx + 1 < out.len) {
        out[byte_idx + 1] |= @intCast(shifted >> 8);
    }
}

fn readBits(data: []const u8, bit_pos: *usize, bits: u8) usize {
    var remaining: usize = bits;
    var shift: usize = 0;
    var result: usize = 0;

    while (remaining > 0) {
        const byte_idx = bit_pos.* / 8;
        const bit_offset = bit_pos.* % 8;
        const take = @min(remaining, 8 - bit_offset);
        const mask = (@as(usize, 1) << @intCast(take)) - 1;
        const chunk = (@as(usize, data[byte_idx]) >> @intCast(bit_offset)) & mask;
        result |= chunk << @intCast(shift);
        bit_pos.* += take;
        remaining -= take;
        shift += take;
    }

    return result;
}

fn buildStandardNormalCodebook(centroids: []f32, boundaries: []f32) void {
    const levels = centroids.len;

    var current_storage: [MAX_LEVELS]f64 = undefined;
    var next_storage: [MAX_LEVELS]f64 = undefined;
    const current = current_storage[0..levels];
    const next = next_storage[0..levels];

    if (levels == 1) {
        centroids[0] = 0;
        return;
    }

    for (0..levels) |i| {
        const ratio = @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(levels - 1));
        current[i] = -3.0 + 6.0 * ratio;
    }

    const neg_inf = -std.math.inf(f64);
    const pos_inf = std.math.inf(f64);

    for (0..128) |_| {
        for (0..levels) |i| {
            const lower = if (i == 0) neg_inf else 0.5 * (current[i - 1] + current[i]);
            const upper = if (i + 1 == levels) pos_inf else 0.5 * (current[i] + current[i + 1]);
            next[i] = truncatedNormalMean(lower, upper);
        }

        var max_delta: f64 = 0;
        for (0..levels) |i| {
            max_delta = @max(max_delta, @abs(next[i] - current[i]));
            current[i] = next[i];
        }
        if (max_delta < 1e-8) break;
    }

    for (0..levels) |i| {
        centroids[i] = @floatCast(current[i]);
    }
    for (0..boundaries.len) |i| {
        boundaries[i] = @floatCast(0.5 * (current[i] + current[i + 1]));
    }
}

fn truncatedNormalMean(lower: f64, upper: f64) f64 {
    const lower_pdf = if (std.math.isInf(lower)) 0 else normalPdf(lower);
    const upper_pdf = if (std.math.isInf(upper)) 0 else normalPdf(upper);
    const prob = normalCdf(upper) - normalCdf(lower);
    if (prob <= 1e-12) return 0;
    return (lower_pdf - upper_pdf) / prob;
}

fn normalPdf(x: f64) f64 {
    return @exp(-0.5 * x * x) / SQRT_TWO_PI;
}

fn normalCdf(x: f64) f64 {
    if (x == std.math.inf(f64)) return 1;
    if (x == -std.math.inf(f64)) return 0;

    const abs_x = @abs(x);
    const t = 1.0 / (1.0 + 0.2316419 * abs_x);
    const poly = (((((1.330274429 * t) - 1.821255978) * t) + 1.781477937) * t - 0.356563782) * t + 0.319381530;
    const approx = 1.0 - normalPdf(abs_x) * poly * t;
    return if (x >= 0) approx else 1.0 - approx;
}

test "1-bit codebook matches standard normal optimum" {
    const allocator = std.testing.allocator;

    var codebook = try Codebook.init(allocator, 1);
    defer codebook.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), codebook.centroids.len);
    try std.testing.expectApproxEqAbs(-@as(f32, @floatCast(@sqrt(2.0 / std.math.pi))), codebook.centroids[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(@sqrt(2.0 / std.math.pi))), codebook.centroids[1], 1e-3);
}

test "initInto builds codebook without allocation" {
    var centroids: [8]f32 = undefined;
    var boundaries: [7]f32 = undefined;
    const codebook = try Codebook.initInto(&centroids, &boundaries, 3);
    try std.testing.expectEqual(@as(u8, 3), codebook.bits);
    try std.testing.expectEqual(@as(usize, 8), codebook.centroids.len);
}

test "encodeInto decodeInto use nearest centroids" {
    const allocator = std.testing.allocator;

    var codebook = try Codebook.init(allocator, 2);
    defer codebook.deinit(allocator);

    const coord_scale: f32 = 0.5;
    const values = [_]f32{ -0.9, -0.2, 0.2, 0.9 };

    var encoded: [1]u8 = undefined;
    try encodeInto(&encoded, &values, &codebook, coord_scale);

    var decoded: [4]f32 = undefined;
    try decodeInto(&decoded, &encoded, &codebook, coord_scale);

    for (values, decoded) |original, reconstructed| {
        const idx = quantizeIndex(&codebook, original, coord_scale);
        try std.testing.expectApproxEqAbs(codebook.centroids[idx] * coord_scale, reconstructed, 1e-6);
    }
}

test "decodeInto rejects short buffers as buffer too small" {
    const allocator = std.testing.allocator;

    var codebook = try Codebook.init(allocator, 3);
    defer codebook.deinit(allocator);

    var decoded: [4]f32 = undefined;
    const short = [_]u8{0};
    try std.testing.expectError(ScalarError.BufferTooSmall, decodeInto(&decoded, &short, &codebook, 0.25));
}

test "encodeIntoDecoded matches encodeInto plus decodeInto" {
    const allocator = std.testing.allocator;

    var codebook = try Codebook.init(allocator, 3);
    defer codebook.deinit(allocator);

    const coord_scale: f32 = 0.25;
    const values = [_]f32{ -0.9, -0.2, 0.2, 0.9, -0.4, 0.6 };

    var encoded_a: [3]u8 = undefined;
    try encodeInto(&encoded_a, &values, &codebook, coord_scale);

    var encoded_b: [3]u8 = undefined;
    var decoded_b: [values.len]f32 = undefined;
    try encodeIntoDecoded(&encoded_b, &decoded_b, &values, &codebook, coord_scale);

    var decoded_a: [values.len]f32 = undefined;
    try decodeInto(&decoded_a, &encoded_a, &codebook, coord_scale);

    try std.testing.expectEqualSlices(u8, &encoded_a, &encoded_b);
    try std.testing.expectEqualSlices(f32, &decoded_a, &decoded_b);
}

test "dotProduct matches decoded vector dot" {
    const allocator = std.testing.allocator;

    var codebook = try Codebook.init(allocator, 3);
    defer codebook.deinit(allocator);

    const coord_scale: f32 = 0.25;
    const values = [_]f32{ -0.3, -0.1, 0.2, 0.8 };
    const q = [_]f32{ 1.0, 2.0, -1.0, 0.5 };

    const encoded = try encode(allocator, &values, &codebook, coord_scale);
    defer allocator.free(encoded);

    var decoded: [4]f32 = undefined;
    try decodeInto(&decoded, encoded, &codebook, coord_scale);

    var decoded_dot: f32 = 0;
    for (decoded, q) |dv, qv| decoded_dot += dv * qv;

    const packed_dot = try dotProduct(&q, encoded, &codebook, coord_scale);
    try std.testing.expectApproxEqAbs(decoded_dot, packed_dot, 1e-6);
}
