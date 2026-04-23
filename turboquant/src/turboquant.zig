const std = @import("std");
const log = std.log.scoped(.turboquant);

pub const format = @import("format.zig");
pub const math = @import("math.zig");
pub const qjl = @import("qjl.zig");
const rotation = @import("rotation.zig");
pub const scalar = @import("scalar.zig");

pub const EncodeError = error{
    BufferTooSmall,
    InvalidBufferAlignment,
    InvalidBitWidth,
    InvalidDimension,
    OutOfMemory,
};

pub const DecodeError = error{
    BufferTooSmall,
    InvalidHeader,
    InvalidPayload,
    OutOfMemory,
};

pub const EngineConfig = struct {
    dim: usize,
    seed: u32,
    bits_per_dim: u8 = 4,
};

pub const PreparedQuery = struct {
    dim: usize,
    rotated: []f32,
    qjl_projected: []f32,

    pub fn deinit(query: *PreparedQuery, allocator: std.mem.Allocator) void {
        allocator.free(query.rotated);
        allocator.free(query.qjl_projected);
        query.* = undefined;
    }
};

const BufferCursor = struct {
    buffer: []u8,
    offset: usize = 0,

    fn takeSlice(cursor: *BufferCursor, comptime T: type, count: usize) EncodeError![]T {
        const aligned_start = std.mem.alignForward(usize, cursor.offset, @alignOf(T));
        const bytes = count * @sizeOf(T);
        if (aligned_start + bytes > cursor.buffer.len) return EncodeError.BufferTooSmall;
        cursor.offset = aligned_start + bytes;

        const ptr: [*]T = @ptrCast(@alignCast(cursor.buffer[aligned_start..].ptr));
        return ptr[0..count];
    }
};

fn sizeWithAlign(current: usize, comptime T: type, count: usize) usize {
    return std.mem.alignForward(usize, current, @alignOf(T)) + count * @sizeOf(T);
}

pub const Engine = struct {
    dim: usize,
    seed: u32,
    bits_per_dim: u8,
    mse_bits: u8,
    coord_scale: f32,
    owned_storage: ?[]align(@alignOf(f32)) u8,
    storage: []u8,
    rotation_op: rotation.RotationOperator,
    qjl_projection: qjl.ProjectionOperator,
    qjl_workspace: qjl.Workspace,
    codebook: scalar.Codebook,
    scratch_unit: []f32,
    scratch_rotated: []f32,
    scratch_mse_rotated: []f32,
    scratch_mse_decoded: []f32,
    scratch_query_rotated: []f32,

    pub fn init(allocator: std.mem.Allocator, config: EngineConfig) !Engine {
        try validateConfig(config);
        const required = try requiredStorageSize(config);
        const storage = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(@alignOf(f32)), required);
        errdefer allocator.free(storage);
        return try initFromStorage(storage, config, storage);
    }

    pub fn initInBuffer(buffer: []u8, config: EngineConfig) !Engine {
        try validateConfig(config);
        if (@intFromPtr(buffer.ptr) % @alignOf(f32) != 0) return EncodeError.InvalidBufferAlignment;
        return try initFromStorage(buffer, config, null);
    }

    pub fn requiredStorageSize(config: EngineConfig) !usize {
        try validateConfig(config);

        const dim = config.dim;
        const mse_bits = config.bits_per_dim - 1;
        const levels = @as(usize, 1) << @intCast(mse_bits);

        var total: usize = 0;
        total = sizeWithAlign(total, f32, levels);
        total = sizeWithAlign(total, f32, levels -| 1);
        total = sizeWithAlign(total, f32, rotation.RotationOperator.requiredStorageSize(dim) / @sizeOf(f32));
        total = sizeWithAlign(total, f32, 2 * dim * dim);
        total = sizeWithAlign(total, f32, dim);
        total = sizeWithAlign(total, f32, dim);
        total = sizeWithAlign(total, f32, dim);
        total = sizeWithAlign(total, f32, dim);
        total = sizeWithAlign(total, f32, dim);
        total = sizeWithAlign(total, f32, dim);
        return total;
    }

    pub fn compressedLen(e: *const Engine) usize {
        return format.HEADER_SIZE + e.payloadLen();
    }

    pub fn payloadLen(e: *const Engine) usize {
        const scalar_len = scalar.encodedLen(e.dim, e.mse_bits) catch unreachable;
        const qjl_len = qjl.encodedLen(e.dim) catch unreachable;
        return scalar_len + qjl_len;
    }

    pub fn deinit(e: *Engine, allocator: std.mem.Allocator) void {
        if (e.owned_storage) |storage| allocator.free(storage);
        e.* = undefined;
    }

    pub fn encode(e: *Engine, allocator: std.mem.Allocator, x: []const f32) ![]u8 {
        const out = try allocator.alloc(u8, e.compressedLen());
        errdefer allocator.free(out);
        try e.encodeInto(out, x);
        return out;
    }

    pub fn encodeInto(e: *Engine, out: []u8, x: []const f32) !void {
        const dim = e.dim;
        if (x.len != dim) return EncodeError.InvalidDimension;
        if (out.len < e.compressedLen()) return EncodeError.BufferTooSmall;

        const vector_norm = math.norm(x);
        if (vector_norm == 0) {
            e.encodeZeroVectorInto(out[0..e.compressedLen()]);
            return;
        }

        const inv_norm = 1.0 / vector_norm;
        for (x, e.scratch_unit) |value, *dst| dst.* = value * inv_norm;

        e.rotation_op.matVecMul(e.scratch_unit, e.scratch_rotated);

        const header = out[0..format.HEADER_SIZE];
        const scalar_len = scalar.encodedLen(dim, e.mse_bits) catch unreachable;
        const payload = out[format.HEADER_SIZE .. format.HEADER_SIZE + scalar_len + (qjl.encodedLen(dim) catch unreachable)];
        const scalar_out = payload[0..scalar_len];
        const qjl_out = payload[scalar_len..];

        try scalar.encodeIntoDecoded(scalar_out, e.scratch_mse_rotated, e.scratch_rotated, &e.codebook, e.coord_scale);
        e.rotation_op.matVecMulTransposed(e.scratch_mse_rotated, e.scratch_mse_decoded);

        math.sub(e.scratch_unit, e.scratch_mse_decoded, e.scratch_unit);
        const gamma = math.norm(e.scratch_unit);
        try qjl.encodeInto(qjl_out, e.scratch_unit, &e.qjl_projection, &e.qjl_workspace);

        format.writeHeader(
            header,
            @intCast(dim),
            e.bits_per_dim,
            @intCast(scalar_len),
            @intCast(qjl_out.len),
            vector_norm,
            gamma,
        );

        const payload_bits = (scalar_out.len + qjl_out.len) * 8;
        const bpd = @as(f32, @floatFromInt(payload_bits)) / @as(f32, @floatFromInt(dim));
        log.debug("encoded: dim={}, bytes={}, bits/dim={d:.3}", .{ dim, e.compressedLen(), bpd });
    }

    pub fn decode(e: *Engine, allocator: std.mem.Allocator, compressed: []const u8) ![]f32 {
        const out = try allocator.alloc(f32, e.dim);
        errdefer allocator.free(out);
        try e.decodeInto(out, compressed);
        return out;
    }

    pub fn decodeInto(e: *Engine, out: []f32, compressed: []const u8) !void {
        if (out.len < e.dim) return DecodeError.BufferTooSmall;

        const header = format.readHeader(compressed) catch return DecodeError.InvalidHeader;
        try e.validateHeader(header);

        const payload = format.slicePayload(compressed, header) catch return DecodeError.InvalidPayload;
        const out_view = out[0..e.dim];

        if (header.vector_norm == 0) {
            @memset(out_view, 0);
            return;
        }

        try scalar.decodeInto(e.scratch_mse_rotated, payload.scalar, &e.codebook, e.coord_scale);
        e.rotation_op.matVecMulTransposed(e.scratch_mse_rotated, e.scratch_mse_decoded);
        const qjl_scale = qjl.sqrtPiOver2() * header.gamma / @as(f32, @floatFromInt(e.dim));
        for (0..e.dim) |i| {
            e.scratch_mse_rotated[i] = if (((payload.qjl[i / 8] >> @intCast(i % 8)) & 1) == 1) qjl_scale else -qjl_scale;
        }
        e.qjl_projection.matVecMulTransposed(e.scratch_mse_rotated, e.scratch_query_rotated);
        math.addInPlace(e.scratch_mse_decoded, e.scratch_query_rotated);
        copyScaled(out_view, e.scratch_mse_decoded, header.vector_norm);
    }

    pub fn dot(e: *Engine, q: []const f32, compressed: []const u8) f32 {
        const header = format.readHeader(compressed) catch return 0;
        e.validateHeader(header) catch return 0;
        if (q.len != e.dim or header.vector_norm == 0) return 0;

        const payload = format.slicePayload(compressed, header) catch return 0;

        e.rotation_op.matVecMul(q, e.scratch_query_rotated);
        const mse_dot = scalar.dotProduct(e.scratch_query_rotated, payload.scalar, &e.codebook, e.coord_scale) catch return 0;
        const qjl_dot = qjl.estimateDotWithWorkspace(
            q,
            payload.qjl,
            header.gamma,
            &e.qjl_projection,
            &e.qjl_workspace,
        );
        return header.vector_norm * (mse_dot + qjl_dot);
    }

    pub fn prepareQuery(e: *const Engine, allocator: std.mem.Allocator, q: []const f32) !PreparedQuery {
        if (q.len != e.dim) return EncodeError.InvalidDimension;

        const rotated = try allocator.alloc(f32, e.dim);
        errdefer allocator.free(rotated);

        const qjl_projected = try allocator.alloc(f32, e.dim);
        errdefer allocator.free(qjl_projected);

        e.prepareQueryInto(rotated, qjl_projected, q);
        return .{
            .dim = e.dim,
            .rotated = rotated,
            .qjl_projected = qjl_projected,
        };
    }

    pub fn prepareQueryInto(e: *const Engine, rotated: []f32, qjl_projected: []f32, q: []const f32) void {
        std.debug.assert(q.len == e.dim);
        std.debug.assert(rotated.len >= e.dim);
        std.debug.assert(qjl_projected.len >= e.dim);

        e.rotation_op.matVecMul(q, rotated[0..e.dim]);
        e.qjl_projection.matVecMul(q, qjl_projected[0..e.dim]);
    }

    pub fn dotPrepared(e: *const Engine, query: PreparedQuery, compressed: []const u8) f32 {
        if (query.dim != e.dim or query.rotated.len < e.dim or query.qjl_projected.len < e.dim) return 0;

        const header = format.readHeader(compressed) catch return 0;
        e.validateHeader(header) catch return 0;
        if (header.vector_norm == 0) return 0;

        const payload = format.slicePayload(compressed, header) catch return 0;
        const mse_dot = scalar.dotProduct(query.rotated[0..e.dim], payload.scalar, &e.codebook, e.coord_scale) catch return 0;
        const qjl_dot = qjl.estimateDotFromProjection(query.qjl_projected[0..e.dim], payload.qjl, header.gamma);
        return header.vector_norm * (mse_dot + qjl_dot);
    }

    fn encodeZeroVectorInto(e: *const Engine, out: []u8) void {
        std.debug.assert(out.len >= e.compressedLen());
        @memset(out[0..e.compressedLen()], 0);
        format.writeHeader(
            out,
            @intCast(e.dim),
            e.bits_per_dim,
            @intCast(scalar.encodedLen(e.dim, e.mse_bits) catch unreachable),
            @intCast(qjl.encodedLen(e.dim) catch unreachable),
            0,
            0,
        );
    }

    fn validateHeader(e: *const Engine, header: format.Header) DecodeError!void {
        if (header.dim != e.dim) return DecodeError.InvalidPayload;
        if (header.bits_per_dim != e.bits_per_dim) return DecodeError.InvalidPayload;

        const expected_scalar = scalar.encodedLen(e.dim, e.mse_bits) catch return DecodeError.InvalidPayload;
        const expected_qjl = qjl.encodedLen(e.dim) catch return DecodeError.InvalidPayload;
        if (header.scalar_bytes != expected_scalar or header.qjl_bytes != expected_qjl) {
            return DecodeError.InvalidPayload;
        }
    }
};

fn validateConfig(config: EngineConfig) EncodeError!void {
    if (config.dim == 0) return EncodeError.InvalidDimension;
    if (config.bits_per_dim == 0 or config.bits_per_dim > 8) return EncodeError.InvalidBitWidth;
}

fn initFromStorage(storage: []u8, config: EngineConfig, owned_storage: ?[]align(@alignOf(f32)) u8) !Engine {
    var cursor = BufferCursor{ .buffer = storage };
    const dim = config.dim;
    const mse_bits = config.bits_per_dim - 1;
    const levels = @as(usize, 1) << @intCast(mse_bits);

    const centroids = try cursor.takeSlice(f32, levels);
    const boundaries = try cursor.takeSlice(f32, levels -| 1);
    const rotation_storage_words = rotation.RotationOperator.requiredStorageSize(dim) / @sizeOf(f32);
    const rot_storage_a = try cursor.takeSlice(f32, rotation_storage_words);
    const qjl_matrix = try cursor.takeSlice(f32, dim * dim);
    const qjl_matrix_t = try cursor.takeSlice(f32, dim * dim);
    const qjl_projected = try cursor.takeSlice(f32, dim);
    const scratch_unit = try cursor.takeSlice(f32, dim);
    const scratch_rotated = try cursor.takeSlice(f32, dim);
    const scratch_mse_rotated = try cursor.takeSlice(f32, dim);
    const scratch_mse_decoded = try cursor.takeSlice(f32, dim);
    const scratch_query_rotated = try cursor.takeSlice(f32, dim);

    const codebook = try scalar.Codebook.initInto(centroids, boundaries, mse_bits);
    const rotation_op = try rotation.RotationOperator.prepareInto(
        rot_storage_a,
        rot_storage_a[dim * dim ..],
        dim,
        config.seed,
    );
    const qjl_projection = try qjl.ProjectionOperator.prepareInto(
        qjl_matrix,
        qjl_matrix_t,
        dim,
        config.seed ^ 0x9E3779B9,
    );
    const qjl_workspace = try qjl.Workspace.initInto(qjl_projected, dim);

    return .{
        .dim = dim,
        .seed = config.seed,
        .bits_per_dim = config.bits_per_dim,
        .mse_bits = mse_bits,
        .coord_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(dim))),
        .owned_storage = owned_storage,
        .storage = storage,
        .rotation_op = rotation_op,
        .qjl_projection = qjl_projection,
        .qjl_workspace = qjl_workspace,
        .codebook = codebook,
        .scratch_unit = scratch_unit,
        .scratch_rotated = scratch_rotated,
        .scratch_mse_rotated = scratch_mse_rotated,
        .scratch_mse_decoded = scratch_mse_decoded,
        .scratch_query_rotated = scratch_query_rotated,
    };
}

fn copyScaled(out: []f32, src: []const f32, scale_factor: f32) void {
    std.debug.assert(out.len == src.len);
    for (src, out) |value, *dst| dst.* = value * scale_factor;
}

test "initInBuffer produces same results as heap init" {
    const allocator = std.testing.allocator;
    const config = EngineConfig{ .dim = 16, .seed = 12345, .bits_per_dim = 4 };
    const compressed_len = format.HEADER_SIZE + ((config.dim * (config.bits_per_dim - 1) + 7) / 8) + ((config.dim + 7) / 8);

    var heap_engine = try Engine.init(allocator, config);
    defer heap_engine.deinit(allocator);

    const required_storage = try Engine.requiredStorageSize(config);
    const storage = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(@alignOf(f32)), required_storage);
    defer allocator.free(storage);
    var external_engine = try Engine.initInBuffer(storage, config);

    const x = [_]f32{ 1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0, 11.0, -12.0, 13.0, -14.0, 15.0, -16.0 };
    const qv = [_]f32{ 0.5, -0.25, 0.75, -1.0, 1.0, 0.25, -0.5, 0.125, 0.2, -0.4, 0.6, -0.8, 0.9, -1.1, 1.3, -1.5 };

    var heap_encoded: [compressed_len]u8 = undefined;
    try heap_engine.encodeInto(&heap_encoded, &x);

    var external_encoded: [compressed_len]u8 = undefined;
    try external_engine.encodeInto(&external_encoded, &x);

    try std.testing.expectEqualSlices(u8, &heap_encoded, &external_encoded);
    try std.testing.expectApproxEqAbs(heap_engine.dot(&qv, &heap_encoded), external_engine.dot(&qv, &external_encoded), 1e-5);
}

test "encodeInto and decodeInto avoid allocation and match wrapper APIs" {
    const allocator = std.testing.allocator;
    const config = EngineConfig{ .dim = 32, .seed = 7777, .bits_per_dim = 4 };
    const compressed_len = format.HEADER_SIZE + ((config.dim * (config.bits_per_dim - 1) + 7) / 8) + ((config.dim + 7) / 8);

    var engine = try Engine.init(allocator, config);
    defer engine.deinit(allocator);

    var rng = std.Random.DefaultPrng.init(7777);
    const random = rng.random();

    var x: [32]f32 = undefined;
    for (&x) |*value| value.* = random.float(f32) * 2 - 1;

    const wrapped = try engine.encode(allocator, &x);
    defer allocator.free(wrapped);

    var encoded_buf: [compressed_len]u8 = undefined;
    try engine.encodeInto(&encoded_buf, &x);
    try std.testing.expectEqualSlices(u8, wrapped, &encoded_buf);

    const wrapped_decoded = try engine.decode(allocator, &encoded_buf);
    defer allocator.free(wrapped_decoded);

    var decoded: [32]f32 = undefined;
    try engine.decodeInto(&decoded, &encoded_buf);
    try std.testing.expectEqualSlices(f32, wrapped_decoded, &decoded);
}

test "dot matches decoded-space dot" {
    const allocator = std.testing.allocator;
    const dim: usize = 64;
    const compressed_len = format.HEADER_SIZE + ((dim * 3 + 7) / 8) + ((dim + 7) / 8);

    var engine = try Engine.init(allocator, .{ .dim = dim, .seed = 7777, .bits_per_dim = 4 });
    defer engine.deinit(allocator);

    var rng = std.Random.DefaultPrng.init(7777);
    const random = rng.random();

    var x: [dim]f32 = undefined;
    var qv: [dim]f32 = undefined;
    for (&x) |*value| value.* = random.float(f32) * 2 - 1;
    for (&qv) |*value| value.* = random.float(f32) * 2 - 1;

    var compressed: [compressed_len]u8 = undefined;
    try engine.encodeInto(&compressed, &x);

    var decoded: [dim]f32 = undefined;
    try engine.decodeInto(&decoded, &compressed);

    var decoded_dot: f32 = 0;
    for (decoded, qv) |dv, q| decoded_dot += dv * q;

    const direct_dot = engine.dot(&qv, &compressed);
    try std.testing.expectApproxEqAbs(decoded_dot, direct_dot, @abs(decoded_dot) * 1e-4 + 1e-4);
}

test "prepared query dot matches regular dot" {
    const allocator = std.testing.allocator;
    const dim: usize = 64;
    const compressed_len = format.HEADER_SIZE + ((dim * 3 + 7) / 8) + ((dim + 7) / 8);

    var engine = try Engine.init(allocator, .{ .dim = dim, .seed = 2468, .bits_per_dim = 4 });
    defer engine.deinit(allocator);

    var rng = std.Random.DefaultPrng.init(2468);
    const random = rng.random();

    var x: [dim]f32 = undefined;
    var qv: [dim]f32 = undefined;
    for (&x) |*value| value.* = random.float(f32) * 2 - 1;
    for (&qv) |*value| value.* = random.float(f32) * 2 - 1;

    var compressed: [compressed_len]u8 = undefined;
    try engine.encodeInto(&compressed, &x);

    var prepared = try engine.prepareQuery(allocator, &qv);
    defer prepared.deinit(allocator);

    try std.testing.expectApproxEqAbs(engine.dot(&qv, &compressed), engine.dotPrepared(prepared, &compressed), 1e-5);
}

test "zero vector encodes and decodes safely" {
    const allocator = std.testing.allocator;
    const dim: usize = 16;
    const compressed_len = format.HEADER_SIZE + ((dim * 2 + 7) / 8) + ((dim + 7) / 8);

    var engine = try Engine.init(allocator, .{ .dim = dim, .seed = 9999, .bits_per_dim = 3 });
    defer engine.deinit(allocator);

    const x = [_]f32{0} ** dim;
    const qv = [_]f32{1} ** dim;

    var compressed: [compressed_len]u8 = undefined;
    try engine.encodeInto(&compressed, &x);

    var decoded: [dim]f32 = undefined;
    try engine.decodeInto(&decoded, &compressed);

    for (decoded) |value| try std.testing.expectEqual(0.0, value);
    try std.testing.expectEqual(0.0, engine.dot(&qv, &compressed));
}

test "init rejects invalid configuration" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(EncodeError.InvalidDimension, Engine.init(allocator, .{ .dim = 0, .seed = 1 }));
    try std.testing.expectError(EncodeError.InvalidBitWidth, Engine.init(allocator, .{ .dim = 8, .seed = 1, .bits_per_dim = 0 }));
}

test "initInBuffer rejects short buffer" {
    var storage: [8]u8 align(@alignOf(f32)) = undefined;
    try std.testing.expectError(EncodeError.BufferTooSmall, Engine.initInBuffer(&storage, .{ .dim = 8, .seed = 1, .bits_per_dim = 4 }));
}

test "initInBuffer rejects misaligned buffer" {
    const required_storage = try Engine.requiredStorageSize(.{ .dim = 8, .seed = 1, .bits_per_dim = 4 });
    var storage: [4096]u8 align(@alignOf(f32)) = undefined;
    try std.testing.expect(required_storage < storage.len);

    const misaligned = storage[1 .. required_storage + 1];
    try std.testing.expectError(EncodeError.InvalidBufferAlignment, Engine.initInBuffer(misaligned, .{ .dim = 8, .seed = 1, .bits_per_dim = 4 }));
}

test "decode rejects truncated header and payload" {
    const allocator = std.testing.allocator;
    var engine = try Engine.init(allocator, .{ .dim = 8, .seed = 12345, .bits_per_dim = 4 });
    defer engine.deinit(allocator);

    const short: [5]u8 = .{ format.PAYLOAD_VERSION, 0, 0, 0, 0 };
    var out: [8]f32 = undefined;
    try std.testing.expectError(DecodeError.InvalidHeader, engine.decodeInto(&out, &short));

    var buf: [format.HEADER_SIZE + 4]u8 = undefined;
    @memset(&buf, 0);
    format.writeHeader(&buf, 8, 4, 8, 1, 1.0, 0.5);
    try std.testing.expectError(DecodeError.InvalidPayload, engine.decodeInto(&out, &buf));
}
