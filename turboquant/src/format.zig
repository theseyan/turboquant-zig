const std = @import("std");

pub const FormatError = error{
    InvalidHeader,
    InvalidPayload,
    OutOfMemory,
};

pub const Header = packed struct {
    version: u8,
    dim: u32,
    bits_per_dim: u8,
    scalar_bytes: u32,
    qjl_bytes: u32,
    vector_norm: f32,
    gamma: f32,
};

pub const PAYLOAD_VERSION: u8 = 2;
pub const HEADER_SIZE: usize = @sizeOf(Header);

pub fn writeHeader(
    out: []u8,
    dim: u32,
    bits_per_dim: u8,
    scalar_bytes: u32,
    qjl_bytes: u32,
    vector_norm: f32,
    gamma: f32,
) void {
    std.debug.assert(out.len >= HEADER_SIZE);

    out[0] = PAYLOAD_VERSION;
    std.mem.writeInt(u32, out[1..5], dim, .little);
    out[5] = bits_per_dim;
    std.mem.writeInt(u32, out[6..10], scalar_bytes, .little);
    std.mem.writeInt(u32, out[10..14], qjl_bytes, .little);
    std.mem.writeInt(u32, out[14..18], @bitCast(vector_norm), .little);
    std.mem.writeInt(u32, out[18..22], @bitCast(gamma), .little);
}

pub fn readHeader(data: []const u8) FormatError!Header {
    if (data.len < HEADER_SIZE) return FormatError.InvalidHeader;
    if (data[0] != PAYLOAD_VERSION) return FormatError.InvalidHeader;
    if (data[5] > 8) return FormatError.InvalidHeader;

    const dim = std.mem.readInt(u32, data[1..5], .little);
    const scalar_bytes = std.mem.readInt(u32, data[6..10], .little);
    const qjl_bytes = std.mem.readInt(u32, data[10..14], .little);
    const vector_norm: f32 = @bitCast(std.mem.readInt(u32, data[14..18], .little));
    const gamma: f32 = @bitCast(std.mem.readInt(u32, data[18..22], .little));

    if (!std.math.isFinite(vector_norm) or !std.math.isFinite(gamma) or vector_norm < 0 or gamma < 0) {
        return FormatError.InvalidHeader;
    }

    return .{
        .version = data[0],
        .dim = dim,
        .bits_per_dim = data[5],
        .scalar_bytes = scalar_bytes,
        .qjl_bytes = qjl_bytes,
        .vector_norm = vector_norm,
        .gamma = gamma,
    };
}

pub fn slicePayload(data: []const u8, header: Header) FormatError!struct { scalar: []const u8, qjl: []const u8 } {
    const payload_start = HEADER_SIZE;
    const payload_end = payload_start + header.scalar_bytes + header.qjl_bytes;
    if (payload_end < payload_start or data.len < payload_end) return FormatError.InvalidPayload;

    return .{
        .scalar = data[payload_start .. payload_start + header.scalar_bytes],
        .qjl = data[payload_start + header.scalar_bytes .. payload_end],
    };
}

test "header roundtrip" {
    const dim: u32 = 128;
    const bits_per_dim: u8 = 4;
    const scalar_bytes: u32 = 64;
    const qjl_bytes: u32 = 16;
    const vector_norm: f32 = 2.5;
    const gamma: f32 = 0.125;

    var buf: [HEADER_SIZE]u8 = undefined;
    writeHeader(&buf, dim, bits_per_dim, scalar_bytes, qjl_bytes, vector_norm, gamma);

    const header = try readHeader(&buf);
    try std.testing.expectEqual(dim, header.dim);
    try std.testing.expectEqual(bits_per_dim, header.bits_per_dim);
    try std.testing.expectEqual(scalar_bytes, header.scalar_bytes);
    try std.testing.expectEqual(qjl_bytes, header.qjl_bytes);
    try std.testing.expectEqual(vector_norm, header.vector_norm);
    try std.testing.expectEqual(gamma, header.gamma);
    try std.testing.expectEqual(PAYLOAD_VERSION, header.version);
}

test "reject short header" {
    const bad: [5]u8 = .{ PAYLOAD_VERSION, 0, 0, 0, 0 };
    const result = readHeader(&bad);
    try std.testing.expectError(FormatError.InvalidHeader, result);
}

test "reject wrong version" {
    var buf: [HEADER_SIZE]u8 = undefined;
    @memset(&buf, 0);
    buf[0] = 99;

    const result = readHeader(&buf);
    try std.testing.expectError(FormatError.InvalidHeader, result);
}

test "reject invalid bit width" {
    var buf: [HEADER_SIZE]u8 = undefined;
    @memset(&buf, 0);
    buf[0] = PAYLOAD_VERSION;
    buf[5] = 9;

    const result = readHeader(&buf);
    try std.testing.expectError(FormatError.InvalidHeader, result);
}

test "payload slicing" {
    const dim: u32 = 8;
    const bits_per_dim: u8 = 3;
    const scalar_bytes: u32 = 3;
    const qjl_bytes: u32 = 1;

    var buf: [HEADER_SIZE + scalar_bytes + qjl_bytes]u8 = undefined;
    writeHeader(&buf, dim, bits_per_dim, scalar_bytes, qjl_bytes, 1.0, 0.5);
    buf[HEADER_SIZE] = 0xAA;
    buf[HEADER_SIZE + 1] = 0xBB;
    buf[HEADER_SIZE + 2] = 0xCC;
    buf[HEADER_SIZE + 3] = 0xDD;

    const header = try readHeader(&buf);
    const payload = try slicePayload(&buf, header);

    try std.testing.expectEqual(@as(usize, 3), payload.scalar.len);
    try std.testing.expectEqual(@as(usize, 1), payload.qjl.len);
    try std.testing.expectEqual(@as(u8, 0xAA), payload.scalar[0]);
    try std.testing.expectEqual(@as(u8, 0xDD), payload.qjl[0]);
}

test "reject truncated payload" {
    var buf: [HEADER_SIZE + 8]u8 = undefined;
    writeHeader(&buf, 8, 3, 100, 1, 1.0, 0.5);

    const header = try readHeader(&buf);
    const result = slicePayload(&buf, header);
    try std.testing.expectError(FormatError.InvalidPayload, result);
}
