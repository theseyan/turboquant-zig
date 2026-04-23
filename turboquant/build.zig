const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const turboquant_mod = b.addModule("turboquant", .{
        .root_source_file = b.path("src/turboquant.zig"),
        .target = target,
        .optimize = optimize,
    });
    turboquant_mod.addImport("matrix", b.addModule("matrix", .{
        .root_source_file = b.path("src/matrix.zig"),
        .target = target,
        .optimize = optimize,
    }));
    turboquant_mod.addImport("qjl", b.addModule("qjl", .{
        .root_source_file = b.path("src/qjl.zig"),
        .target = target,
        .optimize = optimize,
    }));
    turboquant_mod.addImport("scalar", b.addModule("scalar", .{
        .root_source_file = b.path("src/scalar.zig"),
        .target = target,
        .optimize = optimize,
    }));
    turboquant_mod.addImport("format", b.addModule("format", .{
        .root_source_file = b.path("src/format.zig"),
        .target = target,
        .optimize = optimize,
    }));
    turboquant_mod.addImport("rotation", b.addModule("rotation", .{
        .root_source_file = b.path("src/rotation.zig"),
        .target = target,
        .optimize = optimize,
    }));
    turboquant_mod.addImport("math", b.addModule("math", .{
        .root_source_file = b.path("src/math.zig"),
        .target = target,
        .optimize = optimize,
    }));

    const bench_mod = b.addModule("bench", .{
        .root_source_file = b.path("benchmarks/quality.zig"),
        .target = target,
        .optimize = optimize,
    });
    bench_mod.addImport("turboquant", turboquant_mod);

    const tests = b.addTest(.{
        .root_module = turboquant_mod,
    });
    const quality_tests = b.addTest(.{
        .root_module = bench_mod,
    });

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&b.addRunArtifact(tests).step);
    test_step.dependOn(&b.addRunArtifact(quality_tests).step);

    const bench_exe = b.addExecutable(.{
        .name = "bench",
        .root_module = bench_mod,
    });

    const bench_run = b.addRunArtifact(bench_exe);
    if (b.args) |args| bench_run.addArgs(args);

    const bench_step = b.step("bench", "Run benchmarks");
    bench_step.dependOn(&bench_run.step);

    const quality_step = b.step("quality", "Run quality benchmark");
    quality_step.dependOn(&bench_run.step);

    const engine_bench_mod = b.addModule("bench_engine", .{
        .root_source_file = b.path("src/bench.zig"),
        .target = target,
        .optimize = optimize,
    });

    const engine_bench_exe = b.addExecutable(.{
        .name = "bench-engine",
        .root_module = engine_bench_mod,
    });

    const engine_bench_run = b.addRunArtifact(engine_bench_exe);
    if (b.args) |args| engine_bench_run.addArgs(args);

    const engine_bench_step = b.step("bench-engine", "Run low-level engine benchmarks");
    engine_bench_step.dependOn(&engine_bench_run.step);
}
