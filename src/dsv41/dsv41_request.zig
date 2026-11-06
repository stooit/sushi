const std = @import("std");
const model = @import("dsv41_model.zig");
const prefix = @import("dsv41_prefix.zig");
const borrow = @import("dsv41_borrow.zig");
const scheduler = @import("../scheduler.zig");

pub const Request = model.Request;

pub const Binding = struct {
    model: ?*model.Model = null,
    previous: ?*model.Request = null,

    pub fn end(self: Binding) !void {
        const m = self.model orelse return;
        defer m.active_request = self.previous;
        try model.settleRequest(m);
    }
};

/// Only the inference thread binds requests; nested cache commits use the existing binding.
pub fn begin(slot: *scheduler.Slot, prefill: bool) !Binding {
    const xfm = slot.model.transformer orelse return .{};
    const m = xfm.dsv41 orelse return .{};
    if (prefill and slot.dsv41_request == null) {
        std.debug.assert(m.active_request == null);
        try model.settleRequest(m);
        slot.dsv41_request = m.idle_request;
        m.idle_request = .{};
    }
    const request = if (slot.dsv41_request) |*r| r else return .{};
    if (m.active_request == request) return .{};
    const previous = m.active_request;
    if (previous != null) try model.settleRequest(m);
    if (prefill) borrow.endRequestMlx(&xfm.config);
    m.active_request = request;
    return .{ .model = m, .previous = previous };
}

pub fn deinit(slot: *scheduler.Slot) void {
    if (slot.dsv41_request) |*request| request.deinit(slot.model.transformer.?.dsv41.?.gpa);
}

/// Load-time policy is safe to read from HTTP metadata handlers.
pub fn exclusiveLoad(xfm: *const @import("../transformer.zig").Transformer) bool {
    const engine = xfm.expert_stream orelse return true;
    return !engine.plan.resident;
}

/// Admission runs on the inference thread, where borrowed pools can change.
pub fn exclusiveAdmission(xfm: *const @import("../transformer.zig").Transformer) bool {
    if (exclusiveLoad(xfm)) return true;
    for (xfm.expert_stream.?.layers) |layer| {
        if (layer.borrowed or (layer.active and layer.resident_map.ctx == null)) return true;
    }
    return false;
}

/// Keep one completed continuation; persist the previous idle one's points before replacing it.
pub fn retire(slot: *scheduler.Slot) void {
    const request = if (slot.dsv41_request) |*r| r else return;
    const xfm = slot.model.transformer orelse return;
    const m = xfm.dsv41 orelse return;
    if (m.active_request == request) return;
    std.debug.assert(m.active_request == null);
    if (slot.error_code != null or request.state == null) {
        request.deinit(m.gpa);
        return;
    }
    if (slot.model.prefix_cache) |*hc| prefix.releaseLive(hc, m, xfm.s, "completed request");
    m.idle_request.deinit(m.gpa);
    m.idle_request = request.*;
    request.* = .{};
}

test "dsv41 request binding isolates history, cache epoch and retirement" {
    const a = std.testing.allocator;
    var m: model.Model = undefined;
    m.gpa = a;
    m.pipe = .{};
    m.active_request = null;
    m.idle_request = .{ .state = .{ .gpa = a, .layers = &.{}, .n = 3 }, .epoch = 7 };
    defer m.idle_request.deinit(a);
    try m.idle_request.state.?.hist.appendSlice(a, &.{ 11, 13, 17 });
    var xfm: @import("../transformer.zig").Transformer = undefined;
    xfm.dsv41 = &m;
    xfm.config = .{};
    var loaded: @import("../model_registry.zig").LoadedModel = undefined;
    loaded.transformer = &xfm;
    loaded.prefix_cache = null;
    var first: scheduler.Slot = undefined;
    first.model = &loaded;
    first.dsv41_request = null;
    first.error_code = null;
    var second = first;
    defer deinit(&first);
    defer deinit(&second);

    const one = try begin(&first, true);
    try std.testing.expectEqualSlices(u32, &.{ 11, 13, 17 }, m.request().state.?.hist.items);
    m.request().live.ready = true;
    const nested = try begin(&first, false);
    try nested.end();
    try std.testing.expect(m.active_request != null);
    try one.end();
    try std.testing.expect(m.active_request == null);

    const two = try begin(&second, true);
    try std.testing.expect(m.request().state == null);
    try std.testing.expect(!m.request().live.ready);
    m.request().state = .{ .gpa = a, .layers = &.{}, .n = 1 };
    try m.request().state.?.hist.append(a, 23);
    model.resetState(&m);
    try std.testing.expectEqual(@as(u64, 1), m.request().epoch);
    try two.end();
    retire(&second);

    const resumed = try begin(&first, false);
    const interleaved = try begin(&second, false);
    try std.testing.expect(m.active_request == &second.dsv41_request.?);
    try interleaved.end();
    try std.testing.expect(m.active_request == &first.dsv41_request.?);
    try std.testing.expectEqualSlices(u32, &.{ 11, 13, 17 }, m.request().state.?.hist.items);
    try std.testing.expectEqual(@as(u64, 7), m.request().epoch);
    try std.testing.expect(m.request().live.ready);
    try resumed.end();
    retire(&first);
    try std.testing.expect(first.dsv41_request.?.state == null);
    try std.testing.expectEqualSlices(u32, &.{ 11, 13, 17 }, m.idle_request.state.?.hist.items);
}

test "dsv41 interleaved DSpark requests retain their isolated logits (DSV41_TINY)" {
    const mlx = @import("../mlx.zig");
    try mlx.requireGpu();
    const dir = try model.fixtureDir();
    const a = std.testing.allocator;
    const m = try model.tinyModel(dir, .{ .qat = false, .dspark = true }, mlx.gpuStream());
    defer m.deinit();
    const Round = struct {
        fn run(mdl: *model.Model, request: *model.Request, offset: u32) ![]f32 {
            mdl.active_request = request;
            defer mdl.active_request = null;
            if (request.state == null) {
                var prompt: [24]u32 = undefined;
                for (&prompt, 0..) |*t, i| t.* = @intCast((i + offset) % mdl.cfg.vocab);
                _ = mlx.mlx_array_free(try model.extend(mdl, &prompt, .last));
            }
            var draft = try model.draftHeads(mdl, offset, .conf);
            defer draft.deinit();
            const logits = try model.verify(mdl, draft.ids);
            defer _ = mlx.mlx_array_free(logits);
            var scope = @import("dsv41_scope.zig").Scope.init(std.testing.allocator, mdl.s);
            defer scope.deinit();
            const values = try model.hostRows(&scope, logits);
            errdefer std.testing.allocator.free(values);
            try model.finishVerify(mdl, 1);
            try model.settleRequest(mdl);
            return values;
        }
    };
    var expected: [2][3][]f32 = undefined;
    for (0..2) |i| {
        var request: model.Request = .{};
        defer request.deinit(a);
        for (0..3) |round| expected[i][round] = try Round.run(m, &request, @intCast(3 + i * 11));
    }
    defer for (expected) |row| for (row) |values| a.free(values);
    var requests = [_]model.Request{ .{}, .{} };
    defer for (&requests) |*request| request.deinit(a);
    for (0..3) |round| for (&requests, 0..) |*request, i| {
        const got = try Round.run(m, request, @intCast(3 + i * 11));
        defer a.free(got);
        try std.testing.expectEqualSlices(f32, expected[i][round], got);
        try std.testing.expectEqual(@as(usize, 24 + 2 * (round + 1)), request.state.?.n);
    };
}
