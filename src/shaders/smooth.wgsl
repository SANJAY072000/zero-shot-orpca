@group(0) @binding(0) var<storage, read> raw_mask: array<f32>;
@group(0) @binding(1) var<storage, read_write> smooth_mask: array<f32>;
@group(0) @binding(2) var<uniform> config: vec4<f32>; // x=width, y=height, z=total_pixels, w=unused

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_pixels = u32(config.z);
    
    if (idx >= total_pixels) { return; }

    let width = u32(config.x);
    let height = u32(config.y);

    let x = i32(idx % width);
    let y = i32(idx / width);

    var sum = 0.0;
    
    // 5x5 Spatial Neighborhood Voting
    for (var dy: i32 = -2; dy <= 2; dy++) {
        for (var dx: i32 = -2; dx <= 2; dx++) {
            // Clamp to edges so we don't read out of bounds
            let n_x = clamp(x + dx, 0, i32(width) - 1);
            let n_y = clamp(y + dy, 0, i32(height) - 1);
            
            let n_idx = u32(n_y) * width + u32(n_x);
            sum += raw_mask[n_idx];
        }
    }

    // 25 pixels total in the grid.
    // If fewer than 10 neighbors are foreground, it's sensor noise -> snap to background (0.0).
    // If more than 15 neighbors are foreground, it's a solid object -> snap to foreground (1.0).
    if (sum < 10.0) {
        smooth_mask[idx] = 0.0;
    } else if (sum > 15.0) {
        smooth_mask[idx] = 1.0;
    } else {
        smooth_mask[idx] = raw_mask[idx];
    }
}
