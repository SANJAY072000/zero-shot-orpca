@group(0) @binding(2) var<storage, read> m_t: array<f32>;
@group(0) @binding(3) var<storage, read_write> U_t: array<f32>;
@group(0) @binding(6) var<storage, read> c_scalar: array<f32>;
@group(0) @binding(7) var<storage, read_write> mask: array<f32>;
@group(0) @binding(5) var<uniform> config: vec4<f32>; 

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_pixels = u32(config.x);
    
    if (idx >= total_pixels) { return; }
    
    let c = c_scalar[0];
    let u_val = U_t[idx];
    let m_val = m_t[idx];
    
    let L_val = u_val * c;
    let error = m_val - L_val;
    
    let threshold = config.y;
    var is_foreground = 0.0;
    
    if (abs(error) > threshold) {
        is_foreground = 1.0;
    }
    
    mask[idx] = is_foreground;
    
    // Gradient descent on stationary pixels only
    if (is_foreground == 0.0) {
        let learning_rate = config.z;
        U_t[idx] = u_val + (learning_rate * error);
    }
}
