@group(0) @binding(0) var videoSampler: sampler;
@group(0) @binding(1) var videoTexture: texture_external;
@group(0) @binding(2) var<storage, read_write> m_t: array<f32>;
@group(0) @binding(3) var<storage, read> U_t: array<f32>;
@group(0) @binding(4) var<storage, read_write> partial_sums: array<f32>;
@group(0) @binding(5) var<uniform> config: vec4<f32>; 

var<workgroup> shared_sums: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let idx = global_id.x;
    let total_pixels = u32(config.z);
    
    var val = 0.0;
    
    if (idx < total_pixels) {
        let width = u32(config.x);
        let height = u32(config.y);
        let uv = vec2<f32>(f32(idx % width) / config.x, f32(idx / width) / config.y);
        
        let color = textureSampleBaseClampToEdge(videoTexture, videoSampler, uv);
        let luminance = dot(color.rgb, vec3<f32>(0.299, 0.587, 0.114));
        
        m_t[idx] = luminance;
        val = luminance * U_t[idx]; 
    }
    
    // Parallel reduction block
    shared_sums[local_id.x] = val;
    workgroupBarrier();
    
    for (var offset = 128u; offset > 0u; offset = offset / 2u) {
        if (local_id.x < offset) {
            shared_sums[local_id.x] = shared_sums[local_id.x] + shared_sums[local_id.x + offset];
        }
        workgroupBarrier();
    }
    
    if (local_id.x == 0u) {
        partial_sums[group_id.x] = shared_sums[0];
    }
}
