@group(0) @binding(4) var<storage, read_write> partial_sums: array<f32>;
@group(0) @binding(6) var<storage, read_write> c_scalar: array<f32>;
@group(0) @binding(5) var<uniform> config: vec4<f32>; 

var<workgroup> shared_sums: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;
    let num_workgroups = u32(config.w);
    
    var sum = 0.0;
    for (var i = tid; i < num_workgroups; i += 256u) {
        sum += partial_sums[i];
    }
    
    shared_sums[tid] = sum;
    workgroupBarrier();
    
    for (var offset = 128u; offset > 0u; offset = offset / 2u) {
        if (tid < offset) {
            shared_sums[tid] = shared_sums[tid] + shared_sums[tid + offset];
        }
        workgroupBarrier();
    }
    
    if (tid == 0u) {
        c_scalar[0] = shared_sums[0];
    }
}
