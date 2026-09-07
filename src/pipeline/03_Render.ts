// src/pipeline/03_Render.ts

export class RenderPipeline {
  private device: GPUDevice;
  private pipeline!: GPURenderPipeline;
  private sampler!: GPUSampler;
  private format: GPUTextureFormat;
  private renderConfigBuffer!: GPUBuffer;

  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this.device = device;
    this.format = format;
  }

  public initialize(): void {
    this.sampler = this.device.createSampler({ minFilter: 'linear', magFilter: 'linear' });
    this.renderConfigBuffer = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const shaderModule = this.device.createShaderModule({
      code: `
        struct VertexOutput {
          @builtin(position) position: vec4f,
          @location(0) uv: vec2f,
        };

        @vertex
        fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
          var pos = array<vec2f, 6>(
            vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f(-1.0,  1.0),
            vec2f(-1.0,  1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0)
          );
          var uvs = array<vec2f, 6>(
            vec2f(0.0, 1.0), vec2f(1.0, 1.0), vec2f(0.0, 0.0),
            vec2f(0.0, 0.0), vec2f(1.0, 1.0), vec2f(1.0, 0.0)
          );

          var output: VertexOutput;
          output.position = vec4f(pos[vertexIndex], 0.0, 1.0);
          output.uv = uvs[vertexIndex];
          return output;
        }

        @group(0) @binding(0) var videoSampler: sampler;
        @group(0) @binding(1) var videoTexture: texture_external;
        @group(0) @binding(2) var<storage, read> mask: array<f32>;
        @group(0) @binding(3) var<uniform> renderConfig: vec4<f32>;

        @fragment
        fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
            let color = textureSampleBaseClampToEdge(videoTexture, videoSampler, uv);
            
            // 1. UI Toggle Check
            if (renderConfig.x < 0.5) {
                return color;
            }
            
            let dim = textureDimensions(videoTexture);
            let x = u32(uv.x * f32(dim.x));
            let y = u32(uv.y * f32(dim.y));
            let idx = y * dim.x + x;
            
            // 2. Skip blurring if this specific pixel is the user
            if (mask[idx] > 0.5) {
                return color;
            } 
            
            // 3. BRANCHLESS STRIDED GAUSSIAN BLUR
            var blur_color = vec4f(0.0);
            var total_weight = 0.0;
            
            // Radius 5 = 11x11 grid (121 samples). Perfectly optimized for 60fps.
            let RADIUS: i32 = 5;      
            // Stride 3.5 pushes the samples outward to create a massive privacy blur
            let STRIDE: f32 = 3.5;    
            let SIGMA: f32 = 8.0;
            
            for (var r = -RADIUS; r <= RADIUS; r++) {
                for (var c = -RADIUS; c <= RADIUS; c++) {
                    let offset = vec2f(f32(c) * STRIDE, f32(r) * STRIDE) / vec2f(f32(dim.x), f32(dim.y));
                    let sample_uv = clamp(uv + offset, vec2f(0.0), vec2f(1.0));
                    
                    let s_x = u32(sample_uv.x * f32(dim.x));
                    let s_y = u32(sample_uv.y * f32(dim.y));
                    let sample_idx = s_y * dim.x + s_x;
                    
                    let sample_col = textureSampleBaseClampToEdge(videoTexture, videoSampler, sample_uv);
                    
                    // The Branchless Fix: mathematically zeroes out the weight if it hits your t-shirt
                    let is_background = 1.0 - mask[sample_idx];
                    
                    let dist_sq = f32(r*r + c*c) * STRIDE * STRIDE;
                    let weight = exp(-dist_sq / (2.0 * SIGMA * SIGMA)) * is_background;
                    
                    blur_color += sample_col * weight;
                    total_weight += weight;
                }
            }
            
            if (total_weight > 0.0) {
                return blur_color / total_weight;
            }
            return color;
        }
      `
    });

    this.pipeline = this.device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: shaderModule, entryPoint: 'vs_main' },
      fragment: { module: shaderModule, entryPoint: 'fs_main', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
    });
  }

  public render(context: GPUCanvasContext, externalTexture: GPUExternalTexture, maskBuffer: GPUBuffer, commandEncoder: GPUCommandEncoder, isBlurEnabled: boolean): void {
    this.device.queue.writeBuffer(this.renderConfigBuffer, 0, new Float32Array([isBlurEnabled ? 1.0 : 0.0, 0, 0, 0]));

    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: externalTexture },
        { binding: 2, resource: { buffer: maskBuffer } },
        { binding: 3, resource: { buffer: this.renderConfigBuffer } }
      ],
    });

    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear', storeOp: 'store',
      }],
    });

    renderPass.setPipeline(this.pipeline);
    renderPass.setBindGroup(0, bindGroup);
    renderPass.draw(6);
    renderPass.end();
  }
}
