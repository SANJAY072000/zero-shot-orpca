export class RenderPipeline {
  private device: GPUDevice;
  private pipeline!: GPURenderPipeline;
  private sampler!: GPUSampler;
  private format: GPUTextureFormat;

  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this.device = device;
    this.format = format;
  }

  public initialize(): void {
    this.sampler = this.device.createSampler({ minFilter: 'linear', magFilter: 'linear' });

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

        @fragment
        fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
            let color = textureSampleBaseClampToEdge(videoTexture, videoSampler, uv);
            
            let dim = textureDimensions(videoTexture);
            let x = u32(uv.x * f32(dim.x));
            let y = u32(uv.y * f32(dim.y));
            let idx = y * dim.x + x;
            
            let is_foreground = mask[idx];
            
            // Render mask debug view
            if (is_foreground > 0.5) {
                return color;
            } else {
                return vec4f(color.r * 0.2, color.g * 0.2, color.b * 0.8, 1.0); 
            }
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

  public render(context: GPUCanvasContext, externalTexture: GPUExternalTexture, maskBuffer: GPUBuffer, commandEncoder: GPUCommandEncoder): void {
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: externalTexture },
        { binding: 2, resource: { buffer: maskBuffer } }
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
