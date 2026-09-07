import projectionWgsl from '../shaders/projection.wgsl?raw';
import residualWgsl from '../shaders/residual.wgsl?raw';
import updateWgsl from '../shaders/update.wgsl?raw';
import smoothWgsl from '../shaders/smooth.wgsl?raw';

export class ComputePipeline {
  private device: GPUDevice;
  
  private bufferMt!: GPUBuffer;
  private bufferUt!: GPUBuffer;
  private bufferPartialSums!: GPUBuffer;
  private bufferCScalar!: GPUBuffer;
  public bufferMask!: GPUBuffer;
  public bufferSmoothMask!: GPUBuffer;
  
  private bufferConfigProj!: GPUBuffer;
  private bufferConfigRes!: GPUBuffer;
  private bufferConfigUpdate!: GPUBuffer;

  private projPipeline!: GPUComputePipeline;
  private resPipeline!: GPUComputePipeline;
  private updatePipeline!: GPUComputePipeline;
  private smoothPipeline!: GPUComputePipeline;

  private resBindGroup!: GPUBindGroup;
  private updateBindGroup!: GPUBindGroup;
  private smoothBindGroup!: GPUBindGroup;
  private sampler!: GPUSampler;

  private numPixels: number;
  private workgroupCount: number;

  constructor(device: GPUDevice, width: number, height: number) {
    this.device = device;
    this.numPixels = width * height;
    this.workgroupCount = Math.ceil(this.numPixels / 256);
  }

  public initialize(): void {
    const f32Bytes = 4;

    this.bufferMt = this.device.createBuffer({ size: this.numPixels * f32Bytes, usage: GPUBufferUsage.STORAGE });
    this.bufferUt = this.device.createBuffer({ size: this.numPixels * f32Bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.bufferPartialSums = this.device.createBuffer({ size: this.workgroupCount * f32Bytes, usage: GPUBufferUsage.STORAGE });
    this.bufferCScalar = this.device.createBuffer({ size: f32Bytes, usage: GPUBufferUsage.STORAGE });
    this.bufferMask = this.device.createBuffer({ size: this.numPixels * f32Bytes, usage: GPUBufferUsage.STORAGE });
    this.bufferSmoothMask = this.device.createBuffer({ size: this.numPixels * f32Bytes, usage: GPUBufferUsage.STORAGE });
    
    this.bufferConfigProj = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bufferConfigRes = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bufferConfigUpdate = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // Initialize basis vector
    const initialUt = new Float32Array(this.numPixels).fill(1.0 / Math.sqrt(this.numPixels));
    this.device.queue.writeBuffer(this.bufferUt, 0, initialUt);

    this.projPipeline = this.device.createComputePipeline({
      layout: 'auto', compute: { module: this.device.createShaderModule({ code: projectionWgsl }), entryPoint: 'main' }
    });
    this.resPipeline = this.device.createComputePipeline({
      layout: 'auto', compute: { module: this.device.createShaderModule({ code: residualWgsl }), entryPoint: 'main' }
    });
    this.updatePipeline = this.device.createComputePipeline({
      layout: 'auto', compute: { module: this.device.createShaderModule({ code: updateWgsl }), entryPoint: 'main' }
    });
    this.smoothPipeline = this.device.createComputePipeline({
      layout: 'auto', compute: { module: this.device.createShaderModule({ code: smoothWgsl }), entryPoint: 'main' }
    });

    this.sampler = this.device.createSampler({ minFilter: 'linear', magFilter: 'linear' });

    this.resBindGroup = this.device.createBindGroup({
      layout: this.resPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 4, resource: { buffer: this.bufferPartialSums } },
        { binding: 5, resource: { buffer: this.bufferConfigRes } },
        { binding: 6, resource: { buffer: this.bufferCScalar } }
      ]
    });

    this.updateBindGroup = this.device.createBindGroup({
      layout: this.updatePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 2, resource: { buffer: this.bufferMt } },
        { binding: 3, resource: { buffer: this.bufferUt } },
        { binding: 5, resource: { buffer: this.bufferConfigUpdate } },
        { binding: 6, resource: { buffer: this.bufferCScalar } },
        { binding: 7, resource: { buffer: this.bufferMask } }
      ]
    });
    this.smoothBindGroup = this.device.createBindGroup({
      layout: this.smoothPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.bufferMask } },         // Read the noisy mask
        { binding: 1, resource: { buffer: this.bufferSmoothMask } },   // Write to the clean mask
        { binding: 2, resource: { buffer: this.bufferConfigProj } }    // Reuse projection config (has width/height)
      ]
    });
  }

  public execute(commandEncoder: GPUCommandEncoder, externalTexture: GPUExternalTexture, width: number, height: number, threshold: number, learningRate: number): void {
    // Update config uniforms dynamically with our new UI variables!
    this.device.queue.writeBuffer(this.bufferConfigProj, 0, new Float32Array([width, height, this.numPixels, 0]));
    this.device.queue.writeBuffer(this.bufferConfigRes, 0, new Float32Array([0, 0, this.numPixels, this.workgroupCount]));
    this.device.queue.writeBuffer(this.bufferConfigUpdate, 0, new Float32Array([this.numPixels, threshold, learningRate, 0]));

    const projBindGroup = this.device.createBindGroup({
      layout: this.projPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: externalTexture },
        { binding: 2, resource: { buffer: this.bufferMt } },
        { binding: 3, resource: { buffer: this.bufferUt } },
        { binding: 4, resource: { buffer: this.bufferPartialSums } },
        { binding: 5, resource: { buffer: this.bufferConfigProj } }
      ]
    });

    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.projPipeline);
    pass.setBindGroup(0, projBindGroup);
    pass.dispatchWorkgroups(this.workgroupCount);
    
    pass.setPipeline(this.resPipeline);
    pass.setBindGroup(0, this.resBindGroup);
    pass.dispatchWorkgroups(1); 
    
    pass.setPipeline(this.updatePipeline);
    pass.setBindGroup(0, this.updateBindGroup);
    pass.dispatchWorkgroups(this.workgroupCount);

    pass.setPipeline(this.smoothPipeline);
    pass.setBindGroup(0, this.smoothBindGroup);
    pass.dispatchWorkgroups(this.workgroupCount);
    
    pass.end();
  }
}
