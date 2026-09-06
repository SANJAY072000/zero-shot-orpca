export class WebGPUContext {
  public device!: GPUDevice;
  public context!: GPUCanvasContext;
  public format!: GPUTextureFormat;

  public async initialize(canvas: HTMLCanvasElement): Promise<void> {
    if (!navigator.gpu) {
      throw new Error("WebGPU unsupported.");
    }

    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance"
    });

    if (!adapter) {
      throw new Error("No suitable GPU adapter found.");
    }

    this.device = await adapter.requestDevice();
    this.context = canvas.getContext('webgpu') as GPUCanvasContext;
    this.format = navigator.gpu.getPreferredCanvasFormat();

    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: 'premultiplied'
    });
  }
}
