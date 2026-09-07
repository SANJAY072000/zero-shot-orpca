// src/pipeline/ORPCAPipeline.ts
import { IngestionPipeline } from './01_Ingestion';
import { ComputePipeline } from './02_Compute';
import { RenderPipeline } from './03_Render';

export class ORPCAPipeline {
  private ingestion: IngestionPipeline;
  private compute: ComputePipeline;
  private renderer: RenderPipeline;
  private isRunning: boolean = false;

  public isBlurEnabled: boolean = false;

  constructor(private device: GPUDevice, private context: GPUCanvasContext, format: GPUTextureFormat, width: number, height: number) {
    this.ingestion = new IngestionPipeline(device);
    this.compute = new ComputePipeline(device, width, height);
    this.renderer = new RenderPipeline(device, format);
  }

  public initialize(): void {
    this.compute.initialize();
    this.renderer.initialize();
  }

  public start(video: HTMLVideoElement): void {
    this.isRunning = true;

    const frameLoop = () => {
      if (!this.isRunning) return;

      if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        const externalTexture = this.ingestion.importFrame(video);
        const commandEncoder = this.device.createCommandEncoder();

        // Locked in the stable mask variables
        this.compute.execute(commandEncoder, externalTexture, video.videoWidth, video.videoHeight, 0.10, 0.0005);
        
        this.renderer.render(this.context, externalTexture, this.compute.bufferSmoothMask, commandEncoder, this.isBlurEnabled);

        this.device.queue.submit([commandEncoder.finish()]);
      }
      requestAnimationFrame(frameLoop);
    };

    requestAnimationFrame(frameLoop);
  }
}
