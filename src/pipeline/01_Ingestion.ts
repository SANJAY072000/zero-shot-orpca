export class IngestionPipeline {
  private device: GPUDevice;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  public importFrame(video: HTMLVideoElement): GPUExternalTexture {
    // Zero-copy texture binding directly from hardware buffer
    return this.device.importExternalTexture({
      source: video,
    });
  }
}
