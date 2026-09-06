export class CameraFeed {
  public videoElement: HTMLVideoElement;
  private stream: MediaStream | null = null;

  constructor() {
    // Maintain off-DOM buffer to prevent UI compositor thrashing
    this.videoElement = document.createElement('video');
    this.videoElement.autoplay = true;
    this.videoElement.playsInline = true;
    this.videoElement.muted = true;
  }

  public async initialize(): Promise<void> {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          frameRate: { ideal: 30 }
        },
        audio: false
      });

      this.videoElement.srcObject = this.stream;

      await new Promise<void>((resolve) => {
        this.videoElement.onloadedmetadata = () => {
          this.videoElement.play();
          resolve();
        };
      });
    } catch (error) {
      console.error("[Hardware] Media stream acquisition failed.", error);
      throw error;
    }
  }

  public stop(): void {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
    }
  }
}
