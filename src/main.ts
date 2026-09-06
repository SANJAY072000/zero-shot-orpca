import { CameraFeed } from './media/CameraFeed';
import { WebGPUContext } from './core/WebGPUContext';
import { ORPCAPipeline } from './pipeline/ORPCAPipeline';

async function bootstrap() {
  document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
    <div style="background: #111; color: #0f0; font-family: monospace; padding: 2rem; min-height: 100vh;">
      <h2>L9 System: OR-PCA Hardware Passthrough</h2>
      <canvas id="output-canvas" style="border: 1px solid #333; transform: scaleX(-1); margin-top: 1rem;"></canvas>
    </div>
  `;

  try {
    const camera = new CameraFeed();
    await camera.initialize();

    const canvas = document.getElementById('output-canvas') as HTMLCanvasElement;
    canvas.width = camera.videoElement.videoWidth;
    canvas.height = camera.videoElement.videoHeight;
    canvas.style.width = '800px';

    const gpu = new WebGPUContext();
    await gpu.initialize(canvas);

    const pipeline = new ORPCAPipeline(gpu.device, gpu.context, gpu.format, camera.videoElement.videoWidth, camera.videoElement.videoHeight);
    pipeline.initialize();
    pipeline.start(camera.videoElement);

    console.log("[System] OR-PCA pipeline active.");
  } catch (error) {
    console.error("[System] Boot failure:", error);
  }
}

bootstrap();
