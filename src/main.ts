// src/main.ts
import { CameraFeed } from './media/CameraFeed';
import { WebGPUContext } from './core/WebGPUContext';
import { ORPCAPipeline } from './pipeline/ORPCAPipeline';

async function bootstrap() {
  document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
    <div style="background: #111; color: #fff; font-family: sans-serif; display: flex; flex-direction: column; align-items: center; min-height: 100vh; padding-top: 2rem;">
      <h2 style="font-family: monospace; color: #0f0;">L9 System: OR-PCA Hardware Passthrough</h2>
      
      <div style="position: relative; margin-top: 1rem;">
        <canvas id="output-canvas" style="border: 1px solid #333; transform: scaleX(-1); border-radius: 8px; box-shadow: 0 10px 30px rgba(0,0,0,0.5);"></canvas>
        
        <!-- Google Meet Style Controls -->
        <div style="position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%);">
          <button id="toggle-blur" style="padding: 12px 24px; font-size: 14px; border-radius: 24px; cursor: pointer; background: #ea4335; color: white; border: none; font-weight: bold; transition: background 0.3s;">
            Enable Background Blur
          </button>
        </div>
      </div>
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

    // Wire the new UI Button
    const toggleBtn = document.getElementById('toggle-blur') as HTMLButtonElement;
    toggleBtn.addEventListener('click', () => {
      pipeline.isBlurEnabled = !pipeline.isBlurEnabled;
      
      if (pipeline.isBlurEnabled) {
        toggleBtn.innerText = "Disable Background Blur";
        toggleBtn.style.background = "#34a853"; // Green when active
      } else {
        toggleBtn.innerText = "Enable Background Blur";
        toggleBtn.style.background = "#ea4335"; // Red when inactive
      }
    });

    console.log("[System] Production OR-PCA pipeline active.");
  } catch (error) {
    console.error("[System] Boot failure:", error);
  }
}

bootstrap();
