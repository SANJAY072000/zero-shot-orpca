# WebGPU Accelerated OR-PCA: Zero-ML Real-Time Video Segmentation & Blur

A high-performance, client-side video segmentation engine running in the browser. This project replaces heavy, battery-draining Convolutional Neural Networks (CNNs) and Vision Transformers with **Online Robust Principal Component Analysis (OR-PCA)** executed entirely on the GPU via **WebGPU compute shaders**.

---

## 1. Objective

Modern video conferencing platforms rely on heavy deep learning models running on the client device to segment user silhouettes for background blurring and virtual replacements. 

This Proof of Concept (POC) demonstrates that real-time video stream background subtraction and privacy blurring can be achieved with near-zero host CPU utilization (~1.3% vs. ~35%+ in production web apps) by casting background subtraction as an online low-rank matrix decomposition problem ($M_t = L_t + S_t$) running on bare-metal GPU shaders.

---

## 2. Background Context: The Flaws of Current Solutions (Google Meet)

### First-Principles Breakdown of the Problem
Platforms like Google Meet execute client-side deep learning pipelines (typically MobileNet/TFLite-based semantic segmentation models delivered via WebAssembly with SIMD acceleration or WebGL/WebGPU compute backends). 

While effective at identifying generic human silhouettes, this approach suffers from two foundational flaws:

1. **Massive Compute & Energy Overhead:** 
   Evaluating millions of neural network weights per frame at 30–60 FPS forces sustained high CPU and GPU pipeline usage. This causes thermal throttling, loud fan noise, and severe battery drain during video calls.
2. **Semantic Rigidity ("The Disappearing Object Problem"):**
   Semantic segmentation models are strictly supervised: they are trained on datasets labeling human anatomy (hair, face, torso, arms). Because the model has no concept of *physical motion* or *transient foreground*, any non-human object held by the user (such as a pen, notepad, book, or product prototype) is classified as background and aggressively blurred out when moved away from the torso.

### Technical References
* [Google Research: Background Features in Google Meet Powered by Web ML](https://research.google/blog/background-features-in-google-meet-powered-by-web-ml/)
* [MediaPipe Selfie Segmentation Architecture Overview](https://developers.google.com/mediapipe/solutions/vision/image_segmenter)

### How to Verify These Issues Yourself in Google Meet
1. **CPU Overhead:** Open Google Meet in Chrome, turn on Background Blur, and press `Shift + Esc` to open the **Chrome Task Manager**. Observe the dedicated Meet tab consuming 30%–45%+ CPU.
2. **Object Disappearance:** Hold a pen close to your chest (it remains visible). Slowly extend your arm outward to the side of the frame. The neural network will fail to classify the pen as a human feature, instantly blurring it into the background.

---

## 3. Our Solution: Low-Rank Matrix Decomposition via WebGPU

Instead of inferring semantic classes, we model the video feed through **Online Robust PCA (OR-PCA)**:

$$M_t = L_t + S_t$$

* $M_t$: The incoming video frame matrix.
* $L_t$: The low-rank static background subspace ($L_t = U_t \cdot c_t$).
* $S_t$: The sparse foreground matrix representing active motion/variance.

```text
+--------------------------+       +----------------------------+
|  Camera Feed Ingestion   | ----> |  WebGPU Compute Shaders    |
| (importExternalTexture)  |       |  (Projection -> Residual   |
+--------------------------+       |   -> Update -> Smoothing)  |
                                   +----------------------------+
                                                 |
                                                 v
+--------------------------+       +----------------------------+
| 60 FPS Viewport Render   | <---- |  Branchless Strided Blur   |
| (Zero-Copy Framebuffer)  |       |  (Foreground-Aware Mask)   |
+--------------------------+       +----------------------------+
```

By leveraging `importExternalTexture`, video frames are ingested directly into VRAM with zero-copy CPU serialization. Parallel reduction and gradient descent update passes calculate the background basis $U_t$ and foreground mask on the GPU in sub-millisecond dispatch cycles.

* **Detailed Specification:** [Read the Complete System Design Document](https://drive.google.com/file/d/1Kind9SVvETpURYeYvM_yv6SkGAXgELB_/view?usp=sharing)

---

## 4. Setup, Execution & Benchmarking

### Prerequisites
* A browser with native WebGPU support enabled (Google Chrome 113+ or Microsoft Edge 113+ on Windows, macOS, or Linux).
* Node.js (v18.0.0 or higher) and `npm`.

### Installation & Local Run
```bash
# 1. Clone the repository
git clone [https://github.com/](https://github.com/)<your-username>/zero-shot-orpca.git
cd zero-shot-orpca

# 2. Install dependencies
npm install

# 3. Start local development server
npm run dev
```
Open your browser and navigate to the local Vite URL (typically `http://localhost:5173`).

### How to Benchmark
To profile hardware efficiency accurately:
1. Open the application tab alongside Google Chrome Task Manager (`Shift + Esc`) and Windows Task Manager (Performance tab) in a **side-by-side** configuration.
   *(Note: Avoid completely overlaying or minimizing the application window, as Chrome's Window Occlusion Culling will automatically throttle or pause the render loop).*
2. Compare the main-thread Tab CPU utilization of this WebGPU pipeline (~1.3%) against Google Meet's segmentation tab (~35%+).
3. Test the **Peripheral Object Presentation Test**: Extend a handheld pen or object away from your body. Observe that the object remains in the foreground without disappearing into the blur.

* **Verification Video:** [Watch the Benchmarking Walkthrough](https://drive.google.com/file/d/1rKvvZ5-9S6LhvF5KF400xKYsRCmpqlKg/view?usp=sharing)

---

## 5. Current Limitations & Mathematical Roadmap

While the pipeline delivers low compute overhead and solves peripheral object tracking, variance-based segmentation has fundamental trade-offs:

### Why Static Objects (e.g., Clocks, Wall Fixtures) Can Remain Visible
* **First-Principles Variance Bias:** OR-PCA identifies foreground by measuring the reconstruction error $|m_t - L_t|$ against a variance threshold $\tau$. When initialized with an unpopulated basis matrix $U_0$, high-contrast static fixtures (such as a black wall clock on a white wall) produce a massive initial error spike. If the learning rate $\alpha$ is conservative, the engine requires several frames of zero motion to fully project that high-contrast fixture into the low-rank subspace $L_t$.
* **Uniform Color Interior Holes:** When a user wears clothing that perfectly matches the luminance/chrominance of the wall behind them, the mathematical difference drops below the threshold $\tau$, causing the interior of the silhouette to momentarily classify as background.

### Future Improvements
1. **Dynamic Variance Adaptation:** Replace hardcoded thresholds with a compute-driven global variance reduction pass to automatically adjust $\tau$ and $\alpha$ during sudden scene or lighting changes.
2. **Separable Dual-Pass Gaussian Blur:** Transition the single-pass strided kernel to a two-pass ping-pong compute blur (Horizontal + Vertical) to allow deep, frosted-glass diffusion at large radii with $\mathcal{O}(2N)$ complexity.
3. **Temporal Consistency Filters:** Introduce edge-preserving bilateral filtering across sequential mask buffers to eliminate boundary edge crawling while preserving sub-pixel hair and hand contours.
