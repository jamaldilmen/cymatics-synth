import MetalKit

/// Metal render pipeline: draws background plate + instanced particles as points.
/// Physics runs inside draw() — syncs audio data each frame.
final class MetalRenderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue?
    private var particlePipeline: MTLRenderPipelineState?
    private var platePipeline: MTLRenderPipelineState?
    private var renderUniformBuffer: MTLBuffer?
    private var useGPUCompute = false

    var particleSystem: ParticleSystem?
    var amplitude: Float = 0
    var isMusic: Bool = false

    // Audio sync — set from outside, read in draw()
    var analyzerRef: FrequencyAnalyzer?
    var inputModeRef: (() -> InputMode)?

    // FPS tracking
    private var frameCount = 0
    private var lastFPSTime: CFTimeInterval = 0
    private(set) var fps: Int = 0
    private var drawCallCount = 0

    struct RenderUniforms {
        var viewportSize: SIMD2<Float>
        var particleSize: Float
        var plateRadius: Float
    }

    init(device: MTLDevice) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()
        super.init()
        setupPipelines()
    }

    private func setupPipelines() {
        guard let library = device.makeDefaultLibrary() else {
            print("[CYMATICS] FATAL: No Metal library")
            return
        }

        if library.makeFunction(name: "updateParticles") != nil {
            print("[CYMATICS] Compute shader loaded OK")
            useGPUCompute = true
        } else {
            print("[CYMATICS] WARNING: No compute shader, CPU fallback")
        }

        let particleDesc = MTLRenderPipelineDescriptor()
        particleDesc.vertexFunction = library.makeFunction(name: "particleVertex")
        particleDesc.fragmentFunction = library.makeFunction(name: "particleFragment")
        particleDesc.colorAttachments[0].pixelFormat = .bgra8Unorm
        particleDesc.colorAttachments[0].isBlendingEnabled = true
        particleDesc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        particleDesc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        particleDesc.colorAttachments[0].sourceAlphaBlendFactor = .one
        particleDesc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        do {
            particlePipeline = try device.makeRenderPipelineState(descriptor: particleDesc)
            print("[CYMATICS] Particle render pipeline OK")
        } catch {
            print("[CYMATICS] Particle pipeline FAILED: \(error)")
        }

        let plateDesc = MTLRenderPipelineDescriptor()
        plateDesc.vertexFunction = library.makeFunction(name: "plateVertex")
        plateDesc.fragmentFunction = library.makeFunction(name: "plateFragment")
        plateDesc.colorAttachments[0].pixelFormat = .bgra8Unorm
        plateDesc.colorAttachments[0].isBlendingEnabled = true
        plateDesc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        plateDesc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        do {
            platePipeline = try device.makeRenderPipelineState(descriptor: plateDesc)
        } catch {
            print("[CYMATICS] Plate pipeline FAILED: \(error)")
        }

        renderUniformBuffer = device.makeBuffer(
            length: MemoryLayout<RenderUniforms>.size,
            options: .storageModeShared
        )
    }

    // MARK: - MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        // NO SEMAPHORE — we're called on main thread via needsDisplay,
        // so blocking here = deadlock. Just render synchronously.

        let now = CACurrentMediaTime()
        frameCount += 1
        drawCallCount += 1
        if now - lastFPSTime >= 1.0 {
            fps = frameCount
            frameCount = 0
            lastFPSTime = now
            // Log every second for first 30 seconds
            if drawCallCount < 1800 {
                // Read back first particle position to verify movement
                var p0 = ""
                if let buf = particleSystem?.particleBuffer {
                    let ptr = buf.contents().bindMemory(to: Float.self, capacity: 5)
                    p0 = " p0=(\(String(format:"%.4f",ptr[0])),\(String(format:"%.4f",ptr[1]))) v=(\(String(format:"%.5f",ptr[2])),\(String(format:"%.5f",ptr[3])))"
                }
                let mode = particleSystem?.currentMode
                let modeStr = mode != nil ? " mode=(\(mode!.m),\(mode!.n)) f=\(Int(mode!.resonantFreq))Hz" : ""
                print("[CYMATICS] fps=\(fps) amp=\(String(format:"%.3f", amplitude))\(modeStr)\(p0)")
            }
        }

        guard let ps = particleSystem,
              let drawable = view.currentDrawable,
              let passDesc = view.currentRenderPassDescriptor,
              let cq = commandQueue,
              let commandBuffer = cq.makeCommandBuffer() else {
            return
        }

        // Sync audio -> physics
        if let analyzer = analyzerRef {
            amplitude = analyzer.amplitude
            let mode = inputModeRef?() ?? .voice
            let freq = mode == .music ? analyzer.smoothCentroid : analyzer.dominantFreq
            ps.updateMode(amplitude: amplitude, frequency: freq)
        }

        // Physics
        let dt: Float = 1.0 / 60.0
        if useGPUCompute {
            ps.step(amplitude: amplitude, dt: dt, commandBuffer: commandBuffer)
        } else {
            ps.stepCPU(amplitude: amplitude, dt: dt)
        }

        // Uniforms
        let size = view.drawableSize
        let plateRadius = Float(min(size.width, size.height) / max(size.width, size.height)) * 0.9
        var uniforms = RenderUniforms(
            viewportSize: SIMD2<Float>(Float(size.width), Float(size.height)),
            particleSize: isMusic ? 3.5 : 2.5,
            plateRadius: plateRadius
        )
        if let buf = renderUniformBuffer {
            memcpy(buf.contents(), &uniforms, MemoryLayout<RenderUniforms>.size)
        }

        passDesc.colorAttachments[0].clearColor = MTLClearColor(red: 0.04, green: 0.04, blue: 0.06, alpha: 1.0)
        passDesc.colorAttachments[0].loadAction = .clear

        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: passDesc) else { return }

        if let platePipe = platePipeline {
            renderEncoder.setRenderPipelineState(platePipe)
            renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        }

        if let particlePipe = particlePipeline, let pBuf = ps.particleBuffer {
            renderEncoder.setRenderPipelineState(particlePipe)
            renderEncoder.setVertexBuffer(pBuf, offset: 0, index: 0)
            if let rBuf = renderUniformBuffer {
                renderEncoder.setVertexBuffer(rBuf, offset: 0, index: 1)
            }
            renderEncoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: ps.particleCount)
        }

        renderEncoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}
