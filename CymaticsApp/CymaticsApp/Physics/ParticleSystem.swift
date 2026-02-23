import Foundation
import Metal
import QuartzCore

/// Manages particle state and Metal compute shader for GPU-accelerated physics.
final class ParticleSystem {
    struct Particle {
        var x: Float
        var y: Float
        var vx: Float
        var vy: Float
        var color: Float // 0 = orange, 1 = blue
    }

    /// Uniforms passed to compute shader each frame
    struct PhysicsUniforms {
        var amplitude: Float
        var forceScale: Float
        var friction: Float
        var noise: Float
        var maxSpeed: Float
        var dt: Float
        var particleCount: UInt32
        var gridSize: UInt32
        var isMusic: UInt32
        var _pad: UInt32 = 0
    }

    private(set) var particleBuffer: MTLBuffer?
    private var lutBuffer: MTLBuffer?
    private var uniformBuffer: MTLBuffer?
    private var computePipeline: MTLComputePipelineState?
    private let device: MTLDevice

    private(set) var particleCount: Int = 15000
    var inputMode: InputMode = .voice

    // Current LUT and mode
    private var currentLUT: [Float]
    private(set) var currentMode: ChladniMode

    // Background LUT generation queue
    private let lutQueue = DispatchQueue(label: "cymatics.lut", qos: .userInitiated)

    // Mode hysteresis for music
    private var pendingMode: String?
    private var pendingModeFrames = 0
    private let modeHysteresis = 15

    // Triple buffering for uniforms
    private var frameIndex = 0
    private let maxFramesInFlight = 3

    init(device: MTLDevice) {
        self.device = device
        self.currentMode = ChladniField.allModes.first { $0.m == 1 && $0.n == 1 } ?? ChladniField.allModes[0]
        // Generate initial LUT synchronously (128x128 is fast)
        self.currentLUT = ChladniField.makeLUT(mode: currentMode)
        print("[CYMATICS] Initial mode (\(currentMode.m),\(currentMode.n)) freq=\(Int(currentMode.resonantFreq))Hz")

        setupCompute()
        initParticles(count: particleCount)

        // Pre-compute all other LUTs in background so mode switches are instant
        ChladniField.precomputeAllLUTs { print("[CYMATICS] All LUTs ready — mode switches are now instant") }
    }

    // MARK: - Setup

    private func setupCompute() {
        guard let library = device.makeDefaultLibrary(),
              let function = library.makeFunction(name: "updateParticles") else {
            print("Failed to load compute shader")
            return
        }
        computePipeline = try? device.makeComputePipelineState(function: function)
    }

    func initParticles(count: Int) {
        particleCount = count
        var particles = [Particle]()
        particles.reserveCapacity(count)
        for _ in 0..<count {
            let angle = Float.random(in: 0..<(.pi * 2))
            let r = sqrtf(Float.random(in: 0..<1)) * 0.95
            particles.append(Particle(
                x: r * cosf(angle),
                y: r * sinf(angle),
                vx: 0, vy: 0,
                color: Float.random(in: 0..<1) < 0.5 ? 0 : 1
            ))
        }
        particleBuffer = device.makeBuffer(
            bytes: particles,
            length: MemoryLayout<Particle>.stride * count,
            options: .storageModeShared
        )
        updateLUTBuffer()
    }

    private func updateLUTBuffer() {
        let g = ChladniField.gridSize
        lutBuffer = device.makeBuffer(
            bytes: currentLUT,
            length: MemoryLayout<Float>.size * g * g * 2,
            options: .storageModeShared
        )
    }

    // MARK: - Async LUT Generation

    private func generateLUTAsync(mode: ChladniMode) {
        lutQueue.async { [weak self] in
            let start = CACurrentMediaTime()
            let lut = ChladniField.makeLUT(mode: mode)
            let elapsed = (CACurrentMediaTime() - start) * 1000
            print("[CYMATICS] LUT (\(mode.m),\(mode.n)) generated in \(Int(elapsed))ms")
            DispatchQueue.main.async {
                guard let self = self, mode == self.currentMode else { return }
                self.currentLUT = lut
                self.updateLUTBuffer()
            }
        }
    }

    // MARK: - Mode Update

    func updateMode(amplitude: Float, frequency: Float) {
        guard amplitude > 0.05 else { return }

        // Chladni's law: map input frequency directly to closest resonant mode
        let mode = ChladniField.modeForFrequency(frequency)
        let key = mode.key

        if key != currentMode.key {
            if key == pendingMode {
                pendingModeFrames += 1
                if pendingModeFrames >= modeHysteresis {
                    switchMode(mode)
                    pendingMode = nil
                    pendingModeFrames = 0
                }
            } else {
                pendingMode = key
                pendingModeFrames = 1
            }
        } else {
            pendingMode = nil
            pendingModeFrames = 0
        }
    }

    private func switchMode(_ mode: ChladniMode) {
        currentMode = mode
        // Keep old LUT active while new one generates (no dead zones)
        if ChladniField.hasCachedLUT(mode: mode) {
            currentLUT = ChladniField.makeLUT(mode: mode)
            updateLUTBuffer()
        } else {
            generateLUTAsync(mode: mode)
        }
    }

    // MARK: - Physics Step (GPU)

    func step(amplitude: Float, dt: Float, commandBuffer: MTLCommandBuffer) {
        guard let pipeline = computePipeline,
              let pBuf = particleBuffer,
              let lBuf = lutBuffer else { return }

        // Clamp dt to prevent friction spikes on frame hiccups (match HTML)
        let clampedDt = min(dt, 0.033)

        let isMusic = inputMode == .music
        // Force scale: boost well beyond HTML values for snappy response
        let fs = min(amplitude, 1) * (isMusic ? 0.25 : 0.15)
        let fric = powf(isMusic ? 0.12 : 0.05, clampedDt)
        let noise: Float = amplitude > 0.05 ? 0.001 * amplitude : 0.0001
        let maxSpd: Float = isMusic ? 0.015 : 0.02

        var uniforms = PhysicsUniforms(
            amplitude: amplitude,
            forceScale: fs,
            friction: fric,
            noise: noise,
            maxSpeed: maxSpd,
            dt: clampedDt,
            particleCount: UInt32(particleCount),
            gridSize: UInt32(ChladniField.gridSize),
            isMusic: isMusic ? 1 : 0
        )

        if uniformBuffer == nil {
            uniformBuffer = device.makeBuffer(
                length: MemoryLayout<PhysicsUniforms>.size * maxFramesInFlight,
                options: .storageModeShared
            )
        }

        let offset = frameIndex * MemoryLayout<PhysicsUniforms>.stride
        memcpy(uniformBuffer!.contents() + offset, &uniforms, MemoryLayout<PhysicsUniforms>.size)
        frameIndex = (frameIndex + 1) % maxFramesInFlight

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(pBuf, offset: 0, index: 0)
        encoder.setBuffer(lBuf, offset: 0, index: 1)
        encoder.setBuffer(uniformBuffer!, offset: offset, index: 2)

        let threadGroupSize = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        let threadGroups = (particleCount + threadGroupSize - 1) / threadGroupSize
        encoder.dispatchThreadgroups(
            MTLSize(width: threadGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    // MARK: - CPU Fallback

    func stepCPU(amplitude: Float, dt: Float) {
        guard let buffer = particleBuffer else { return }
        let ptr = buffer.contents().bindMemory(to: Particle.self, capacity: particleCount)

        let clampedDt = min(dt, 0.033)
        let isMusic = inputMode == .music
        let fs = min(amplitude, 1) * (isMusic ? 0.25 : 0.15)
        let fric = powf(isMusic ? 0.12 : 0.05, clampedDt)
        let noise: Float = amplitude > 0.05 ? 0.001 * amplitude : 0.0001
        let maxSpd: Float = isMusic ? 0.015 : 0.02

        for i in 0..<particleCount {
            var p = ptr[i]
            if fs > 0.001 {
                let (gx, gy) = ChladniField.sampleLUT(currentLUT, px: p.x, py: p.y)
                p.vx -= gx * fs
                p.vy -= gy * fs
            }
            p.vx += Float.random(in: -0.5...0.5) * noise
            p.vy += Float.random(in: -0.5...0.5) * noise

            let sp = sqrtf(p.vx*p.vx + p.vy*p.vy)
            if sp > maxSpd { p.vx = p.vx/sp * maxSpd; p.vy = p.vy/sp * maxSpd }
            p.vx *= fric; p.vy *= fric
            p.x += p.vx * clampedDt * 60
            p.y += p.vy * clampedDt * 60

            let r = sqrtf(p.x*p.x + p.y*p.y)
            if r > 0.99 {
                p.x = p.x/r * 0.98; p.y = p.y/r * 0.98
                p.vx *= -0.3; p.vy *= -0.3
            }
            ptr[i] = p
        }
    }
}
