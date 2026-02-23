import SwiftUI
import MetalKit

/// Holds all Metal resources in a class so SwiftUI lifecycle doesn't recreate them.
final class RenderManager: ObservableObject {
    let device: MTLDevice
    let renderer: MetalRenderer
    let particleSystem: ParticleSystem

    init() {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal not available on this Mac")
        }
        self.device = dev
        self.renderer = MetalRenderer(device: dev)
        self.particleSystem = ParticleSystem(device: dev)
        self.renderer.particleSystem = self.particleSystem
    }
}

/// Main app view: info bar, Metal canvas, pitch display, controls.
struct ContentView: View {
    @StateObject private var audioEngine = AudioEngine()
    @StateObject private var renderManager = RenderManager()

    @State private var sensitivity: Double = 2.0
    @State private var particleCount: Double = 15000
    @State private var inputMode: InputMode = .voice
    
    @State private var uiTick: Int = 0
    let uiTimer = Timer.publish(every: 1.0 / 15.0, on: .main, in: .common).autoconnect()

    var body: some View {
        let rm = renderManager

        VStack(spacing: 4) {
            // Info bar
            HStack(spacing: 20) {
                Text("Mode (\(rm.particleSystem.currentMode.m),\(rm.particleSystem.currentMode.n))")
                Text("\(rm.particleSystem.particleCount) particles")
                Text("\(rm.renderer.fps) fps")
            }
            .font(.caption)
            .foregroundColor(.secondary)
            .monospacedDigit()

            // Metal canvas (circular)
            CymaticsMetalView(renderer: rm.renderer, device: rm.device)
                .frame(width: 540, height: 540)
                .clipShape(Circle())

            // Pitch display
            PitchDisplay(analyzer: audioEngine.analyzer, isRunning: audioEngine.isRunning, uiTick: uiTick)

            // Controls
            ControlsView(
                audioEngine: audioEngine,
                sensitivity: $sensitivity,
                particleCount: $particleCount,
                inputMode: $inputMode,
                onStart: startAudio,
                onStop: stopAudio,
                onReset: resetParticles
            )
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(red: 0.04, green: 0.04, blue: 0.06))
        .onAppear {
            // Wire up analyzer reference so draw() can read audio data directly
            rm.renderer.analyzerRef = audioEngine.analyzer
            rm.renderer.inputModeRef = { [self] in inputMode }
        }
        .onChange(of: inputMode) { _, newMode in
            audioEngine.analyzer.inputMode = newMode
            rm.particleSystem.inputMode = newMode
            rm.renderer.isMusic = newMode == .music
        }
        .onChange(of: sensitivity) { _, newSens in
            audioEngine.analyzer.sensitivity = Float(newSens)
        }
        .onChange(of: particleCount) { _, newCount in
            rm.particleSystem.initParticles(count: Int(newCount))
        }
        .onReceive(uiTimer) { _ in
            if audioEngine.isRunning {
                uiTick &+= 1
            }
        }
    }

    private func startAudio() {
        audioEngine.start(mode: inputMode)
    }

    private func stopAudio() {
        audioEngine.stop()
        renderManager.renderer.amplitude = 0
    }

    private func resetParticles() {
        renderManager.particleSystem.initParticles(count: Int(particleCount))
    }
}

// MARK: - Pitch Display

struct PitchDisplay: View {
    let analyzer: FrequencyAnalyzer
    let isRunning: Bool
    let uiTick: Int

    var body: some View {
        HStack(spacing: 16) {
            Text(noteName)
                .font(.system(size: 56, weight: .heavy))
                .foregroundColor(noteColor)
                .frame(minWidth: 100)
                .monospacedDigit()

            VStack(alignment: .leading, spacing: 4) {
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 5)
                            .fill(Color(white: 0.1))
                            .frame(height: 10)
                        Rectangle()
                            .fill(Color(white: 0.27))
                            .frame(width: 2, height: 10)
                            .offset(x: geo.size.width / 2 - 1)
                        RoundedRectangle(cornerRadius: 3)
                            .fill(centsBarColor)
                            .frame(width: 6, height: 10)
                            .offset(x: centsBarOffset(width: geo.size.width))
                    }
                }
                .frame(width: 180, height: 10)

                Text(centsText)
                    .font(.subheadline.weight(.semibold))
                    .foregroundColor(noteColor)
                Text(freqText)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    private var noteName: String {
        guard isRunning, let note = analyzer.currentNote else { return "--" }
        return "\(note.name)\(note.octave)"
    }

    private var centsText: String {
        guard isRunning, analyzer.currentNote != nil else { return " " }
        let sc = analyzer.smoothedCents
        let ac = abs(sc)
        if ac < 10 { return "In tune" }
        let label = Int(sc.rounded())
        return "\(label > 0 ? "+" : "")\(label) cents"
    }

    private var freqText: String {
        guard isRunning else { return "-- Hz" }
        return "\(Int(analyzer.smoothedFreq)) Hz"
    }

    private var noteColor: Color {
        guard isRunning, analyzer.currentNote != nil else { return .primary }
        let ac = abs(analyzer.smoothedCents)
        if ac < 10 { return Color(red: 0.36, green: 0.72, blue: 0.36) }
        if ac < 25 { return Color(red: 0.91, green: 0.76, blue: 0.23) }
        return Color(red: 0.85, green: 0.33, blue: 0.31)
    }

    private var centsBarColor: Color { noteColor }

    private func centsBarOffset(width: CGFloat) -> CGFloat {
        guard isRunning, analyzer.currentNote != nil else { return width / 2 - 3 }
        let sc = analyzer.smoothedCents
        let pct = CGFloat(50 + max(-50, min(50, sc))) / 100
        return pct * width - 3
    }
}
