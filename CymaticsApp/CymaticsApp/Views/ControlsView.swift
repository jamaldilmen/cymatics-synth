import SwiftUI

struct ControlsView: View {
    @ObservedObject var audioEngine: AudioEngine
    @Binding var sensitivity: Double
    @Binding var particleCount: Double
    @Binding var inputMode: InputMode
    var onStart: () -> Void
    var onStop: () -> Void
    var onReset: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            // Device picker
            Picker("Input", selection: $audioEngine.selectedDeviceID) {
                ForEach(audioEngine.audioDevices) { device in
                    Text(device.name).tag(device.id)
                }
            }
            .frame(width: 160)

            Button(audioEngine.isRunning ? "Stop" : "Start") {
                if audioEngine.isRunning { onStop() } else { onStart() }
            }
            .buttonStyle(.bordered)
            .tint(audioEngine.isRunning ? .blue : nil)

            Button(inputMode.rawValue) {
                inputMode = inputMode == .voice ? .music : .voice
            }
            .buttonStyle(.bordered)
            .tint(inputMode == .music ? .blue : nil)

            Button("Reset") { onReset() }
                .buttonStyle(.bordered)

            VStack(spacing: 2) {
                Text("Sensitivity").font(.caption2).foregroundColor(.secondary)
                Slider(value: $sensitivity, in: 0.5...5, step: 0.1)
                    .frame(width: 100)
            }

            VStack(spacing: 2) {
                Text("Particles").font(.caption2).foregroundColor(.secondary)
                Slider(value: $particleCount, in: 3000...50000, step: 1000)
                    .frame(width: 100)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }
}
