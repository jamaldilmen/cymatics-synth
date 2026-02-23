import AVFoundation
import Accelerate
import CoreAudio
import AudioToolbox

/// Audio engine: AVAudioEngine + FFT + device selection via AudioUnit.
final class AudioEngine: ObservableObject {
    private var engine = AVAudioEngine()
    private var fftSetup: FFTSetup?
    private let fftSize = 2048
    private let bufferSize: AVAudioFrameCount = 2048
    private var tapCallCount = 0

    // Pre-allocated arrays for audio processing to avoid allocations on the real-time thread
    private var samples: [Float]
    private var window: [Float]
    private var realp: [Float]
    private var imagp: [Float]
    private var magnitudes: [Float]

    @Published var isRunning = false
    @Published var audioDevices: [AudioDevice] = []
    @Published var selectedDeviceID: AudioDeviceID = 0

    private(set) var analyzer = FrequencyAnalyzer()

    struct AudioDevice: Identifiable, Hashable {
        let id: AudioDeviceID
        let name: String
    }

    init() {
        // Initialize pre-allocated arrays
        self.samples = [Float](repeating: 0, count: fftSize)
        self.window = [Float](repeating: 0, count: fftSize)
        
        // Pre-compute Hann window
        vDSP_hann_window(&self.window, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))
        
        let halfN = fftSize / 2
        self.realp = [Float](repeating: 0, count: halfN)
        self.imagp = [Float](repeating: 0, count: halfN)
        self.magnitudes = [Float](repeating: 0, count: halfN)

        refreshDevices()
    }

    // MARK: - Device Enumeration (read-only CoreAudio)

    func refreshDevices() {
        var propAddr = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var dataSize: UInt32 = 0
        var status = AudioObjectGetPropertyDataSize(
            AudioObjectID(kAudioObjectSystemObject), &propAddr, 0, nil, &dataSize
        )
        guard status == noErr else { return }

        let deviceCount = Int(dataSize) / MemoryLayout<AudioDeviceID>.size
        var deviceIDs = [AudioDeviceID](repeating: 0, count: deviceCount)
        status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &propAddr, 0, nil, &dataSize, &deviceIDs
        )
        guard status == noErr else { return }

        var result: [AudioDevice] = []
        for devID in deviceIDs {
            // Check if device has input channels
            var inputAddr = AudioObjectPropertyAddress(
                mSelector: kAudioDevicePropertyStreamConfiguration,
                mScope: kAudioDevicePropertyScopeInput,
                mElement: kAudioObjectPropertyElementMain
            )
            var inputSize: UInt32 = 0
            guard AudioObjectGetPropertyDataSize(devID, &inputAddr, 0, nil, &inputSize) == noErr else { continue }
            let bufferListPtr = UnsafeMutableRawPointer.allocate(byteCount: Int(inputSize), alignment: MemoryLayout<AudioBufferList>.alignment)
            defer { bufferListPtr.deallocate() }
            guard AudioObjectGetPropertyData(devID, &inputAddr, 0, nil, &inputSize, bufferListPtr) == noErr else { continue }
            let bufferList = bufferListPtr.assumingMemoryBound(to: AudioBufferList.self).pointee
            guard bufferList.mNumberBuffers > 0 else { continue }

            // Get device name
            var nameAddr = AudioObjectPropertyAddress(
                mSelector: kAudioDevicePropertyDeviceNameCFString,
                mScope: kAudioObjectPropertyScopeGlobal,
                mElement: kAudioObjectPropertyElementMain
            )
            var nameSize = UInt32(MemoryLayout<CFString>.size)
            var nameRef: CFString = "" as CFString
            guard AudioObjectGetPropertyData(devID, &nameAddr, 0, nil, &nameSize, &nameRef) == noErr else { continue }
            let name = nameRef as String
            result.append(AudioDevice(id: devID, name: name))
        }

        audioDevices = result

        // Set default device if not set
        if selectedDeviceID == 0, let first = result.first {
            selectedDeviceID = first.id
        }
    }

    // MARK: - Start / Stop

    func start(mode: InputMode) {
        guard !isRunning else { return }

        engine = AVAudioEngine()

        // Set input device via AudioUnit BEFORE starting
        if selectedDeviceID != 0, let audioUnit = engine.inputNode.audioUnit {
            var devID = selectedDeviceID
            let status = AudioUnitSetProperty(
                audioUnit,
                kAudioOutputUnitProperty_CurrentDevice,
                kAudioUnitScope_Global, 0,
                &devID, UInt32(MemoryLayout<AudioDeviceID>.size)
            )
            if status != noErr {
                print("[CYMATICS] WARNING: Could not set input device \(selectedDeviceID), status=\(status)")
            } else {
                print("[CYMATICS] Set input device to ID \(selectedDeviceID)")
            }
        }

        let inputNode = engine.inputNode
        let hwFormat = inputNode.inputFormat(forBus: 0)
        print("[CYMATICS] Audio HW format: \(hwFormat)")

        let sampleRate = Float(hwFormat.sampleRate)
        guard sampleRate > 0 else {
            print("[CYMATICS] ERROR: sample rate is 0 — no audio input available")
            return
        }

        analyzer.sampleRate = sampleRate
        analyzer.inputMode = mode

        let log2n = vDSP_Length(log2(Float(fftSize)))
        fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2))

        let tapFormat = inputNode.outputFormat(forBus: 0)
        print("[CYMATICS] Tap format: \(tapFormat)")

        inputNode.installTap(onBus: 0, bufferSize: bufferSize, format: tapFormat) { [weak self] buffer, _ in
            self?.processBuffer(buffer)
        }

        do {
            try engine.start()
            isRunning = true
            print("[CYMATICS] Audio engine STARTED — sampleRate=\(sampleRate)")
        } catch {
            print("[CYMATICS] Audio engine FAILED: \(error)")
        }
    }

    func stop() {
        guard isRunning else { return }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        isRunning = false
        tapCallCount = 0
        analyzer.reset()
        print("[CYMATICS] Audio engine stopped")
    }

    // MARK: - Audio Processing

    private func processBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData else { return }
        let frameCount = Int(buffer.frameLength)

        let copyCount = min(frameCount, fftSize)
        memcpy(&samples, channelData[0], copyCount * MemoryLayout<Float>.size)

        // Save raw samples for YIN pitch detection (before windowing)
        let rawSamples = Array(UnsafeBufferPointer(start: channelData[0], count: min(frameCount, fftSize)))

        tapCallCount += 1
        if tapCallCount <= 3 {
            var maxSample: Float = 0
            vDSP_maxv(channelData[0], 1, &maxSample, vDSP_Length(frameCount))
            print("[CYMATICS] TAP #\(tapCallCount): frames=\(frameCount) maxSample=\(maxSample)")
        }

        // Apply pre-computed Hann window
        vDSP_vmul(samples, 1, window, 1, &samples, 1, vDSP_Length(fftSize))

        let halfN = fftSize / 2

        realp.withUnsafeMutableBufferPointer { realBuf in
            imagp.withUnsafeMutableBufferPointer { imagBuf in
                var splitComplex = DSPSplitComplex(realp: realBuf.baseAddress!, imagp: imagBuf.baseAddress!)
                samples.withUnsafeBufferPointer { sampleBuf in
                    sampleBuf.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: halfN) { complexPtr in
                        vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(halfN))
                    }
                }
                if let setup = self.fftSetup {
                    vDSP_fft_zrip(setup, &splitComplex, 1, vDSP_Length(log2(Float(self.fftSize))), FFTDirection(kFFTDirection_Forward))
                }

                vDSP_zvmags(&splitComplex, 1, &magnitudes, 1, vDSP_Length(halfN))

                var scale = Float(1.0 / Float(self.fftSize))
                vDSP_vsmul(magnitudes, 1, &scale, &magnitudes, 1, vDSP_Length(halfN))

                for i in 0..<halfN {
                    magnitudes[i] = max(0, min(255, sqrt(magnitudes[i]) * 512))
                }

                self.analyzer.process(spectrum: magnitudes, rawSamples: rawSamples)
            }
        }
    }
}

enum InputMode: String, CaseIterable {
    case voice = "Voice"
    case music = "Music"
}
