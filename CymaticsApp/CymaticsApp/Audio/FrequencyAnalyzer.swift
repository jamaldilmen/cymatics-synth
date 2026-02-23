import Foundation
import QuartzCore
import Accelerate

/// Multi-band energy extraction, spectral centroid, envelope follower.
/// Voice mode: YIN pitch detection. Music mode: FFT spectral centroid.
final class FrequencyAnalyzer {
    var sampleRate: Float = 48000
    var inputMode: InputMode = .voice
    var sensitivity: Float = 2.0

    // Results (read from main thread)
    private(set) var amplitude: Float = 0
    private(set) var dominantFreq: Float = 0
    private(set) var smoothedFreq: Float = 0
    private(set) var currentNote: NoteInfo?
    private(set) var smoothedCents: Float = 0
    private(set) var smoothCentroid: Float = 500
    private(set) var autoGain: Float = 1.0
    private(set) var energyEnv: Float = 0
    private(set) var peakEnv: Float = 0

    // Internal state
    private var envelope: Float = 0
    private var autoGainRMS: Float = 0
    private var lastRawFreq: Float = 0
    private var lastProcessTime: Double = 0

    // YIN pre-allocated buffers
    private var yinBuffer: [Float] = []
    private var yinDiff: [Float] = []

    // Noise gate
    var noiseFloor: [Float]?
    var noiseGateRMS: Float = 0

    private(set) var lastSpectrum: [Float] = []

    private static let noteNames = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

    struct NoteInfo {
        let name: String
        let octave: Int
        let cents: Float
        let midi: Int
    }

    func reset() {
        amplitude = 0
        envelope = 0
        energyEnv = 0
        peakEnv = 0
        dominantFreq = 0
        smoothedFreq = 0
        currentNote = nil
        smoothedCents = 0
        autoGain = 1.0
        autoGainRMS = 0
        smoothCentroid = 500
        lastRawFreq = 0
        lastProcessTime = 0
        noiseFloor = nil
        lastSpectrum = []
    }

    // MARK: - Main Process (spectrum for energy + raw samples for YIN)

    func process(spectrum: [Float], rawSamples: [Float]?) {
        let now = CACurrentMediaTime()
        let elapsed = lastProcessTime > 0 ? now - lastProcessTime : 1.0 / 60.0
        lastProcessTime = now
        let frames = Float(elapsed * 60)

        lastSpectrum = spectrum
        let bins = spectrum.count
        let ny = sampleRate / 2
        let bHz = ny / Float(bins)
        let lo = Int(60 / bHz)
        let hi = min(Int(3000 / bHz), bins - 1)

        // RMS energy from spectrum (used by both modes)
        var maxVal: Float = 0
        var sum: Float = 0
        var cnt = 0
        var hot = 0
        for i in lo...hi {
            let cleaned = noiseFloor != nil ? max(0, spectrum[i] - noiseFloor![i]) : spectrum[i]
            if cleaned > maxVal { maxVal = cleaned }
            sum += cleaned * cleaned
            cnt += 1
            if let nf = noiseFloor, spectrum[i] > nf[i] * 1.8 { hot += 1 }
        }
        let rms = sqrt(sum / Float(max(cnt, 1)))
        let sens = sensitivity

        if inputMode == .voice {
            // YIN pitch detection from raw samples
            if let samples = rawSamples {
                let yinFreq = yinPitch(samples: samples)
                if yinFreq > 0 {
                    dominantFreq = yinFreq
                }
            }
            processVoice(rms: rms, maxVal: maxVal, hot: hot, sens: sens, frames: frames)
        } else {
            // FFT-based spectral analysis for music
            dominantFreq = fftDominantFreq(spectrum: spectrum, lo: lo, hi: hi, bHz: bHz, maxVal: maxVal)
            processMusic(spectrum: spectrum, bins: bins, bHz: bHz, ny: ny, sens: sens, frames: frames)
        }

        // Pitch display
        if amplitude > 0 && dominantFreq > 60 {
            let sf = smoothPitch(dominantFreq)
            currentNote = freqToNote(sf)
            if let note = currentNote {
                smoothedCents += (note.cents - smoothedCents) * 0.2
            }
        } else {
            currentNote = nil
            lastRawFreq = 0
            smoothedCents = 0
        }
    }

    // MARK: - YIN Pitch Detection

    /// YIN algorithm: robust fundamental frequency detection from time-domain samples.
    /// Returns detected frequency in Hz, or 0 if no pitch found.
    private func yinPitch(samples: [Float]) -> Float {
        let n = samples.count
        let halfN = n / 2
        guard halfN > 0 else { return 0 }

        // Ensure buffers are sized
        if yinDiff.count != halfN {
            yinDiff = [Float](repeating: 0, count: halfN)
        }

        // Step 1: Difference function d(τ) = Σ(x[j] - x[j+τ])²
        yinDiff[0] = 0
        for tau in 1..<halfN {
            var sum: Float = 0
            for j in 0..<halfN {
                let diff = samples[j] - samples[j + tau]
                sum += diff * diff
            }
            yinDiff[tau] = sum
        }

        // Step 2: Cumulative mean normalized difference
        var runningSum: Float = 0
        yinDiff[0] = 1
        for tau in 1..<halfN {
            runningSum += yinDiff[tau]
            if runningSum > 0 {
                yinDiff[tau] = yinDiff[tau] * Float(tau) / runningSum
            } else {
                yinDiff[tau] = 1
            }
        }

        // Step 3: Absolute threshold — find first dip below threshold, then its local minimum
        let threshold: Float = 0.2
        let tauMin = max(1, Int(sampleRate / 1500))  // 1500 Hz max
        let tauMax = min(halfN - 1, Int(sampleRate / 60))  // 60 Hz min

        var bestTau = -1
        var tau = tauMin
        while tau < tauMax {
            if yinDiff[tau] < threshold {
                // Walk to local minimum
                var minTau = tau
                while minTau + 1 < tauMax && yinDiff[minTau + 1] < yinDiff[minTau] {
                    minTau += 1
                }
                bestTau = minTau
                break
            }
            tau += 1
        }

        // Fallback: global minimum
        if bestTau < 0 {
            var minVal: Float = Float.greatestFiniteMagnitude
            for t in tauMin..<tauMax {
                if yinDiff[t] < minVal {
                    minVal = yinDiff[t]
                    bestTau = t
                }
            }
            if minVal > 0.5 { return 0 }
        }

        guard bestTau > 0 && bestTau < tauMax else { return 0 }

        // Step 4: Octave check — if there's a dip at 2x tau (fundamental),
        // prefer it over the harmonic. This prevents octave-too-high errors.
        let doubleTau = bestTau * 2
        if doubleTau < tauMax {
            // Find local minimum near 2*bestTau (within ±10%)
            let searchLo = max(tauMin, Int(Float(doubleTau) * 0.9))
            let searchHi = min(tauMax - 1, Int(Float(doubleTau) * 1.1))
            var minAt2x: Float = Float.greatestFiniteMagnitude
            var bestAt2x = -1
            for t in searchLo...searchHi {
                if yinDiff[t] < minAt2x {
                    minAt2x = yinDiff[t]
                    bestAt2x = t
                }
            }
            // If the dip at 2x is reasonable (< 0.3), it's likely the true fundamental
            if bestAt2x > 0 && minAt2x < 0.3 {
                bestTau = bestAt2x
            }
        }

        // Step 5: Parabolic interpolation for sub-sample accuracy
        let t = bestTau
        if t > 0 && t < halfN - 1 {
            let a = yinDiff[t - 1]
            let b = yinDiff[t]
            let c = yinDiff[t + 1]
            let den = 2 * (a - 2*b + c)
            if den != 0 {
                let shift = (a - c) / den
                return sampleRate / (Float(t) + shift)
            }
        }

        return sampleRate / Float(bestTau)
    }

    // MARK: - FFT Peak (music mode)

    private func fftDominantFreq(spectrum: [Float], lo: Int, hi: Int, bHz: Float, maxVal: Float) -> Float {
        var mxVal: Float = 0
        var maxBin = 0
        for i in lo...hi {
            let cleaned = noiseFloor != nil ? max(0, spectrum[i] - noiseFloor![i]) : spectrum[i]
            if cleaned > mxVal { mxVal = cleaned; maxBin = i }
        }
        if maxBin > lo && maxBin < hi && mxVal > 5 {
            let a = noiseFloor != nil ? max(0, spectrum[maxBin-1] - noiseFloor![maxBin-1]) : spectrum[maxBin-1]
            let b = mxVal
            let c = noiseFloor != nil ? max(0, spectrum[maxBin+1] - noiseFloor![maxBin+1]) : spectrum[maxBin+1]
            let den = 2 * (a - 2*b + c)
            return (Float(maxBin) + (den != 0 ? (a - c) / den : 0)) * bHz
        }
        return Float(maxBin) * bHz
    }

    // MARK: - Voice Mode

    private func processVoice(rms: Float, maxVal: Float, hot: Int, sens: Float, frames: Float) {
        let gate: Float = noiseFloor != nil ? noiseGateRMS * 1.2 : 8
        let hasSignal = noiseFloor != nil ? (rms >= gate && maxVal >= 15 && hot >= 2) : (rms >= gate && maxVal >= 8)
        let gated = hasSignal ? rms / 255 * sens : Float(0)

        let freqRatio = clamp01((dominantFreq - 80) / (2000 - 80))
        // Fast decay: ~250ms to near-zero so 125 BPM beats are clearly visible
        // 0.88^15frames(250ms@60fps) ≈ 0.14 → good beat separation
        let baseFall: Float = 0.88 - (0.88 - 0.80) * freqRatio
        let dynFall = powf(baseFall, frames)

        if gated > envelope { envelope = gated }
        else { envelope *= dynFall }
        if envelope < 0.005 { envelope = 0 }
        amplitude = envelope
    }

    // MARK: - Music Mode

    private func processMusic(spectrum: [Float], bins: Int, bHz: Float, ny: Float, sens: Float, frames: Float) {
        let bassLo = Int(30 / bHz), bassHi = Int(200 / bHz)
        let midLo = Int(200 / bHz), midHi = Int(2000 / bHz)
        let hiLo = Int(2000 / bHz), hiHi = min(Int(8000 / bHz), bins - 1)

        var bassSum: Float = 0, bassCnt = 0
        var midSum: Float = 0, midCnt = 0
        var hiSum: Float = 0, hiCnt = 0
        var centNum: Float = 0, centDen: Float = 0

        for i in bassLo...hiHi {
            let v = spectrum[i]
            let freq_i = Float(i) * bHz
            if i <= bassHi { bassSum += v*v; bassCnt += 1 }
            else if i <= midHi { midSum += v*v; midCnt += 1 }
            else { hiSum += v*v; hiCnt += 1 }
            centNum += v*v * freq_i
            centDen += v*v
        }

        let bassRMS = sqrt(bassSum / Float(max(bassCnt, 1)))
        let midRMS = sqrt(midSum / Float(max(midCnt, 1)))
        let hiRMS = sqrt(hiSum / Float(max(hiCnt, 1)))
        let bandEnergy = bassRMS * 0.55 + midRMS * 0.30 + hiRMS * 0.15

        autoGainRMS += (bandEnergy - autoGainRMS) * 0.008
        if autoGainRMS > 3 {
            let target = 80 / autoGainRMS
            autoGain += (clamp(target, 0.05, 6.0) - autoGain) * 0.01
        }

        let scaled = bandEnergy * autoGain / 255 * sens
        let gated = bandEnergy > 2 ? scaled : Float(0)

        if centDen > 0 {
            let rawCentroid = centNum / centDen
            smoothCentroid += (rawCentroid - smoothCentroid) * 0.03
        }

        let freqRatio = clamp01((smoothCentroid - 80) / (2000 - 80))
        let baseEnergyFall: Float = 0.99 - (0.99 - 0.975) * freqRatio
        let basePeakFall: Float = 0.94 - (0.94 - 0.88) * freqRatio
        let dynAttackAlpha: Float = min(1, (0.08 + (0.25 - 0.08) * freqRatio) * frames)
        let dynEnergyFall = powf(baseEnergyFall, frames)
        let dynPeakFall = powf(basePeakFall, frames)

        if gated > energyEnv { energyEnv += (gated - energyEnv) * dynAttackAlpha }
        else { energyEnv *= dynEnergyFall }
        if energyEnv < 0.001 { energyEnv = 0 }

        if gated > peakEnv { peakEnv = gated }
        else { peakEnv *= dynPeakFall }
        if peakEnv < 0.001 { peakEnv = 0 }

        amplitude = energyEnv * 0.6 + peakEnv * 0.4
    }

    // MARK: - Pitch Smoothing

    private func smoothPitch(_ freq: Float) -> Float {
        if lastRawFreq == 0 {
            smoothedFreq = freq
        } else {
            smoothedFreq += (freq - smoothedFreq) * 0.5
        }
        lastRawFreq = freq
        return smoothedFreq
    }

    private func freqToNote(_ freq: Float) -> NoteInfo? {
        guard freq >= 20 else { return nil }
        let semitones = 12 * log2(freq / 440)
        let rounded = roundf(semitones)
        let cents = (semitones - rounded) * 100
        let midi = Int(rounded) + 69
        guard midi >= 0 && midi <= 127 else { return nil }
        let name = Self.noteNames[((midi % 12) + 12) % 12]
        let octave = midi / 12 - 2  // Ableton convention: C3 = middle C (MIDI 60)
        return NoteInfo(name: name, octave: octave, cents: cents, midi: midi)
    }

    // MARK: - Helpers

    private func clamp01(_ x: Float) -> Float { max(0, min(1, x)) }
    private func clamp(_ x: Float, _ lo: Float, _ hi: Float) -> Float { max(lo, min(hi, x)) }
}
