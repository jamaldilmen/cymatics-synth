import Foundation
import QuartzCore

/// Chladni mode definition and gradient LUT generation.
/// Physics: f_mn = C * α_mn^2 (Chladni's law for circular plate with fixed center)
struct ChladniMode: Equatable {
    let m: Int
    let n: Int
    let alpha: Float
    let resonantFreq: Float // f ∝ α² mapped to our frequency range

    var key: String { "\(m),\(n)" }
}

final class ChladniField {
    static let gridSize = 128

    // Frequency range for mode mapping
    static let freqLow: Float = 80
    static let freqHigh: Float = 2000

    /// All available modes, sorted by resonant frequency (α²)
    /// Chladni's law: f = C(m + 2n)^p ≈ C * α² for circular plates
    static let allModes: [ChladniMode] = {
        var modes: [ChladniMode] = []
        let alphaMin = BesselFunction.zeros[0][0] // 2.4048 — lowest mode
        let alphaMax = BesselFunction.zeros[6][3] // 20.3208 — highest mode
        let freqScale = (freqHigh - freqLow) / (alphaMax * alphaMax - alphaMin * alphaMin)

        for m in 0...6 {
            for n in 0..<4 {
                if m == 0 && n == 0 { continue }
                let alpha = BesselFunction.zeros[m][n]
                let freq = freqLow + (alpha * alpha - alphaMin * alphaMin) * freqScale
                modes.append(ChladniMode(m: m, n: n + 1, alpha: alpha, resonantFreq: freq))
            }
        }
        return modes.sorted { $0.resonantFreq < $1.resonantFreq }
    }()

    /// Map input frequency (Hz) to closest resonant mode
    static func modeForFrequency(_ freq: Float) -> ChladniMode {
        var best = allModes[0]
        var bestDist = abs(freq - best.resonantFreq)
        for mode in allModes {
            let dist = abs(freq - mode.resonantFreq)
            if dist < bestDist {
                bestDist = dist
                best = mode
            }
        }
        return best
    }

    // MARK: - Potential Field

    /// Z²(m, alpha, r, theta) — vibration intensity squared
    private static func z2(_ m: Int, _ alpha: Float, _ r: Float, _ th: Float) -> Float {
        let j = BesselFunction.besselJ(m, alpha * r)
        let a: Float = m == 0 ? 1 : cosf(Float(m) * th)
        return j * a * j * a
    }

    /// Potential = Z² + boundary repulsion
    private static func potential(_ m: Int, _ alpha: Float, _ r: Float, _ th: Float) -> Float {
        if r > 1.0 { return 10 }
        let z = z2(m, alpha, r, th)
        let edge: Float = r > 0.92 ? 0.5 * powf((r - 0.92) / 0.08, 3) : 0
        return z + edge
    }

    // MARK: - Gradient LUT

    // Cache for pre-computed LUTs
    private static var lutCache: [String: [Float]] = [:]

    static func hasCachedLUT(mode: ChladniMode) -> Bool {
        return lutCache[mode.key] != nil
    }

    /// Pre-generate all LUTs on background thread
    static func precomputeAllLUTs(completion: @escaping () -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            let start = CACurrentMediaTime()
            for mode in allModes {
                _ = makeLUT(mode: mode)
            }
            let elapsed = (CACurrentMediaTime() - start) * 1000
            print("[CYMATICS] All \(allModes.count) LUTs pre-computed in \(Int(elapsed))ms")
            DispatchQueue.main.async { completion() }
        }
    }

    /// Build gradient LUT for the given mode.
    static func makeLUT(mode: ChladniMode) -> [Float] {
        if let cached = lutCache[mode.key] {
            return cached
        }

        let g = gridSize
        var lut = [Float](repeating: 0, count: g * g * 2)
        let eps: Float = 2.0 / Float(g)

        for iy in 0..<g {
            for ix in 0..<g {
                let px = Float(ix) / Float(g - 1) * 2 - 1
                let py = Float(iy) / Float(g - 1) * 2 - 1
                let rr = sqrtf(px*px + py*py)
                let base = (iy * g + ix) * 2
                if rr > 1.0 || rr < 0.001 { continue }

                let pxp = px + eps, pxm = px - eps
                let pyp = py + eps, pym = py - eps
                let rxp = sqrtf(pxp*pxp + py*py)
                let rxm = sqrtf(pxm*pxm + py*py)
                let ryp = sqrtf(px*px + pyp*pyp)
                let rym = sqrtf(px*px + pym*pym)

                lut[base] = (potential(mode.m, mode.alpha, rxp, atan2f(py, pxp))
                           - potential(mode.m, mode.alpha, rxm, atan2f(py, pxm))) / (2*eps)
                lut[base+1] = (potential(mode.m, mode.alpha, ryp, atan2f(pyp, px))
                             - potential(mode.m, mode.alpha, rym, atan2f(pym, px))) / (2*eps)
            }
        }

        // Normalize
        var maxMag: Float = 0
        for i in 0..<(g*g) {
            let gx = lut[i*2], gy = lut[i*2+1]
            let mg = sqrtf(gx*gx + gy*gy)
            if mg > maxMag { maxMag = mg }
        }
        if maxMag > 0 {
            for i in 0..<(g*g*2) { lut[i] /= maxMag }
        }

        lutCache[mode.key] = lut
        return lut
    }

    /// Sample the LUT with bilinear interpolation
    static func sampleLUT(_ lut: [Float], px: Float, py: Float) -> (Float, Float) {
        let g = gridSize
        let fx = (px + 1) / 2 * Float(g - 1)
        let fy = (py + 1) / 2 * Float(g - 1)
        let ix = max(0, min(g - 2, Int(fx)))
        let iy = max(0, min(g - 2, Int(fy)))
        let tx = fx - Float(ix)
        let ty = fy - Float(iy)
        let a = (iy * g + ix) * 2
        let b = a + 2
        let c = ((iy + 1) * g + ix) * 2
        let d = c + 2

        let dx = (1-ty) * ((1-tx)*lut[a] + tx*lut[b]) + ty * ((1-tx)*lut[c] + tx*lut[d])
        let dy = (1-ty) * ((1-tx)*lut[a+1] + tx*lut[b+1]) + ty * ((1-tx)*lut[c+1] + tx*lut[d+1])
        return (dx, dy)
    }
}
