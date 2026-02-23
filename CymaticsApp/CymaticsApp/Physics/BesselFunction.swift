import Foundation
import Accelerate

/// Bessel function J_m(x) computation and zeros lookup.
enum BesselFunction {
    /// Bessel zeros J_m,n for m=0..6, n=1..4
    static let zeros: [[Float]] = [
        [2.4048, 5.5201, 8.6537, 11.7915],
        [3.8317, 7.0156, 10.1735, 13.3237],
        [5.1356, 8.4172, 11.6198, 14.7960],
        [6.3802, 9.7610, 13.0152, 16.2235],
        [7.5883, 11.0647, 14.3725, 17.6160],
        [8.7715, 12.3386, 15.7002, 18.9801],
        [9.9361, 13.5893, 17.0038, 20.3208]
    ]

    /// Compute J_m(x) using power series
    static func besselJ(_ m: Int, _ x: Float) -> Float {
        if abs(x) < 1e-10 { return m == 0 ? 1 : 0 }
        var sum: Float = 0
        let hx = x / 2
        for k in 0..<25 {
            let sign: Float = k % 2 == 0 ? 1 : -1
            let num = powf(hx, Float(2*k + m))
            var den: Float = 1
            for i in 1...max(1, k) { den *= Float(i) }
            if k == 0 && m > 0 {
                for i in 1...m { den *= Float(i) }
            } else if k > 0 {
                for i in 1...(k + m) { den *= Float(i) }
            }
            let term = sign * num / den
            sum += term
            if abs(term) < 1e-15 { break }
        }
        return sum
    }
}
