import SwiftUI
import MetalKit

struct CymaticsMetalView: NSViewRepresentable {
    let renderer: MetalRenderer
    let device: MTLDevice

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeNSView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: device)
        view.delegate = renderer
        view.colorPixelFormat = .bgra8Unorm
        view.clearColor = MTLClearColor(red: 0.04, green: 0.04, blue: 0.06, alpha: 1.0)
        view.layer?.isOpaque = true
        
        // Use native MTKView internal loop (VSync on macOS)
        view.isPaused = false
        view.enableSetNeedsDisplay = false
        view.preferredFramesPerSecond = 60
        
        return view
    }

    func updateNSView(_ view: MTKView, context: Context) {}

    static func dismantleNSView(_ view: MTKView, coordinator: Coordinator) {
    }

    class Coordinator {
    }
}
