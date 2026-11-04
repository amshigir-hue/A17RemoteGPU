import SwiftUI
import Metal
import Network

@main
struct A17RemoteGPUApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

final class MetalServerModel: ObservableObject {
    @Published var serverStatus = "Initializing..."
    @Published var log = "Ready."

    private var listener: NWListener?
    private let device: MTLDevice?
    private let commandQueue: MTLCommandQueue?
    private var pipelineState: MTLComputePipelineState?

    // Metal Raytracing Shader
    private let metalSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void render_scene(device uchar4* output [[buffer(0)]],
                             constant float& time [[buffer(1)]],
                             uint2 id [[thread_position_in_grid]],
                             uint2 grid_size [[threads_per_grid]]) {
        if (id.x >= grid_size.x || id.y >= grid_size.y) return;

        float u = (float(id.x) / float(grid_size.x)) * 2.0 - 1.0;
        float v = (float(id.y) / float(grid_size.y)) * 2.0 - 1.0;
        v = -v; // Invert Y

        float3 ro = float3(0.0, 0.0, -3.0);
        float3 rd = normalize(float3(u * (float(grid_size.x) / float(grid_size.y)), v, 1.5));

        // Animated Sphere position
        float3 spherePos = float3(sin(time) * 1.5, cos(time * 0.7) * 0.5, 0.0);
        float sphereRadius = 1.0;

        float3 oc = ro - spherePos;
        float b = dot(oc, rd);
        float c = dot(oc, oc) - sphereRadius * sphereRadius;
        float h = b * b - c;

        uchar3 color = uchar3(20, 25, 40); // Background color

        if (h > 0.0) {
            float t = -b - sqrt(h);
            if (t > 0.0) {
                float3 hit = ro + rd * t;
                float3 norm = normalize(hit - spherePos);
                float3 light = normalize(float3(1.0, 1.0, -1.0));
                float diff = max(dot(norm, light), 0.1);
                
                color = uchar3(uchar(diff * 220.0), uchar(diff * 80.0), uchar(diff * 255.0));
            }
        }

        uint index = id.y * grid_size.x + id.x;
        output[index] = uchar4(color.r, color.g, color.b, 255);
    }
    """

    init() {
        let dev = MTLCreateSystemDefaultDevice()
        self.device = dev
        self.commandQueue = dev?.makeCommandQueue()

        if let dev = dev {
            do {
                let library = try dev.makeLibrary(source: metalSource, options: nil)
                if let function = library.makeFunction(name: "render_scene") {
                    self.pipelineState = try dev.makeComputePipelineState(function: function)
                }
            } catch {
                self.pipelineState = nil
            }
        }
    }

    func startTCPServer() {
        guard device != nil, pipelineState != nil, commandQueue != nil else {
            self.serverStatus = "Metal initialization failed"
            return
        }

        do {
            let parameters = NWParameters.tcp
            parameters.allowLocalEndpointReuse = true
            let newListener = try NWListener(using: parameters, on: 8765)

            newListener.stateUpdateHandler = { [weak self] state in
                DispatchQueue.main.async {
                    switch state {
                    case .ready:
                        self?.serverStatus = "Listening on port 8765"
                    case .failed(let err):
                        self?.serverStatus = "Listener failed: \(err.localizedDescription)"
                    default:
                        break
                    }
                }
            }

            newListener.newConnectionHandler = { [weak self] connection in
                connection.start(queue: .global(qos: .userInteractive))
                self?.receiveFrameRequest(on: connection)
            }

            newListener.start(queue: .global())
            self.listener = newListener
        } catch {
            self.serverStatus = "Failed to create listener: \(error.localizedDescription)"
        }
    }

    private func receiveFrameRequest(on connection: NWConnection) {
        // Read 12-byte header: width (uint32), height (uint32), time (float32)
        connection.receive(minimumIncompleteLength: 12, maximumLength: 12) { [weak self] content, _, _, _ in
            guard let self = self, let data = content, data.count == 12 else {
                connection.cancel()
                return
            }

            let width = Int(data.subdata(in: 0..<4).withUnsafeBytes { $0.load(as: UInt32.self) })
            let height = Int(data.subdata(in: 4..<8).withUnsafeBytes { $0.load(as: UInt32.self) })
            var timeVal = data.subdata(in: 8..<12).withUnsafeBytes { $0.load(as: Float.self) }

            let frameData = self.renderFrame(width: width, height: height, time: &timeVal)

            connection.send(content: frameData, completion: .contentProcessed({ _ in
                // Keep receiving frames over the same socket connection
                self.receiveFrameRequest(on: connection)
            }))
        }
    }

    private func renderFrame(width: Int, height: Int, time: inout Float) -> Data {
        guard let device = device,
              let commandQueue = commandQueue,
              let pipelineState = pipelineState else { return Data() }

        let totalPixels = width * height
        let bufferSize = totalPixels * 4 // RGBA8888 (4 bytes per pixel)

        guard let outBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return Data()
        }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(outBuffer, offset: 0, index: 0)
        encoder.setBytes(&time, length: MemoryLayout<Float>.size, index: 1)

        let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let gridSize = MTLSize(width: width, height: height, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return Data(bytes: outBuffer.contents(), count: bufferSize)
    }
}

struct ContentView: View {
    @StateObject private var server = MetalServerModel()

    var body: some View {
        VStack(spacing: 20) {
            Text("A17 Pro Remote GPU Server")
                .font(.headline)
            Text(server.serverStatus)
                .foregroundColor(.green)
                .font(.subheadline)
            Text(server.log)
                .font(.caption)
        }
        .onAppear {
            server.startTCPServer()
        }
    }
}
