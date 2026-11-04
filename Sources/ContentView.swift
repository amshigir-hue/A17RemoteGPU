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
    private let pipelineState: MTLComputePipelineState?

    init() {
        let dev = MTLCreateSystemDefaultDevice()
        self.device = dev
        self.commandQueue = dev?.makeCommandQueue()

        if let dev = dev,
           let library = dev.makeDefaultLibrary(),
           let function = library.makeFunction(name: "compute_checksum") {
            self.pipelineState = try? dev.makeComputePipelineState(function: function)
        } else {
            self.pipelineState = nil
        }
    }

    func startTCPServer() {
        guard device != nil, pipelineState != nil, commandQueue != nil else {
            self.serverStatus = "Metal initialization failed (GPU/Shader missing)"
            return
        }

        do {
            let parameters = NWParameters.tcp
            parameters.allowLocalEndpointReuse = true
            // Listening on 8765 to match gpu_test.py
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
                self?.receive(on: connection)
            }

            newListener.start(queue: .global())
            self.listener = newListener
        } catch {
            self.serverStatus = "Failed to create listener: \(error.localizedDescription)"
        }
    }

    private func receive(on connection: NWConnection) {
        // Read 4-byte little-endian count
        connection.receive(minimumIncompleteLength: 4, maximumLength: 4) { [weak self] content, _, _, _ in
            guard let self = self, let data = content, data.count == 4 else {
                connection.cancel()
                return
            }

            let count = Int(data.withUnsafeBytes { $0.load(as: UInt32.self) })
            
            // Execute on Apple GPU
            let checksum = self.runCompute(count: count)

            // Construct 12-byte response: b"OKAY" + uint64 checksum
            var response = Data("OKAY".utf8)
            var sum = checksum
            response.append(UnsafeBufferPointer(start: &sum, count: 1))

            connection.send(content: response, completion: .contentProcessed({ _ in
                connection.cancel()
            }))

            DispatchQueue.main.async {
                self.log = "Calculated checksum for \(count) elements on GPU:\n\(checksum)"
            }
        }
    }

    private func runCompute(count: Int) -> UInt64 {
        guard let device = device,
              let commandQueue = commandQueue,
              let pipelineState = pipelineState,
              count > 0 else { return 0 }

        // Allocate buffer for per-thread output
        let bufferSize = count * MemoryLayout<UInt32>.size
        guard let outBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return 0
        }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(outBuffer, offset: 0, index: 0)

        let maxThreads = pipelineState.maxTotalThreadsPerThreadgroup
        let threadsPerGroup = MTLSize(width: min(count, maxThreads), height: 1, depth: 1)
        let numGroups = MTLSize(width: (count + threadsPerGroup.width - 1) / threadsPerGroup.width, height: 1, depth: 1)

        encoder.dispatchThreadgroups(numGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Sum up GPU results with 64-bit wrap-around
        let rawPointer = outBuffer.contents().bindMemory(to: UInt32.self, capacity: count)
        var total: UInt64 = 0
        for i in 0..<count {
            total = total &+ UInt64(rawPointer[i])
        }
        return total
    }
}

struct ContentView: View {
    @StateObject private var server = MetalServerModel()

    var body: some View {
        VStack(spacing: 20) {
            Text("A17 Remote Metal Worker")
                .font(.headline)

            Text(server.serverStatus)
                .foregroundColor(.green)
                .font(.subheadline)

            ScrollView {
                Text(server.log)
                    .font(.system(.caption, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
            }
            .background(Color(.secondarySystemBackground))
            .cornerRadius(8)
            .padding()
        }
        .onAppear {
            server.startTCPServer()
        }
    }
}
