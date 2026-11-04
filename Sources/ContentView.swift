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
           let function = library.makeFunction(name: "square_array") {
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
            let newListener = try NWListener(using: parameters, on: 8888)

            newListener.stateUpdateHandler = { [weak self] state in
                DispatchQueue.main.async {
                    switch state {
                    case .ready:
                        self?.serverStatus = "Listening on port 8888"
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
        connection.receive(minimumIncompleteLength: 4, maximumLength: 4) { [weak self] content, _, _, _ in
            guard let self = self, let data = content, data.count == 4 else {
                connection.cancel()
                return
            }

            let count = Int(data.withUnsafeBytes { $0.load(as: UInt32.self) })
            let payloadBytes = count * MemoryLayout<Float>.size

            connection.receive(minimumIncompleteLength: payloadBytes, maximumLength: payloadBytes) { [weak self] bufferData, _, _, _ in
                guard let self = self, let bufferData = bufferData, bufferData.count == payloadBytes else {
                    connection.cancel()
                    return
                }

                let inputs: [Float] = bufferData.withUnsafeBytes { rawPtr in
                    let bound = rawPtr.bindMemory(to: Float.self)
                    return Array(bound)
                }

                let outputs = self.runCompute(inputs: inputs)

                var responseData = Data()
                outputs.withUnsafeBufferPointer { ptr in
                    responseData.append(ptr)
                }

                connection.send(content: responseData, completion: .contentProcessed({ _ in
                    connection.cancel()
                }))

                DispatchQueue.main.async {
                    self.log = "Processed \(count) floats on Apple GPU."
                }
            }
        }
    }

    private func runCompute(inputs: [Float]) -> [Float] {
        guard let device = device,
              let commandQueue = commandQueue,
              let pipelineState = pipelineState,
              !inputs.isEmpty else { return [] }

        let count = inputs.count
        let bufferSize = count * MemoryLayout<Float>.size

        guard let inBuffer = device.makeBuffer(bytes: inputs, length: bufferSize, options: .storageModeShared),
              let outBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return []
        }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(inBuffer, offset: 0, index: 0)
        encoder.setBuffer(outBuffer, offset: 0, index: 1)

        let maxThreads = pipelineState.maxTotalThreadsPerThreadgroup
        let threadsPerGroup = MTLSize(width: min(count, maxThreads), height: 1, depth: 1)
        let numGroups = MTLSize(width: (count + threadsPerGroup.width - 1) / threadsPerGroup.width, height: 1, depth: 1)

        encoder.dispatchThreadgroups(numGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let rawPointer = outBuffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: rawPointer, count: count))
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
