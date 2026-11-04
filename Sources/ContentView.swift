import SwiftUI
import Network
import Metal

final class GPUServer: ObservableObject {

    @Published var status = "Stopped"

    private let queue = DispatchQueue(
        label: "A17.RemoteGPU"
    )

    private var listener: NWListener?

    private let device = MTLCreateSystemDefaultDevice()

    func start() {
        guard let device = device else {
            status = "Metal GPU unavailable"
            return
        }

        do {
            let library = try device.makeDefaultLibrary()

            guard let function = library.makeFunction(
                name: "square_ids"
            ) else {
                status = "Metal shader missing"
                return
            }

            let pipeline = try device.makeComputePipelineState(
                function: function
            )

            guard let commandQueue = device.makeCommandQueue() else {
                status = "Command queue failed"
                return
            }

            let server = try NWListener(
                using: .tcp,
                on: NWEndpoint.Port(rawValue: 8765)!
            )

            listener = server

            server.newConnectionHandler = { [weak self] connection in
                guard let self = self else { return }

                connection.start(queue: self.queue)

                self.receiveRequest(
                    connection,
                    pipeline,
                    commandQueue
                )
            }

            server.stateUpdateHandler = { [weak self] state in
                DispatchQueue.main.async {
                    switch state {
                    case .ready:
                        self?.status = "Listening on TCP 8765"
                    case .failed(let error):
                        self?.status = "Error: \(error)"
                    default:
                        break
                    }
                }
            }

            server.start(queue: queue)

        } catch {
            status = "Error: \(error.localizedDescription)"
        }
    }

    private func receiveRequest(
        _ connection: NWConnection,
        _ pipeline: MTLComputePipelineState,
        _ commandQueue: MTLCommandQueue
    ) {

        connection.receive(
            minimumIncompleteLength: 4,
            maximumLength: 4
        ) { [weak self] data, _, _, error in

            guard let self = self,
                  error == nil,
                  let data = data,
                  data.count == 4 else {
                connection.cancel()
                return
            }

            let count = Int(
                data.withUnsafeBytes {
                    $0.loadUnaligned(as: UInt32.self)
                }.littleEndian
            )

            guard count > 0 && count <= 1_000_000 else {
                connection.cancel()
                return
            }

            guard let output = self.device?.makeBuffer(
                length: count * MemoryLayout<UInt32>.stride,
                options: .storageModeShared
            ),
            let command = commandQueue.makeCommandBuffer(),
            let encoder = command.makeComputeCommandEncoder() else {
                connection.cancel()
                return
            }

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(output, offset: 0, index: 0)

            var gpuCount = UInt32(count)

            encoder.setBytes(
                &gpuCount,
                length: MemoryLayout<UInt32>.stride,
                index: 1
            )

            let threads = 256

            let groups = (
                count + threads - 1
            ) / threads

            encoder.dispatchThreadgroups(
                MTLSize(width: groups, height: 1, depth: 1),
                threadsPerThreadgroup:
                    MTLSize(width: threads, height: 1, depth: 1)
            )

            encoder.endEncoding()

            command.commit()
            command.waitUntilCompleted()

            guard command.status == .completed else {
                connection.cancel()
                return
            }

            let values = output.contents().bindMemory(
                to: UInt32.self,
                capacity: count
            )

            var checksum: UInt64 = 0

            for i in 0..<count {
                checksum &+= UInt64(values[i])
            }

            var result = Data("OKAY".utf8)

            var littleEndianChecksum = checksum.littleEndian

            withUnsafeBytes(of: &littleEndianChecksum) {
                result.append(contentsOf: $0)
            }

            connection.send(
                content: result,
                completion: .contentProcessed { _ in
                    connection.cancel()
                }
            )

            DispatchQueue.main.async {
                self.status = "GPU completed \(count) elements"
            }
        }
    }
}

struct ContentView: View {

    @StateObject private var server = GPUServer()

    var body: some View {
        VStack(spacing: 20) {

            Image(systemName: "cpu")
                .font(.system(size: 55))

            Text("A17 Remote GPU")
                .font(.largeTitle.bold())

            Text("Apple Metal Compute Server")

            Text(server.status)
                .foregroundStyle(.green)

            Button("Start GPU Server") {
                server.start()
            }
            .buttonStyle(.borderedProminent)

            Text("TCP Port: 8765")
                .font(.caption)
        }
        .padding()
    }
}