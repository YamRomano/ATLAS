import AVFoundation
import Foundation

enum MuxError: Error, CustomStringConvertible {
    case usage
    case missingVideoTrack
    case missingAudioTrack
    case compositionTrack
    case exportSession
    case exportFailed(String)

    var description: String {
        switch self {
        case .usage:
            return "Usage: mux_atlas_demo <silent-video.mp4> <soundtrack.wav> <output.mp4>"
        case .missingVideoTrack:
            return "The input video has no video track."
        case .missingAudioTrack:
            return "The input soundtrack has no audio track."
        case .compositionTrack:
            return "Could not create an AVFoundation composition track."
        case .exportSession:
            return "Could not create an AVFoundation export session."
        case let .exportFailed(message):
            return "Export failed: \(message)"
        }
    }
}

@main
struct AtlasMux {
    static func main() async {
        do {
            guard CommandLine.arguments.count == 4 else { throw MuxError.usage }
            let videoURL = URL(fileURLWithPath: CommandLine.arguments[1])
            let audioURL = URL(fileURLWithPath: CommandLine.arguments[2])
            let outputURL = URL(fileURLWithPath: CommandLine.arguments[3])

            let videoAsset = AVURLAsset(url: videoURL)
            let audioAsset = AVURLAsset(url: audioURL)
            let videoDuration = try await videoAsset.load(.duration)
            guard let sourceVideo = try await videoAsset.loadTracks(withMediaType: .video).first else {
                throw MuxError.missingVideoTrack
            }
            guard let sourceAudio = try await audioAsset.loadTracks(withMediaType: .audio).first else {
                throw MuxError.missingAudioTrack
            }

            let composition = AVMutableComposition()
            guard let videoTrack = composition.addMutableTrack(
                withMediaType: .video,
                preferredTrackID: kCMPersistentTrackID_Invalid
            ) else { throw MuxError.compositionTrack }
            guard let audioTrack = composition.addMutableTrack(
                withMediaType: .audio,
                preferredTrackID: kCMPersistentTrackID_Invalid
            ) else { throw MuxError.compositionTrack }

            let fullRange = CMTimeRange(start: .zero, duration: videoDuration)
            try videoTrack.insertTimeRange(fullRange, of: sourceVideo, at: .zero)
            try audioTrack.insertTimeRange(fullRange, of: sourceAudio, at: .zero)
            videoTrack.preferredTransform = try await sourceVideo.load(.preferredTransform)

            try? FileManager.default.removeItem(at: outputURL)
            guard let exporter = AVAssetExportSession(asset: composition, presetName: AVAssetExportPresetHighestQuality) else {
                throw MuxError.exportSession
            }
            exporter.outputURL = outputURL
            exporter.outputFileType = .mp4
            exporter.shouldOptimizeForNetworkUse = true
            await exporter.export()
            if exporter.status != .completed {
                throw MuxError.exportFailed(exporter.error?.localizedDescription ?? "unknown error")
            }
            print(outputURL.path)
        } catch {
            fputs("\(error)\n", stderr)
            exit(1)
        }
    }
}
