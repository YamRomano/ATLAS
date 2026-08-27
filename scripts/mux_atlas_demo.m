#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>

static void Fail(NSString *message) {
    fprintf(stderr, "%s\n", message.UTF8String);
    exit(1);
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 4) {
            Fail(@"Usage: mux_atlas_demo <silent-video.mp4> <soundtrack.wav> <output.mp4>");
        }

        NSString *videoPath = [[NSString stringWithUTF8String:argv[1]] stringByStandardizingPath];
        NSString *audioPath = [[NSString stringWithUTF8String:argv[2]] stringByStandardizingPath];
        NSString *outputPath = [[NSString stringWithUTF8String:argv[3]] stringByStandardizingPath];
        if (![videoPath isAbsolutePath]) videoPath = [[[NSFileManager defaultManager] currentDirectoryPath] stringByAppendingPathComponent:videoPath];
        if (![audioPath isAbsolutePath]) audioPath = [[[NSFileManager defaultManager] currentDirectoryPath] stringByAppendingPathComponent:audioPath];
        if (![outputPath isAbsolutePath]) outputPath = [[[NSFileManager defaultManager] currentDirectoryPath] stringByAppendingPathComponent:outputPath];
        NSURL *videoURL = [NSURL fileURLWithPath:videoPath];
        NSURL *audioURL = [NSURL fileURLWithPath:audioPath];
        NSURL *outputURL = [NSURL fileURLWithPath:outputPath];
        AVURLAsset *videoAsset = [AVURLAsset URLAssetWithURL:videoURL options:nil];
        AVURLAsset *audioAsset = [AVURLAsset URLAssetWithURL:audioURL options:nil];
        AVAssetTrack *sourceVideo = [videoAsset tracksWithMediaType:AVMediaTypeVideo].firstObject;
        AVAssetTrack *sourceAudio = [audioAsset tracksWithMediaType:AVMediaTypeAudio].firstObject;
        if (sourceVideo == nil) Fail(@"The input video has no video track.");
        if (sourceAudio == nil) Fail(@"The soundtrack has no audio track.");

        AVMutableComposition *composition = [AVMutableComposition composition];
        AVMutableCompositionTrack *videoTrack = [composition addMutableTrackWithMediaType:AVMediaTypeVideo preferredTrackID:kCMPersistentTrackID_Invalid];
        AVMutableCompositionTrack *audioTrack = [composition addMutableTrackWithMediaType:AVMediaTypeAudio preferredTrackID:kCMPersistentTrackID_Invalid];
        if (videoTrack == nil || audioTrack == nil) Fail(@"Could not create composition tracks.");

        NSError *insertError = nil;
        CMTimeRange range = CMTimeRangeMake(kCMTimeZero, videoAsset.duration);
        [videoTrack insertTimeRange:range ofTrack:sourceVideo atTime:kCMTimeZero error:&insertError];
        if (insertError != nil) Fail([NSString stringWithFormat:@"Video insert failed: %@", insertError.localizedDescription]);
        [audioTrack insertTimeRange:range ofTrack:sourceAudio atTime:kCMTimeZero error:&insertError];
        if (insertError != nil) Fail([NSString stringWithFormat:@"Audio insert failed: %@", insertError.localizedDescription]);
        videoTrack.preferredTransform = sourceVideo.preferredTransform;

        [[NSFileManager defaultManager] removeItemAtURL:outputURL error:nil];
        AVAssetExportSession *exporter = [[AVAssetExportSession alloc] initWithAsset:composition presetName:AVAssetExportPresetPassthrough];
        if (exporter == nil) Fail(@"Could not create export session.");
        exporter.outputURL = outputURL;
        exporter.outputFileType = [exporter.supportedFileTypes containsObject:AVFileTypeMPEG4] ? AVFileTypeMPEG4 : exporter.supportedFileTypes.firstObject;
        exporter.shouldOptimizeForNetworkUse = YES;

        dispatch_semaphore_t finished = dispatch_semaphore_create(0);
        [exporter exportAsynchronouslyWithCompletionHandler:^{
            dispatch_semaphore_signal(finished);
        }];
        dispatch_semaphore_wait(finished, DISPATCH_TIME_FOREVER);
        if (exporter.status != AVAssetExportSessionStatusCompleted) {
            NSError *error = exporter.error;
            Fail([NSString stringWithFormat:@"Export failed: %@ | domain=%@ code=%ld | %@",
                error.localizedDescription ?: @"unknown error",
                error.domain ?: @"none",
                (long)error.code,
                error.userInfo ?: @{}]);
        }
        printf("%s\n", outputURL.path.UTF8String);
    }
    return 0;
}
