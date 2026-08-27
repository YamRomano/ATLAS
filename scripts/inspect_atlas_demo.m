#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 2) return 2;
        NSString *path = [[NSString stringWithUTF8String:argv[1]] stringByStandardizingPath];
        if (![path isAbsolutePath]) path = [[[NSFileManager defaultManager] currentDirectoryPath] stringByAppendingPathComponent:path];
        AVURLAsset *asset = [AVURLAsset URLAssetWithURL:[NSURL fileURLWithPath:path] options:nil];
        NSArray<AVAssetTrack *> *video = [asset tracksWithMediaType:AVMediaTypeVideo];
        NSArray<AVAssetTrack *> *audio = [asset tracksWithMediaType:AVMediaTypeAudio];
        double duration = CMTimeGetSeconds(asset.duration);
        CGSize size = video.firstObject.naturalSize;
        float fps = video.firstObject.nominalFrameRate;
        printf("duration=%.3f\nvideo_tracks=%lu\naudio_tracks=%lu\nsize=%.0fx%.0f\nfps=%.3f\n",
            duration, (unsigned long)video.count, (unsigned long)audio.count, size.width, size.height, fps);
    }
    return 0;
}
