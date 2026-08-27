#import <AVFoundation/AVFoundation.h>
#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 4) return 2;
        NSString *videoPath = [[NSString stringWithUTF8String:argv[1]] stringByStandardizingPath];
        NSString *outputDir = [[NSString stringWithUTF8String:argv[2]] stringByStandardizingPath];
        if (![videoPath isAbsolutePath]) videoPath = [[[NSFileManager defaultManager] currentDirectoryPath] stringByAppendingPathComponent:videoPath];
        if (![outputDir isAbsolutePath]) outputDir = [[[NSFileManager defaultManager] currentDirectoryPath] stringByAppendingPathComponent:outputDir];
        [[NSFileManager defaultManager] createDirectoryAtPath:outputDir withIntermediateDirectories:YES attributes:nil error:nil];
        AVURLAsset *asset = [AVURLAsset URLAssetWithURL:[NSURL fileURLWithPath:videoPath] options:nil];
        AVAssetImageGenerator *generator = [[AVAssetImageGenerator alloc] initWithAsset:asset];
        generator.appliesPreferredTrackTransform = YES;
        generator.requestedTimeToleranceBefore = CMTimeMakeWithSeconds(0.05, 600);
        generator.requestedTimeToleranceAfter = CMTimeMakeWithSeconds(0.05, 600);
        for (int index = 3; index < argc; index++) {
            double seconds = atof(argv[index]);
            NSError *error = nil;
            CGImageRef image = [generator copyCGImageAtTime:CMTimeMakeWithSeconds(seconds, 600) actualTime:nil error:&error];
            if (image == nil) {
                fprintf(stderr, "frame %.1f failed: %s\n", seconds, error.localizedDescription.UTF8String);
                return 1;
            }
            NSBitmapImageRep *rep = [[NSBitmapImageRep alloc] initWithCGImage:image];
            NSData *jpeg = [rep representationUsingType:NSBitmapImageFileTypeJPEG properties:@{NSImageCompressionFactor: @0.92}];
            NSString *name = [NSString stringWithFormat:@"verify_%02d_%04.1fs.jpg", index - 2, seconds];
            [jpeg writeToFile:[outputDir stringByAppendingPathComponent:name] atomically:YES];
            CGImageRelease(image);
        }
    }
    return 0;
}
