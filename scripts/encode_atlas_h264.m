#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>

static void Fail(NSString *message) {
    fprintf(stderr, "%s\n", message.UTF8String);
    exit(1);
}

static CVPixelBufferRef PixelBufferFromJPEG(NSString *path, size_t width, size_t height) {
    NSURL *url = [NSURL fileURLWithPath:path];
    CGImageSourceRef source = CGImageSourceCreateWithURL((__bridge CFURLRef)url, NULL);
    if (source == NULL) return NULL;
    CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, NULL);
    CFRelease(source);
    if (image == NULL) return NULL;

    NSDictionary *attributes = @{
        (NSString *)kCVPixelBufferCGImageCompatibilityKey: @YES,
        (NSString *)kCVPixelBufferCGBitmapContextCompatibilityKey: @YES,
        (NSString *)kCVPixelBufferIOSurfacePropertiesKey: @{}
    };
    CVPixelBufferRef buffer = NULL;
    CVReturn result = CVPixelBufferCreate(
        kCFAllocatorDefault,
        width,
        height,
        kCVPixelFormatType_32BGRA,
        (__bridge CFDictionaryRef)attributes,
        &buffer
    );
    if (result != kCVReturnSuccess || buffer == NULL) {
        CGImageRelease(image);
        return NULL;
    }

    CVPixelBufferLockBaseAddress(buffer, 0);
    void *base = CVPixelBufferGetBaseAddress(buffer);
    size_t stride = CVPixelBufferGetBytesPerRow(buffer);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(
        base,
        width,
        height,
        8,
        stride,
        colorSpace,
        kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Little
    );
    CGColorSpaceRelease(colorSpace);
    if (context == NULL) {
        CVPixelBufferUnlockBaseAddress(buffer, 0);
        CVPixelBufferRelease(buffer);
        CGImageRelease(image);
        return NULL;
    }
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), image);
    CGContextRelease(context);
    CVPixelBufferUnlockBaseAddress(buffer, 0);
    CGImageRelease(image);
    return buffer;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 7) {
            Fail(@"Usage: encode_atlas_h264 <frames-dir> <output.mp4> <width> <height> <fps> <frame-count>");
        }
        NSString *framesDir = [[NSString stringWithUTF8String:argv[1]] stringByStandardizingPath];
        NSString *outputPath = [[NSString stringWithUTF8String:argv[2]] stringByStandardizingPath];
        if (![framesDir isAbsolutePath]) framesDir = [[[NSFileManager defaultManager] currentDirectoryPath] stringByAppendingPathComponent:framesDir];
        if (![outputPath isAbsolutePath]) outputPath = [[[NSFileManager defaultManager] currentDirectoryPath] stringByAppendingPathComponent:outputPath];
        int width = atoi(argv[3]);
        int height = atoi(argv[4]);
        int fps = atoi(argv[5]);
        int frameCount = atoi(argv[6]);
        NSURL *outputURL = [NSURL fileURLWithPath:outputPath];
        [[NSFileManager defaultManager] removeItemAtURL:outputURL error:nil];

        NSError *writerError = nil;
        AVAssetWriter *writer = [[AVAssetWriter alloc] initWithURL:outputURL fileType:AVFileTypeMPEG4 error:&writerError];
        if (writer == nil) Fail([NSString stringWithFormat:@"Writer creation failed: %@", writerError.localizedDescription]);
        NSDictionary *settings = @{
            AVVideoCodecKey: AVVideoCodecTypeH264,
            AVVideoWidthKey: @(width),
            AVVideoHeightKey: @(height)
        };
        AVAssetWriterInput *input = [AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo outputSettings:settings];
        input.expectsMediaDataInRealTime = NO;
        NSDictionary *pixelAttributes = @{
            (NSString *)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
            (NSString *)kCVPixelBufferWidthKey: @(width),
            (NSString *)kCVPixelBufferHeightKey: @(height),
            (NSString *)kCVPixelBufferIOSurfacePropertiesKey: @{}
        };
        AVAssetWriterInputPixelBufferAdaptor *adaptor = [AVAssetWriterInputPixelBufferAdaptor
            assetWriterInputPixelBufferAdaptorWithAssetWriterInput:input
            sourcePixelBufferAttributes:pixelAttributes];
        if (![writer canAddInput:input]) Fail(@"Writer cannot add video input.");
        [writer addInput:input];
        if (![writer startWriting]) Fail([NSString stringWithFormat:@"startWriting failed: %@", writer.error.localizedDescription]);
        [writer startSessionAtSourceTime:kCMTimeZero];

        for (int index = 0; index < frameCount; index++) {
            @autoreleasepool {
                while (!input.readyForMoreMediaData && writer.status == AVAssetWriterStatusWriting) usleep(2000);
                if (writer.status != AVAssetWriterStatusWriting) {
                    Fail([NSString stringWithFormat:@"Writer stopped at %d: %@", index, writer.error.localizedDescription]);
                }
                NSString *name = [NSString stringWithFormat:@"frame_%05d.jpg", index];
                CVPixelBufferRef buffer = PixelBufferFromJPEG([framesDir stringByAppendingPathComponent:name], width, height);
                if (buffer == NULL) Fail([NSString stringWithFormat:@"Could not read frame %d", index]);
                if (![adaptor appendPixelBuffer:buffer withPresentationTime:CMTimeMake(index, fps)]) {
                    CVPixelBufferRelease(buffer);
                    Fail([NSString stringWithFormat:@"Append failed at %d: %@", index, writer.error.localizedDescription]);
                }
                CVPixelBufferRelease(buffer);
            }
            if (index % 240 == 0) printf("encoded %d\n", index);
        }
        [input markAsFinished];
        dispatch_semaphore_t finished = dispatch_semaphore_create(0);
        [writer finishWritingWithCompletionHandler:^{ dispatch_semaphore_signal(finished); }];
        dispatch_semaphore_wait(finished, DISPATCH_TIME_FOREVER);
        if (writer.status != AVAssetWriterStatusCompleted) {
            Fail([NSString stringWithFormat:@"Finish failed: %@", writer.error.localizedDescription]);
        }
        printf("%s\n", outputPath.UTF8String);
    }
    return 0;
}
