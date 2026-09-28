#import <AVFoundation/AVFoundation.h>
#import <CoreImage/CoreImage.h>
#import <MediaPipeTasksVision/MediaPipeTasksVision.h>
#import "UnityInterface.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

// The Unity scene consumes the same version-4 packet as My project (2)'s Python tracker.
// All three Tasks receive the same 256-pixel letterboxed camera frame.
static NSString *const kReceiver = @"Main Camera";
static AVCaptureSession *s_session;
static AVCaptureVideoDataOutput *s_output;
static dispatch_queue_t s_queue;
static CIContext *s_context;
static MPPPoseLandmarker *s_pose;
static MPPHandLandmarker *s_hands;
static MPPFaceLandmarker *s_face;
static NSUInteger s_frame;
static NSInteger s_lastTimestamp;
static BOOL s_loggedHuman;

static void PoseLog(NSString *message) {
    NSString *line = [NSString stringWithFormat:@"%@ %@\n", NSDate.date, message];
    NSString *documents = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject;
    NSString *path = [documents stringByAppendingPathComponent:@"native-pose-debug.log"];
    @synchronized(kReceiver) {
        FILE *file = fopen(path.fileSystemRepresentation, "ab");
        if (file) {
            const char *bytes = line.UTF8String;
            fwrite(bytes, 1, strlen(bytes), file);
            fflush(file);
            fsync(fileno(file));
            fclose(file);
        }
    }
    NSLog(@"[CodexMediaPipe] %@", message);
    dispatch_async(dispatch_get_main_queue(), ^{
        UnitySendMessage(kReceiver.UTF8String, "OnNativePoseLog", message.UTF8String);
    });
}

static NSDictionary *HeadRotation(MPPTransformMatrix *matrix) {
    if (!matrix || matrix.rows < 3 || matrix.columns < 3) return nil;
    float r[3][3];
    for (NSUInteger i = 0; i < 3; ++i)
        for (NSUInteger j = 0; j < 3; ++j)
            r[i][j] = [matrix valueAtRow:i column:j];
    // Polar decomposition removes the face matrix scale, as the Windows SVD does.
    for (NSUInteger n = 0; n < 6; ++n) {
        float d = r[0][0]*(r[1][1]*r[2][2]-r[1][2]*r[2][1])
                - r[0][1]*(r[1][0]*r[2][2]-r[1][2]*r[2][0])
                + r[0][2]*(r[1][0]*r[2][1]-r[1][1]*r[2][0]);
        if (fabsf(d) < 0.000001f) return nil;
        float t[3][3] = {
            {(r[1][1]*r[2][2]-r[1][2]*r[2][1])/d, (r[1][2]*r[2][0]-r[1][0]*r[2][2])/d, (r[1][0]*r[2][1]-r[1][1]*r[2][0])/d},
            {(r[0][2]*r[2][1]-r[0][1]*r[2][2])/d, (r[0][0]*r[2][2]-r[0][2]*r[2][0])/d, (r[0][1]*r[2][0]-r[0][0]*r[2][1])/d},
            {(r[0][1]*r[1][2]-r[0][2]*r[1][1])/d, (r[0][2]*r[2][1]-r[0][1]*r[2][0])/d, (r[0][0]*r[1][1]-r[0][1]*r[1][0])/d}
        };
        for (NSUInteger i = 0; i < 3; ++i)
            for (NSUInteger j = 0; j < 3; ++j)
                r[i][j] = 0.5f * (r[i][j] + t[i][j]);
    }
    const float axis[3] = {1, -1, -1};
    for (NSUInteger i = 0; i < 3; ++i)
        for (NSUInteger j = 0; j < 3; ++j)
            r[i][j] *= axis[i] * axis[j];
    float x, y, z, w, q;
    float trace = r[0][0] + r[1][1] + r[2][2];
    if (trace > 0) {
        q = sqrtf(trace + 1) * 2;
        w = .25f*q; x = (r[2][1]-r[1][2])/q; y = (r[0][2]-r[2][0])/q; z = (r[1][0]-r[0][1])/q;
    } else if (r[0][0] > r[1][1] && r[0][0] > r[2][2]) {
        q = sqrtf(1+r[0][0]-r[1][1]-r[2][2])*2;
        x=.25f*q; y=(r[0][1]+r[1][0])/q; z=(r[0][2]+r[2][0])/q; w=(r[2][1]-r[1][2])/q;
    } else if (r[1][1] > r[2][2]) {
        q = sqrtf(1+r[1][1]-r[0][0]-r[2][2])*2;
        x=(r[0][1]+r[1][0])/q; y=.25f*q; z=(r[1][2]+r[2][1])/q; w=(r[0][2]-r[2][0])/q;
    } else {
        q = sqrtf(1+r[2][2]-r[0][0]-r[1][1])*2;
        x=(r[0][2]+r[2][0])/q; y=(r[1][2]+r[2][1])/q; z=.25f*q; w=(r[1][0]-r[0][1])/q;
    }
    float norm = sqrtf(x*x+y*y+z*z+w*w);
    return norm > .00001f ? @{@"x":@(x/norm), @"y":@(y/norm), @"z":@(z/norm), @"w":@(w/norm)} : nil;
}

static NSNumber *Number(float value) { return @(isfinite(value) ? value : 0.0f); }
static NSDictionary *PosePointJSON(NSString *name, MPPLandmark *world, MPPNormalizedLandmark *image,
                           float confidence, float width, float height, float scale, float left, float top) {
    float ix = (image.x*256.0f-left)/(scale*width);
    float iy = (image.y*256.0f-top)/(scale*height);
    return @{@"name":name, @"x":Number(world.x), @"y":Number(world.y), @"z":Number(world.z),
             @"confidence":Number(confidence), @"image_x":Number(ix), @"image_y":Number(iy),
             @"image_z":Number(image.z)};
}

@interface CodexCaptureDelegate : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>
@end
@implementation CodexCaptureDelegate
- (void)captureOutput:(AVCaptureOutput *)output didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection {
    if (!s_pose || !s_hands || !s_face) return;
    CVPixelBufferRef camera = CMSampleBufferGetImageBuffer(sampleBuffer);
    if (!camera) return;
    CMTime time = CMSampleBufferGetPresentationTimeStamp(sampleBuffer);
    NSInteger timestamp = (NSInteger)llround(CMTimeGetSeconds(time)*1000.0);
    if (timestamp <= s_lastTimestamp) timestamp = s_lastTimestamp + 1;
    // Windows tracks at 20 Hz; AVFoundation drops late frames while inference runs.
    if (timestamp - s_lastTimestamp < 50) return;
    s_lastTimestamp = timestamp;
    size_t width = CVPixelBufferGetWidth(camera), height = CVPixelBufferGetHeight(camera);
    float scale = fminf(256.0f/(float)width, 256.0f/(float)height);
    float scaledWidth = roundf(width*scale), scaledHeight = roundf(height*scale);
    float left = floorf((256.0f-scaledWidth)/2), top = floorf((256.0f-scaledHeight)/2);
    CVPixelBufferRef square = NULL;
    NSDictionary *attributes = @{(id)kCVPixelBufferIOSurfacePropertiesKey:@{}};
    if (CVPixelBufferCreate(kCFAllocatorDefault, 256, 256, kCVPixelFormatType_32BGRA,
                            (__bridge CFDictionaryRef)attributes, &square) != kCVReturnSuccess) return;
    if (!s_context) s_context = [CIContext contextWithOptions:nil];
    CIImage *source = [CIImage imageWithCVPixelBuffer:camera];
    CIImage *resized = [source imageByApplyingTransform:CGAffineTransformMakeScale(scaledWidth/(float)width, scaledHeight/(float)height)];
    resized = [resized imageByApplyingTransform:CGAffineTransformMakeTranslation(left, top)];
    CIImage *black = [CIImage imageWithColor:CIColor.blackColor];
    CIImage *letterboxed = [resized imageByCompositingOverImage:[black imageByCroppingToRect:CGRectMake(0, 0, 256, 256)]];
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    [s_context render:letterboxed toCVPixelBuffer:square bounds:CGRectMake(0, 0, 256, 256)
          colorSpace:colorSpace];
    CGColorSpaceRelease(colorSpace);
    NSError *error = nil;
    MPPImage *image = [[MPPImage alloc] initWithPixelBuffer:square error:&error];
    CVPixelBufferRelease(square);
    if (!image) { PoseLog([NSString stringWithFormat:@"MPPImage: %@", error]); return; }
    MPPPoseLandmarkerResult *pose = [s_pose detectVideoFrame:image timestampInMilliseconds:timestamp error:&error];
    if (!pose) { PoseLog([NSString stringWithFormat:@"pose: %@", error]); return; }
    MPPHandLandmarkerResult *hands = [s_hands detectVideoFrame:image timestampInMilliseconds:timestamp error:&error];
    if (!hands) { PoseLog([NSString stringWithFormat:@"hands: %@", error]); return; }
    MPPFaceLandmarkerResult *face = [s_face detectVideoFrame:image timestampInMilliseconds:timestamp error:&error];
    if (!face) { PoseLog([NSString stringWithFormat:@"face: %@", error]); return; }

    static NSArray<NSString *> *poseNames;
    static NSArray<NSNumber *> *poseIndices;
    static NSArray<NSString *> *handNames;
    static dispatch_once_t namesOnce;
    dispatch_once(&namesOnce, ^{
        poseNames = @[@"nose",@"left_eye",@"right_eye",@"left_ear",@"right_ear",@"left_shoulder",@"right_shoulder",
            @"left_elbow",@"right_elbow",@"left_wrist",@"right_wrist",@"left_pinky",@"right_pinky",
            @"left_index",@"right_index",@"left_thumb",@"right_thumb",@"left_hip",@"right_hip",
            @"left_knee",@"right_knee",@"left_ankle",@"right_ankle",@"left_heel",@"right_heel",
            @"left_foot_index",@"right_foot_index"];
        poseIndices = @[@0,@2,@5,@7,@8,@11,@12,@13,@14,@15,@16,@17,@18,@19,@20,@21,@22,@23,@24,@25,@26,@27,@28,@29,@30,@31,@32];
        handNames = @[@"wrist",@"thumb_cmc",@"thumb_mcp",@"thumb_ip",@"thumb",
            @"index_mcp",@"index_pip",@"index_dip",@"index",@"middle_mcp",@"middle_pip",
            @"middle_dip",@"middle",@"ring_mcp",@"ring_pip",@"ring_dip",@"ring",
            @"pinky_mcp",@"pinky_pip",@"pinky_dip",@"pinky"];
    });
    NSArray<MPPNormalizedLandmark *> *poseImage = pose.landmarks.firstObject;
    NSArray<MPPLandmark *> *poseWorld = pose.worldLandmarks.firstObject;
    NSMutableArray *points = [NSMutableArray array];
    for (NSUInteger i=0; i<poseNames.count; ++i) {
        NSUInteger index = poseIndices[i].unsignedIntegerValue;
        if (index >= poseImage.count || index >= poseWorld.count) continue;
        MPPNormalizedLandmark *p = poseImage[index];
        [points addObject:PosePointJSON(poseNames[i], poseWorld[index], p, p.visibility.floatValue,
                                width, height, scale, left, top)];
    }
    // Associate detected hands with pose wrists; maintain one result per side.
    NSMutableSet<NSString *> *assigned = [NSMutableSet set];
    for (NSUInteger h=0; h<hands.landmarks.count && h<2; ++h) {
        NSArray<MPPNormalizedLandmark *> *handImage = hands.landmarks[h];
        NSArray<MPPLandmark *> *handWorld = h<hands.worldLandmarks.count ? hands.worldLandmarks[h] : nil;
        if (handImage.count<21 || handWorld.count<21) continue;
        MPPNormalizedLandmark *wrist = handImage[0];
        MPPCategory *category = h<hands.handedness.count ? [hands.handedness[h] firstObject] : nil;
        NSString *side = category.categoryName.lowercaseString;
        if (poseImage.count>16 && poseImage[15].visibility.floatValue>=.2f &&
            poseImage[16].visibility.floatValue>=.2f) {
            float ld=hypotf(wrist.x-poseImage[15].x,wrist.y-poseImage[15].y);
            float rd=hypotf(wrist.x-poseImage[16].x,wrist.y-poseImage[16].y);
            side=ld<=rd ? @"left" : @"right";
        }
        if (![side isEqualToString:@"left"] && ![side isEqualToString:@"right"])
            side=wrist.x<.5f ? @"left" : @"right";
        if ([assigned containsObject:side]) side=[side isEqualToString:@"left"] ? @"right" : @"left";
        [assigned addObject:side];
        float score=category ? category.score : .5f;
        for (NSUInteger i=0; i<21; ++i)
            [points addObject:PosePointJSON([NSString stringWithFormat:@"%@_hand_%@",side,handNames[i]],
                                    handWorld[i],handImage[i],score,width,height,scale,left,top)];
        float wx=0,wy=0,wz=0,ix=0,iy=0,iz=0;
        for (NSNumber *n in @[@5,@9,@17]) {
            NSUInteger i=n.unsignedIntegerValue;
            wx+=handWorld[i].x; wy+=handWorld[i].y; wz+=handWorld[i].z;
            ix+=handImage[i].x; iy+=handImage[i].y; iz+=handImage[i].z;
        }
        [points addObject:@{@"name":[NSString stringWithFormat:@"%@_hand_palm",side],
            @"x":Number(wx/3),@"y":Number(wy/3),@"z":Number(wz/3),@"confidence":Number(score),
            @"image_x":Number((ix/3*256-left)/(scale*width)),
            @"image_y":Number((iy/3*256-top)/(scale*height)),@"image_z":Number(iz/3)}];
    }
    NSMutableArray *shapes=[NSMutableArray array];
    NSSet *wanted=[NSSet setWithArray:@[@"eyeBlinkLeft",@"eyeBlinkRight",@"jawOpen",@"mouthSmileLeft",@"mouthSmileRight"]];
    for (MPPCategory *category in face.faceBlendshapes.firstObject.categories)
        if ([wanted containsObject:category.categoryName])
            [shapes addObject:@{@"name":category.categoryName,@"score":Number(category.score)}];
    NSMutableDictionary *packet=[@{@"version":@4,@"frame":@(++s_frame),
        @"timestamp_ms":@(timestamp),@"source_width":@(width),@"source_height":@(height),
        @"tracking":@(points.count>0),@"face_blendshapes":shapes,@"points":points} mutableCopy];
    NSDictionary *rotation=HeadRotation(face.facialTransformationMatrixes.firstObject);
    if (rotation) packet[@"head_rotation"]=rotation;
    NSData *json=[NSJSONSerialization dataWithJSONObject:packet options:0 error:&error];
    if (!json) { PoseLog([NSString stringWithFormat:@"packet: %@",error]); return; }
    NSString *body=[[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding];
    if (!s_loggedHuman && points.count) { s_loggedHuman=YES; PoseLog(@"first MediaPipe body detected"); }
    if (s_frame%120==0) PoseLog([NSString stringWithFormat:@"MediaPipe frame %lu, points %lu",
                                  (unsigned long)s_frame,(unsigned long)points.count]);
    dispatch_async(dispatch_get_main_queue(), ^{
        UnitySendMessage(kReceiver.UTF8String,"OnNativePoseJson",body.UTF8String);
    });
}
@end

static CodexCaptureDelegate *s_delegate;
static void StartAuthorized(void) {
    if (s_session) return;
    NSString *raw=[NSBundle mainBundle].bundlePath;
    NSString *models=[raw stringByAppendingPathComponent:@"Data/Raw"];
    NSString *posePath=[models stringByAppendingPathComponent:@"pose_landmarker_lite.task"];
    NSString *handPath=[models stringByAppendingPathComponent:@"hand_landmarker.task"];
    NSString *facePath=[models stringByAppendingPathComponent:@"face_landmarker.task"];
    NSFileManager *files=NSFileManager.defaultManager;
    if (![files fileExistsAtPath:posePath] || ![files fileExistsAtPath:handPath] ||
        ![files fileExistsAtPath:facePath]) { PoseLog(@"MediaPipe model missing from Data/Raw"); return; }
    NSError *error=nil;
    MPPPoseLandmarkerOptions *po=[MPPPoseLandmarkerOptions new];
    po.baseOptions.modelAssetPath=posePath; po.runningMode=MPPRunningModeVideo;
    po.numPoses=1; po.shouldOutputSegmentationMasks=NO;
    s_pose=[[MPPPoseLandmarker alloc] initWithOptions:po error:&error];
    if (!s_pose) { PoseLog([NSString stringWithFormat:@"pose init: %@",error]); return; }
    MPPHandLandmarkerOptions *ho=[MPPHandLandmarkerOptions new];
    ho.baseOptions.modelAssetPath=handPath; ho.runningMode=MPPRunningModeVideo;
    ho.numHands=2; ho.minHandDetectionConfidence=.35f;
    ho.minHandPresenceConfidence=.35f; ho.minTrackingConfidence=.35f;
    s_hands=[[MPPHandLandmarker alloc] initWithOptions:ho error:&error];
    if (!s_hands) { PoseLog([NSString stringWithFormat:@"hand init: %@",error]); s_pose=nil; return; }
    MPPFaceLandmarkerOptions *fo=[MPPFaceLandmarkerOptions new];
    fo.baseOptions.modelAssetPath=facePath; fo.runningMode=MPPRunningModeVideo;
    fo.numFaces=1; fo.outputFaceBlendshapes=YES; fo.outputFacialTransformationMatrixes=YES;
    s_face=[[MPPFaceLandmarker alloc] initWithOptions:fo error:&error];
    if (!s_face) { PoseLog([NSString stringWithFormat:@"face init: %@",error]); s_pose=nil; s_hands=nil; return; }
    AVCaptureDevice *camera=[AVCaptureDevice defaultDeviceWithDeviceType:AVCaptureDeviceTypeBuiltInWideAngleCamera
                                                                  mediaType:AVMediaTypeVideo position:AVCaptureDevicePositionBack];
    AVCaptureDeviceInput *input=[AVCaptureDeviceInput deviceInputWithDevice:camera error:&error];
    if (!input) { PoseLog([NSString stringWithFormat:@"camera input: %@",error]); return; }
    s_session=[AVCaptureSession new]; [s_session beginConfiguration];
    if ([s_session canSetSessionPreset:AVCaptureSessionPreset640x480])
        s_session.sessionPreset=AVCaptureSessionPreset640x480;
    if ([s_session canAddInput:input]) [s_session addInput:input];
    s_output=[AVCaptureVideoDataOutput new];
    s_output.videoSettings=@{(id)kCVPixelBufferPixelFormatTypeKey:@(kCVPixelFormatType_32BGRA)};
    s_output.alwaysDiscardsLateVideoFrames=YES;
    s_queue=dispatch_queue_create("jp.codex.mediapipe.camera",DISPATCH_QUEUE_SERIAL);
    s_delegate=[CodexCaptureDelegate new];
    [s_output setSampleBufferDelegate:s_delegate queue:s_queue];
    if ([s_session canAddOutput:s_output]) [s_session addOutput:s_output];
    AVCaptureConnection *connection=[s_output connectionWithMediaType:AVMediaTypeVideo];
    if (connection.isVideoOrientationSupported) connection.videoOrientation=AVCaptureVideoOrientationPortrait;
    if (connection.isVideoMirroringSupported) {
        connection.automaticallyAdjustsVideoMirroring=NO; connection.videoMirrored=NO;
    }
    [s_session commitConfiguration];
    s_frame=0; s_lastTimestamp=0; s_loggedHuman=NO;
    [s_session startRunning];
    PoseLog(@"MediaPipe Pose/Hand/Face camera started");
}

extern "C" void CodexRearCameraStart(void) {
    AVAuthorizationStatus status=[AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo];
    if (status==AVAuthorizationStatusNotDetermined) {
        [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo completionHandler:^(BOOL granted) {
            if (granted) dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0), ^{ StartAuthorized(); });
            else PoseLog(@"camera permission denied");
        }];
    } else if (status==AVAuthorizationStatusAuthorized) {
        dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0), ^{ StartAuthorized(); });
    } else PoseLog(@"camera permission denied");
}

extern "C" void CodexRearCameraStop(void) {
    [s_session stopRunning];
    [s_output setSampleBufferDelegate:nil queue:NULL];
    s_session=nil; s_output=nil; s_delegate=nil;
    s_pose=nil; s_hands=nil; s_face=nil; s_context=nil;
}
