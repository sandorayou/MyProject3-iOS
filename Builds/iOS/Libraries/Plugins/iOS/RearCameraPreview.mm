#import <AVFoundation/AVFoundation.h>
#import <UIKit/UIKit.h>
#import <Vision/Vision.h>
#import <ImageIO/ImageIO.h>
#import <simd/simd.h>
#import "UnityInterface.h"

static AVCaptureSession *s_rearCameraSession;
static AVCaptureVideoDataOutput *s_videoOutput;
static AVCaptureDepthDataOutput *s_depthOutput;
static AVCaptureDataOutputSynchronizer *s_outputSynchronizer;
static dispatch_queue_t s_captureQueue;
static VNDetectHumanBodyPoseRequest *s_bodyPoseRequest;
static VNDetectHumanBodyPose3DRequest *s_bodyPose3DRequest;
static VNDetectHumanHandPoseRequest *s_handPoseRequest;
static VNDetectFaceLandmarksRequest *s_faceRequest;
static uint64_t s_poseFrame;
static CMTime s_lastInferenceTime;
static BOOL s_loggedDepthFrame;
static NSString *const s_unityReceiverName = @"Main Camera";

// Vision sees the video rotated right into portrait coordinates. The LiDAR map
// shares the unrotated video buffer's field of view, so undo that rotation.
static CGFloat CodexDepthAtVisionPoint(CVPixelBufferRef map, CGPoint visionPoint) {
    if (map == nil) return NAN;
    size_t width = CVPixelBufferGetWidth(map);
    size_t height = CVPixelBufferGetHeight(map);
    CGFloat rawX = 1.0 - visionPoint.y;
    CGFloat rawY = 1.0 - visionPoint.x;
    NSInteger cx = (NSInteger)floor(rawX * (CGFloat)width);
    NSInteger cy = (NSInteger)floor(rawY * (CGFloat)height);
    float samples[9];
    int count = 0;
    if (cx >= 0 && cy >= 0 && cx < (NSInteger)width && cy < (NSInteger)height) {
        const uint8_t *base = (const uint8_t *)CVPixelBufferGetBaseAddress(map);
        size_t stride = CVPixelBufferGetBytesPerRow(map);
        for (NSInteger dy = -1; dy <= 1; dy++) {
            for (NSInteger dx = -1; dx <= 1; dx++) {
                NSInteger x = cx + dx, y = cy + dy;
                if (x < 0 || y < 0 || x >= (NSInteger)width || y >= (NSInteger)height) continue;
                float value = *(const float *)(base + y * stride + x * sizeof(float));
                if (isfinite(value) && value >= 0.15f && value <= 8.0f) samples[count++] = value;
            }
        }
    }
    if (count == 0) return NAN;
    for (int i = 1; i < count; i++) {
        float value = samples[i];
        int j = i - 1;
        while (j >= 0 && samples[j] > value) { samples[j + 1] = samples[j]; j--; }
        samples[j + 1] = value;
    }
    return samples[count / 2];
}

@interface CodexRearPoseDelegate : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate, AVCaptureDataOutputSynchronizerDelegate>
@end

@implementation CodexRearPoseDelegate
- (void)captureOutput:(AVCaptureOutput *)output
 didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
        fromConnection:(AVCaptureConnection *)connection {
    [self processSampleBuffer:sampleBuffer depthData:nil];
}

- (void)dataOutputSynchronizer:(AVCaptureDataOutputSynchronizer *)synchronizer
    didOutputSynchronizedDataCollection:(AVCaptureSynchronizedDataCollection *)collection {
    AVCaptureSynchronizedSampleBufferData *video =
        (AVCaptureSynchronizedSampleBufferData *)[collection synchronizedDataForCaptureOutput:s_videoOutput];
    if (video == nil || video.sampleBufferWasDropped) return;
    AVCaptureSynchronizedDepthData *depth =
        (AVCaptureSynchronizedDepthData *)[collection synchronizedDataForCaptureOutput:s_depthOutput];
    [self processSampleBuffer:video.sampleBuffer depthData:(depth != nil && !depth.depthDataWasDropped) ? depth.depthData : nil];
}

- (void)processSampleBuffer:(CMSampleBufferRef)sampleBuffer depthData:(AVDepthData *)depthData {
    CMTime timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer);
    if (CMTIME_IS_VALID(s_lastInferenceTime) && CMTimeGetSeconds(CMTimeSubtract(timestamp, s_lastInferenceTime)) < (1.0 / 15.0)) return;
    s_lastInferenceTime = timestamp;

    VNImageRequestHandler *handler = [[VNImageRequestHandler alloc]
        initWithCMSampleBuffer:sampleBuffer
                   orientation:kCGImagePropertyOrientationRight
                       options:@{}];
    NSError *error = nil;
    if (![handler performRequests:@[s_bodyPoseRequest, s_handPoseRequest] error:&error]) return;
    // Optional requests must not suppress a valid 2D body frame when a face is occluded.
    [handler performRequests:@[s_bodyPose3DRequest, s_faceRequest] error:nil];
    VNHumanBodyPoseObservation *observation = s_bodyPoseRequest.results.firstObject;
    if (observation == nil) return;
    VNHumanBodyPose3DObservation *body3D = s_bodyPose3DRequest.results.firstObject;
    AVDepthData *metricDepth = depthData != nil
        ? [depthData depthDataByConvertingToDepthDataType:kCVPixelFormatType_DepthFloat32] : nil;
    CVPixelBufferRef depthMap = metricDepth.depthDataMap;
    if (depthMap != nil && CVPixelBufferLockBaseAddress(depthMap, kCVPixelBufferLock_ReadOnly) != kCVReturnSuccess)
        depthMap = nil;
    if (depthMap != nil && !s_loggedDepthFrame) {
        NSLog(@"[CodexRearPose] Synchronized LiDAR depth received");
        s_loggedDepthFrame = YES;
    }

    NSDictionary<NSString *, NSString *> *jointNames = @{
        @"nose": VNHumanBodyPoseObservationJointNameNose,
        @"left_eye": VNHumanBodyPoseObservationJointNameLeftEye,
        @"right_eye": VNHumanBodyPoseObservationJointNameRightEye,
        @"left_ear": VNHumanBodyPoseObservationJointNameLeftEar,
        @"right_ear": VNHumanBodyPoseObservationJointNameRightEar,
        @"left_shoulder": VNHumanBodyPoseObservationJointNameLeftShoulder,
        @"right_shoulder": VNHumanBodyPoseObservationJointNameRightShoulder,
        @"left_elbow": VNHumanBodyPoseObservationJointNameLeftElbow,
        @"right_elbow": VNHumanBodyPoseObservationJointNameRightElbow,
        @"left_wrist": VNHumanBodyPoseObservationJointNameLeftWrist,
        @"right_wrist": VNHumanBodyPoseObservationJointNameRightWrist,
        @"left_hip": VNHumanBodyPoseObservationJointNameLeftHip,
        @"right_hip": VNHumanBodyPoseObservationJointNameRightHip,
        @"left_knee": VNHumanBodyPoseObservationJointNameLeftKnee,
        @"right_knee": VNHumanBodyPoseObservationJointNameRightKnee,
        @"left_ankle": VNHumanBodyPoseObservationJointNameLeftAnkle,
        @"right_ankle": VNHumanBodyPoseObservationJointNameRightAnkle,
        @"neck": VNHumanBodyPoseObservationJointNameNeck,
        @"root": VNHumanBodyPoseObservationJointNameRoot,
    };

    VNRecognizedPoint *leftShoulder = [observation recognizedPointForJointName:VNHumanBodyPoseObservationJointNameLeftShoulder error:nil];
    VNRecognizedPoint *rightShoulder = [observation recognizedPointForJointName:VNHumanBodyPoseObservationJointNameRightShoulder error:nil];
    CGFloat shoulderWidth = (leftShoulder.confidence > 0.1 && rightShoulder.confidence > 0.1)
        ? fabs(leftShoulder.location.x - rightShoulder.location.x) : 0.25;
    CGFloat metersPerImageUnit = 0.4 / MAX(shoulderWidth, 0.08);
    BOOL hasShoulders = leftShoulder.confidence > 0.1 && rightShoulder.confidence > 0.1;
    CGFloat centerX = hasShoulders ? (leftShoulder.location.x + rightShoulder.location.x) * 0.5 : 0.5;
    CGFloat centerY = hasShoulders ? (leftShoulder.location.y + rightShoulder.location.y) * 0.5 : 0.5;
    NSDictionary<NSString *, VNHumanBodyPose3DObservationJointName> *depthJoints = @{
        @"left_shoulder": VNHumanBodyPose3DObservationJointNameLeftShoulder,
        @"right_shoulder": VNHumanBodyPose3DObservationJointNameRightShoulder,
        @"left_elbow": VNHumanBodyPose3DObservationJointNameLeftElbow,
        @"right_elbow": VNHumanBodyPose3DObservationJointNameRightElbow,
        @"left_wrist": VNHumanBodyPose3DObservationJointNameLeftWrist,
        @"right_wrist": VNHumanBodyPose3DObservationJointNameRightWrist,
        @"left_hip": VNHumanBodyPose3DObservationJointNameLeftHip,
        @"right_hip": VNHumanBodyPose3DObservationJointNameRightHip,
        @"left_knee": VNHumanBodyPose3DObservationJointNameLeftKnee,
        @"right_knee": VNHumanBodyPose3DObservationJointNameRightKnee,
        @"left_ankle": VNHumanBodyPose3DObservationJointNameLeftAnkle,
        @"right_ankle": VNHumanBodyPose3DObservationJointNameRightAnkle,
        @"neck": VNHumanBodyPose3DObservationJointNameCenterShoulder,
        @"nose": VNHumanBodyPose3DObservationJointNameCenterHead,
    };
    NSMutableDictionary<NSString *, NSNumber *> *jointDepth = [NSMutableDictionary dictionary];
    if (body3D != nil) {
        for (NSString *name in depthJoints) {
            simd_float4x4 transform;
            if ([body3D getCameraRelativePosition:&transform forJointName:depthJoints[name] error:nil]) {
                jointDepth[name] = @(fabsf(transform.columns[3].z));
            }
        }
    }
    CGFloat centerDepth = (jointDepth[@"left_shoulder"] && jointDepth[@"right_shoulder"])
        ? (jointDepth[@"left_shoulder"].doubleValue + jointDepth[@"right_shoulder"].doubleValue) * 0.5 : 0;

    NSMutableArray *points = [NSMutableArray arrayWithCapacity:jointNames.count + 42];
    for (NSString *name in jointNames) {
        VNRecognizedPoint *point = [observation recognizedPointForJointName:jointNames[name] error:nil];
        if (point == nil || point.confidence <= 0.01) continue;
        CGFloat imageY = 1.0 - point.location.y;
        [points addObject:@{
            @"name": name,
            @"x": @((point.location.x - centerX) * metersPerImageUnit),
            @"y": @((centerY - point.location.y) * metersPerImageUnit),
            @"z": @(jointDepth[name] ? jointDepth[name].doubleValue - centerDepth : 0),
            @"confidence": @(point.confidence),
            @"image_x": @(point.location.x),
            @"image_y": @(imageY),
            @"image_z": @(jointDepth[name] ? jointDepth[name].doubleValue - centerDepth : 0),
        }];
    }

    // Match each hand to the body wrists. This follows the Windows tracker's
    // spatial left/right assignment and avoids depending on model handedness labels.
    VNRecognizedPoint *bodyLeftWrist = [observation recognizedPointForJointName:VNHumanBodyPoseObservationJointNameLeftWrist error:nil];
    VNRecognizedPoint *bodyRightWrist = [observation recognizedPointForJointName:VNHumanBodyPoseObservationJointNameRightWrist error:nil];
    BOOL hasLeftBodyWrist = bodyLeftWrist.confidence > 0.1;
    BOOL hasRightBodyWrist = bodyRightWrist.confidence > 0.1;
    NSDictionary<NSString *, NSString *> *handJointNames = @{
        @"wrist": VNHumanHandPoseObservationJointNameWrist,
        @"thumb_cmc": VNHumanHandPoseObservationJointNameThumbCMC,
        @"thumb_mcp": VNHumanHandPoseObservationJointNameThumbMP,
        @"thumb_ip": VNHumanHandPoseObservationJointNameThumbIP,
        @"thumb": VNHumanHandPoseObservationJointNameThumbTip,
        @"index_mcp": VNHumanHandPoseObservationJointNameIndexMCP,
        @"index_pip": VNHumanHandPoseObservationJointNameIndexPIP,
        @"index_dip": VNHumanHandPoseObservationJointNameIndexDIP,
        @"index": VNHumanHandPoseObservationJointNameIndexTip,
        @"middle_mcp": VNHumanHandPoseObservationJointNameMiddleMCP,
        @"middle_pip": VNHumanHandPoseObservationJointNameMiddlePIP,
        @"middle_dip": VNHumanHandPoseObservationJointNameMiddleDIP,
        @"middle": VNHumanHandPoseObservationJointNameMiddleTip,
        @"ring_mcp": VNHumanHandPoseObservationJointNameRingMCP,
        @"ring_pip": VNHumanHandPoseObservationJointNameRingPIP,
        @"ring_dip": VNHumanHandPoseObservationJointNameRingDIP,
        @"ring": VNHumanHandPoseObservationJointNameRingTip,
        @"pinky_mcp": VNHumanHandPoseObservationJointNameLittleMCP,
        @"pinky_pip": VNHumanHandPoseObservationJointNameLittlePIP,
        @"pinky_dip": VNHumanHandPoseObservationJointNameLittleDIP,
        @"pinky": VNHumanHandPoseObservationJointNameLittleTip,
    };
    NSMutableArray<NSDictionary *> *detectedHands = [NSMutableArray array];
    for (VNHumanHandPoseObservation *hand in s_handPoseRequest.results) {
        NSMutableDictionary<NSString *, NSDictionary *> *handPoints = [NSMutableDictionary dictionary];
        for (NSString *joint in handJointNames) {
            VNRecognizedPoint *point = [hand recognizedPointForJointName:handJointNames[joint] error:nil];
            if (point == nil || point.confidence <= 0.01) continue;
            handPoints[joint] = @{
                @"x": @(point.location.x),
                @"y": @(point.location.y),
                @"confidence": @(point.confidence),
            };
        }
        NSDictionary *wristValue = handPoints[@"wrist"];
        if (wristValue != nil) {
            CGPoint wrist = CGPointMake([handPoints[@"wrist"][@"x"] doubleValue], [handPoints[@"wrist"][@"y"] doubleValue]);
            CGFloat leftDistance = hasLeftBodyWrist
                ? hypot(wrist.x - bodyLeftWrist.location.x, wrist.y - bodyLeftWrist.location.y) : CGFLOAT_MAX;
            CGFloat rightDistance = hasRightBodyWrist
                ? hypot(wrist.x - bodyRightWrist.location.x, wrist.y - bodyRightWrist.location.y) : CGFLOAT_MAX;
            NSString *side;
            if (hasLeftBodyWrist && hasRightBodyWrist && fabs(leftDistance - rightDistance) > 0.025) {
                side = leftDistance < rightDistance ? @"left" : @"right";
            } else {
                side = hand.chirality == VNChiralityLeft ? @"left" : @"right";
            }
            [detectedHands addObject:@{@"side": side, @"points": handPoints, @"confidence": @(hand.confidence)}];
        }
    }
    for (NSDictionary *detectedHand in detectedHands) {
        NSString *side = detectedHand[@"side"];
        CGFloat wristDepth = jointDepth[[side stringByAppendingString:@"_wrist"]]
            ? jointDepth[[side stringByAppendingString:@"_wrist"]].doubleValue - centerDepth : 0;
        NSDictionary *handWrist = detectedHand[@"points"][@"wrist"];
        CGFloat lidarWristDepth = handWrist ? CodexDepthAtVisionPoint(depthMap,
            CGPointMake([handWrist[@"x"] doubleValue], [handWrist[@"y"] doubleValue])) : NAN;
        NSDictionary<NSString *, NSDictionary *> *handPoints = detectedHand[@"points"];
        for (NSString *joint in handPoints) {
            CGPoint location = CGPointMake([handPoints[joint][@"x"] doubleValue], [handPoints[joint][@"y"] doubleValue]);
            CGFloat lidarJointDepth = CodexDepthAtVisionPoint(depthMap, location);
            CGFloat depthFromWrist = lidarJointDepth - lidarWristDepth;
            CGFloat pointDepth = isfinite(depthFromWrist) && fabs(depthFromWrist) <= 0.35
                ? wristDepth + depthFromWrist : wristDepth;
            [points addObject:@{
                @"name": [NSString stringWithFormat:@"%@_hand_%@", side, joint],
                @"x": @((location.x - centerX) * metersPerImageUnit),
                @"y": @((centerY - location.y) * metersPerImageUnit),
                @"z": @(pointDepth),
                @"confidence": handPoints[joint][@"confidence"],
                @"image_x": @(location.x),
                @"image_y": @(1.0 - location.y),
                @"image_z": @(pointDepth),
            }];
        }
        NSDictionary *wristValue = handPoints[@"wrist"];
        NSDictionary *indexValue = handPoints[@"index_mcp"];
        NSDictionary *middleValue = handPoints[@"middle_mcp"];
        NSDictionary *pinkyValue = handPoints[@"pinky_mcp"];
        if (wristValue != nil && indexValue != nil && middleValue != nil && pinkyValue != nil) {
            CGPoint wrist = CGPointMake([wristValue[@"x"] doubleValue], [wristValue[@"y"] doubleValue]);
            CGPoint index = CGPointMake([indexValue[@"x"] doubleValue], [indexValue[@"y"] doubleValue]);
            CGPoint middle = CGPointMake([middleValue[@"x"] doubleValue], [middleValue[@"y"] doubleValue]);
            CGPoint pinky = CGPointMake([pinkyValue[@"x"] doubleValue], [pinkyValue[@"y"] doubleValue]);
            CGPoint palm = CGPointMake((wrist.x + index.x + middle.x + pinky.x) * 0.25,
                                       (wrist.y + index.y + middle.y + pinky.y) * 0.25);
            [points addObject:@{
                @"name": [NSString stringWithFormat:@"%@_hand_palm", side],
                @"x": @((palm.x - centerX) * metersPerImageUnit),
                @"y": @((centerY - palm.y) * metersPerImageUnit),
                @"z": @(wristDepth),
                @"confidence": detectedHand[@"confidence"],
                @"image_x": @(palm.x),
                @"image_y": @(1.0 - palm.y),
                @"image_z": @(wristDepth),
            }];
        }
    }

    if (depthMap != nil) CVPixelBufferUnlockBaseAddress(depthMap, kCVPixelBufferLock_ReadOnly);

    VNFaceObservation *face = s_faceRequest.results.firstObject;
    NSDictionary *headRotation = nil;
    if (face.pitch != nil && face.yaw != nil && face.roll != nil) {
        simd_quatf pitch = simd_quaternion(face.pitch.floatValue, simd_make_float3(1, 0, 0));
        simd_quatf yaw = simd_quaternion(face.yaw.floatValue, simd_make_float3(0, 1, 0));
        simd_quatf roll = simd_quaternion(face.roll.floatValue, simd_make_float3(0, 0, 1));
        simd_quatf rotation = simd_mul(simd_mul(yaw, pitch), roll);
        headRotation = @{
            @"x": @(rotation.vector.x), @"y": @(rotation.vector.y),
            @"z": @(rotation.vector.z), @"w": @(rotation.vector.w),
        };
    }
    NSMutableArray *faceShapes = [NSMutableArray array];
    if (face.landmarks != nil) {
        VNFaceLandmarkRegion2D *eyes[] = {face.landmarks.leftEye, face.landmarks.rightEye};
        NSString *eyeNames[] = {@"eyeBlinkLeft", @"eyeBlinkRight"};
        for (int eyeIndex = 0; eyeIndex < 2; eyeIndex++) {
            VNFaceLandmarkRegion2D *eye = eyes[eyeIndex];
            if (eye.pointCount < 4) continue;
            const CGPoint *p = eye.normalizedPoints;
            CGFloat minX = 1, maxX = 0, minY = 1, maxY = 0;
            for (NSUInteger i = 0; i < eye.pointCount; i++) {
                minX = MIN(minX, p[i].x); maxX = MAX(maxX, p[i].x);
                minY = MIN(minY, p[i].y); maxY = MAX(maxY, p[i].y);
            }
            CGFloat ratio = (maxY - minY) / MAX(maxX - minX, 0.001);
            CGFloat blink = MAX(0, MIN(1, (0.28 - ratio) / 0.18));
            [faceShapes addObject:@{@"name": eyeNames[eyeIndex], @"score": @(blink)}];
        }
        VNFaceLandmarkRegion2D *lips = face.landmarks.innerLips;
        if (lips.pointCount >= 4) {
            const CGPoint *p = lips.normalizedPoints;
            CGFloat minX = 1, maxX = 0, minY = 1, maxY = 0;
            for (NSUInteger i = 0; i < lips.pointCount; i++) {
                minX = MIN(minX, p[i].x); maxX = MAX(maxX, p[i].x);
                minY = MIN(minY, p[i].y); maxY = MAX(maxY, p[i].y);
            }
            CGFloat ratio = (maxY - minY) / MAX(maxX - minX, 0.001);
            [faceShapes addObject:@{@"name": @"jawOpen", @"score": @(MAX(0, MIN(1, (ratio - 0.12) / 0.65)))}];
        }
    }
    int64_t timestampMs = (int64_t)(CMTimeGetSeconds(timestamp) * 1000.0);
    CVImageBufferRef image = CMSampleBufferGetImageBuffer(sampleBuffer);
    NSDictionary *packet = @{
        @"version": @4,
        @"frame": @(++s_poseFrame),
        @"timestamp_ms": @(timestampMs),
        @"source_width": @(image != nil ? CVPixelBufferGetHeight(image) : 480),
        @"source_height": @(image != nil ? CVPixelBufferGetWidth(image) : 640),
        @"tracking": @(points.count > 0),
        @"face_blendshapes": faceShapes,
        @"head_rotation": headRotation ?: [NSNull null],
        @"points": points,
    };
    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:packet options:0 error:nil];
    NSString *json = [[NSString alloc] initWithData:jsonData encoding:NSUTF8StringEncoding];
    if (json.length == 0) return;
    dispatch_async(dispatch_get_main_queue(), ^{
        UnitySendMessage(s_unityReceiverName.UTF8String, "OnNativePoseJson", json.UTF8String);
    });
}
@end

static CodexRearPoseDelegate *s_poseDelegate;

static AVCaptureDevice *CodexRearCameraDevice(void) {
    AVCaptureDeviceDiscoverySession *discovery = [AVCaptureDeviceDiscoverySession
        discoverySessionWithDeviceTypes:@[AVCaptureDeviceTypeBuiltInLiDARDepthCamera,
                                         AVCaptureDeviceTypeBuiltInWideAngleCamera]
        mediaType:AVMediaTypeVideo
        position:AVCaptureDevicePositionBack];
    for (AVCaptureDevice *device in discovery.devices)
        if ([device.deviceType isEqualToString:AVCaptureDeviceTypeBuiltInLiDARDepthCamera]) return device;
    return discovery.devices.firstObject;
}

static void CodexStartRearCameraOnMainThread(void) {
    if (s_rearCameraSession != nil) return;
    AVAuthorizationStatus status = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo];
    if (status == AVAuthorizationStatusNotDetermined) {
        [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo completionHandler:^(BOOL granted) {
            if (granted) dispatch_async(dispatch_get_main_queue(), ^{ CodexStartRearCameraOnMainThread(); });
        }];
        return;
    }
    if (status != AVAuthorizationStatusAuthorized) return;

    AVCaptureDevice *device = CodexRearCameraDevice();
    if (device == nil) return;
    NSError *error = nil;
    AVCaptureDeviceInput *input = [AVCaptureDeviceInput deviceInputWithDevice:device error:&error];
    if (input == nil) return;

    AVCaptureSession *session = [AVCaptureSession new];
    BOOL wantsLiDAR = [device.deviceType isEqualToString:AVCaptureDeviceTypeBuiltInLiDARDepthCamera];
    if (wantsLiDAR) {
        session.sessionPreset = AVCaptureSessionPresetInputPriority;
        AVCaptureDeviceFormat *selected = nil;
        int selectedPixels = 0;
        for (AVCaptureDeviceFormat *format in device.formats) {
            if (format.supportedDepthDataFormats.count == 0) continue;
            CMVideoDimensions size = CMVideoFormatDescriptionGetDimensions(format.formatDescription);
            if (size.width * 3 != size.height * 4 || size.width < 640 || size.width > 1920) continue;
            int pixels = size.width * size.height;
            if (pixels > selectedPixels) { selected = format; selectedPixels = pixels; }
        }
        if (selected != nil && [device lockForConfiguration:&error]) {
            device.activeFormat = selected;
            device.activeDepthDataFormat = selected.supportedDepthDataFormats.lastObject;
            [device unlockForConfiguration];
        }
    } else if ([session canSetSessionPreset:AVCaptureSessionPreset640x480]) {
        session.sessionPreset = AVCaptureSessionPreset640x480;
    }
    if (![session canAddInput:input]) return;
    [session addInput:input];

    AVCaptureVideoDataOutput *output = [AVCaptureVideoDataOutput new];
    output.alwaysDiscardsLateVideoFrames = YES;
    output.videoSettings = @{(id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)};
    if (![session canAddOutput:output]) return;
    [session addOutput:output];
    AVCaptureDepthDataOutput *depthOutput = [AVCaptureDepthDataOutput new];
    BOOL hasDepth = wantsLiDAR && device.activeDepthDataFormat != nil &&
                    [session canAddOutput:depthOutput];
    if (hasDepth) {
        depthOutput.alwaysDiscardsLateDepthData = YES;
        depthOutput.filteringEnabled = YES;
        [session addOutput:depthOutput];
    }
    s_captureQueue = dispatch_queue_create("jp.myproject.rear-pose", DISPATCH_QUEUE_SERIAL);
    s_poseDelegate = [CodexRearPoseDelegate new];
    if (hasDepth) {
        s_depthOutput = depthOutput;
        s_outputSynchronizer = [[AVCaptureDataOutputSynchronizer alloc] initWithDataOutputs:@[output, depthOutput]];
        [s_outputSynchronizer setDelegate:s_poseDelegate queue:s_captureQueue];
        NSLog(@"[CodexRearPose] LiDAR video/depth synchronization enabled");
    } else {
        [output setSampleBufferDelegate:s_poseDelegate queue:s_captureQueue];
        NSLog(@"[CodexRearPose] LiDAR unavailable; using Vision wrist depth");
    }
    s_bodyPoseRequest = [VNDetectHumanBodyPoseRequest new];
    s_bodyPoseRequest.preferBackgroundProcessing = YES;
    s_bodyPose3DRequest = [VNDetectHumanBodyPose3DRequest new];
    s_bodyPose3DRequest.preferBackgroundProcessing = YES;
    s_handPoseRequest = [VNDetectHumanHandPoseRequest new];
    s_handPoseRequest.maximumHandCount = 2;
    s_handPoseRequest.preferBackgroundProcessing = YES;
    s_faceRequest = [VNDetectFaceLandmarksRequest new];
    s_faceRequest.preferBackgroundProcessing = YES;
    s_poseFrame = 0;
    s_lastInferenceTime = kCMTimeInvalid;
    s_loggedDepthFrame = NO;

    s_videoOutput = output;
    s_rearCameraSession = session;
    [session startRunning];
}

extern "C" void CodexRearCameraStart(void) {
    dispatch_async(dispatch_get_main_queue(), ^{ CodexStartRearCameraOnMainThread(); });
}

extern "C" void CodexRearCameraStop(void) {
    dispatch_async(dispatch_get_main_queue(), ^{
        [s_outputSynchronizer setDelegate:nil queue:NULL];
        [s_videoOutput setSampleBufferDelegate:nil queue:NULL];
        [s_rearCameraSession stopRunning];
        s_videoOutput = nil;
        s_depthOutput = nil;
        s_outputSynchronizer = nil;
        s_rearCameraSession = nil;
        s_poseDelegate = nil;
        s_bodyPoseRequest = nil;
        s_bodyPose3DRequest = nil;
        s_handPoseRequest = nil;
        s_faceRequest = nil;
    });
}
