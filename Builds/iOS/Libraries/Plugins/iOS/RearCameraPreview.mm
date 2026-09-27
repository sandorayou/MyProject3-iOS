#import <AVFoundation/AVFoundation.h>
#import <UIKit/UIKit.h>
#import <Vision/Vision.h>
#import <ImageIO/ImageIO.h>
#import "UnityInterface.h"

static AVCaptureSession *s_rearCameraSession;
static AVCaptureVideoPreviewLayer *s_rearCameraPreview;
static AVCaptureVideoDataOutput *s_videoOutput;
static dispatch_queue_t s_captureQueue;
static VNDetectHumanBodyPoseRequest *s_bodyPoseRequest;
static VNDetectHumanHandPoseRequest *s_handPoseRequest;
static uint64_t s_poseFrame;
static CMTime s_lastInferenceTime;
static NSString *const s_unityReceiverName = @"Main Camera";

@interface CodexRearPoseDelegate : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>
@end

@implementation CodexRearPoseDelegate
- (void)captureOutput:(AVCaptureOutput *)output
 didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
        fromConnection:(AVCaptureConnection *)connection {
    CMTime timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer);
    if (CMTIME_IS_VALID(s_lastInferenceTime) && CMTimeGetSeconds(CMTimeSubtract(timestamp, s_lastInferenceTime)) < (1.0 / 15.0)) return;
    s_lastInferenceTime = timestamp;

    VNImageRequestHandler *handler = [[VNImageRequestHandler alloc]
        initWithCMSampleBuffer:sampleBuffer
                   orientation:kCGImagePropertyOrientationRight
                       options:@{}];
    NSError *error = nil;
    if (![handler performRequests:@[s_bodyPoseRequest, s_handPoseRequest] error:&error]) return;
    VNHumanBodyPoseObservation *observation = s_bodyPoseRequest.results.firstObject;
    if (observation == nil) return;

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

    NSMutableArray *points = [NSMutableArray arrayWithCapacity:jointNames.count + 42];
    for (NSString *name in jointNames) {
        VNRecognizedPoint *point = [observation recognizedPointForJointName:jointNames[name] error:nil];
        if (point == nil || point.confidence <= 0.01) continue;
        CGFloat imageY = 1.0 - point.location.y;
        [points addObject:@{
            @"name": name,
            @"x": @((point.location.x - centerX) * metersPerImageUnit),
            @"y": @((centerY - point.location.y) * metersPerImageUnit),
            @"z": @0,
            @"confidence": @(point.confidence),
            @"image_x": @(point.location.x),
            @"image_y": @(imageY),
            @"image_z": @0,
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
        NSDictionary<NSString *, NSDictionary *> *handPoints = detectedHand[@"points"];
        for (NSString *joint in handPoints) {
            CGPoint location = CGPointMake([handPoints[joint][@"x"] doubleValue], [handPoints[joint][@"y"] doubleValue]);
            [points addObject:@{
                @"name": [NSString stringWithFormat:@"%@_hand_%@", side, joint],
                @"x": @((location.x - centerX) * metersPerImageUnit),
                @"y": @((centerY - location.y) * metersPerImageUnit),
                @"z": @0,
                @"confidence": handPoints[joint][@"confidence"],
                @"image_x": @(location.x),
                @"image_y": @(1.0 - location.y),
                @"image_z": @0,
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
                @"z": @0,
                @"confidence": detectedHand[@"confidence"],
                @"image_x": @(palm.x),
                @"image_y": @(1.0 - palm.y),
                @"image_z": @0,
            }];
        }
    }

    int64_t timestampMs = (int64_t)(CMTimeGetSeconds(timestamp) * 1000.0);
    NSDictionary *packet = @{
        @"version": @4,
        @"frame": @(++s_poseFrame),
        @"timestamp_ms": @(timestampMs),
        @"source_width": @480,
        @"source_height": @640,
        @"tracking": @(points.count > 0),
        @"face_blendshapes": @[],
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
        discoverySessionWithDeviceTypes:@[AVCaptureDeviceTypeBuiltInWideAngleCamera]
        mediaType:AVMediaTypeVideo
        position:AVCaptureDevicePositionBack];
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
    if ([session canSetSessionPreset:AVCaptureSessionPreset640x480])
        session.sessionPreset = AVCaptureSessionPreset640x480;
    if (![session canAddInput:input]) return;
    [session addInput:input];

    AVCaptureVideoDataOutput *output = [AVCaptureVideoDataOutput new];
    output.alwaysDiscardsLateVideoFrames = YES;
    output.videoSettings = @{(id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)};
    if (![session canAddOutput:output]) return;
    [session addOutput:output];
    s_captureQueue = dispatch_queue_create("jp.myproject.rear-pose", DISPATCH_QUEUE_SERIAL);
    s_poseDelegate = [CodexRearPoseDelegate new];
    [output setSampleBufferDelegate:s_poseDelegate queue:s_captureQueue];
    s_bodyPoseRequest = [VNDetectHumanBodyPoseRequest new];
    s_bodyPoseRequest.preferBackgroundProcessing = YES;
    s_handPoseRequest = [VNDetectHumanHandPoseRequest new];
    s_handPoseRequest.maximumHandCount = 2;
    s_handPoseRequest.preferBackgroundProcessing = YES;
    s_poseFrame = 0;
    s_lastInferenceTime = kCMTimeInvalid;

    UIView *unityView = UnityGetMainWindow().rootViewController.view;
    AVCaptureVideoPreviewLayer *preview = [AVCaptureVideoPreviewLayer layerWithSession:session];
    preview.videoGravity = AVLayerVideoGravityResizeAspectFill;
    preview.frame = unityView.bounds;
    AVCaptureConnection *previewConnection = preview.connection;
    if ([previewConnection isVideoOrientationSupported]) previewConnection.videoOrientation = AVCaptureVideoOrientationPortrait;
    [unityView.layer insertSublayer:preview atIndex:0];

    s_videoOutput = output;
    s_rearCameraPreview = preview;
    s_rearCameraSession = session;
    [session startRunning];
}

extern "C" void CodexRearCameraStart(void) {
    dispatch_async(dispatch_get_main_queue(), ^{ CodexStartRearCameraOnMainThread(); });
}

extern "C" void CodexRearCameraStop(void) {
    dispatch_async(dispatch_get_main_queue(), ^{
        [s_videoOutput setSampleBufferDelegate:nil queue:NULL];
        [s_rearCameraSession stopRunning];
        [s_rearCameraPreview removeFromSuperlayer];
        s_videoOutput = nil;
        s_rearCameraPreview = nil;
        s_rearCameraSession = nil;
        s_poseDelegate = nil;
        s_bodyPoseRequest = nil;
        s_handPoseRequest = nil;
    });
}
