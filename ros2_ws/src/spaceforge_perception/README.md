# SpaceForge perception

The perception node implements a detector-to-pose boundary only. It never selects actions, invokes policy, or applies mission safety.

The contract is frozen in PERCEPTION_CONTRACT.md (frames, calibration, keypoints, timestamps, confidence, latency, failure statuses). DeterministicMockDetector and synthetic_target_image provide dependency-free tests. YOLOv8Detector (ultralytics) and OpenCVPnPSolver (opencv) are optional and fail explicitly with SOLVER_UNAVAILABLE when not installed; install `opencv-python-headless` for the real PnP path.

ROS topics are image, camera_info, target_pose, detections, debris, and status under the node namespace. The node preserves source image timestamps and reports confidence, latency, clock skew warnings, and failure status. VALID poses publish a scored bbox Detection2D; failures publish an empty debris array plus the status reason. Optional static output_frame transform applies to VALID poses only.

Tests (dependency-free except opencv for the ground-truth PnP test): valid detection with measured pose error vs ground truth, no detection, ambiguous PnP, bad calibration, invalid detection, solver unavailable, stale image, clock warning, and frame transforms.
