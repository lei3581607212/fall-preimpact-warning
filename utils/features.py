"""Pose validation and temporal feature construction shared by offline and live code."""

import numpy as np

NUM_KEYPOINTS = 17
POSE_DIM = NUM_KEYPOINTS * 2
FEATURE_DIM = POSE_DIM * 2
ARM_KEYPOINTS = (7, 8, 9, 10)
BODY_RELATIVE_PROFILE = "body_relative_v1"
CORE_GEOMETRY_PROFILE = "core_geometry_v1"
CORE_FUSION_PROFILE = "core_fusion_v1"
LONG_MOTION_PROFILE = "long_motion_v1"
STABILITY_MOTION_PROFILE = "stability_motion_v1"
PAPER_RAW_PROFILE = "paper_raw_v1"
PAPER_PHYSICS_PROFILE = "paper_physics_v1"
PAPER_PHYSICS_NO_QUALITY_PROFILE = "paper_physics_no_quality_v1"
GRAPH_POSE_PROFILE = "graph_pose_v1"
REFERENCE_YOLO26_PROFILE = "reference_yolo26_v1"

# The reference project uses coordinates only. These are the 13 joints shared
# by MediaPipe's training CSV and the K230 YOLO Pose model: nose, shoulders,
# elbows, wrists, hips, knees, and ankles.
REFERENCE_YOLO26_COORDINATES = (0, 1, *range(10, 34))


def normalize_keypoints(keypoints, width, height):
    points = np.asarray(keypoints, dtype=np.float32).reshape(NUM_KEYPOINTS, 2).copy()
    points[:, 0] /= max(width, 1)
    points[:, 1] /= max(height, 1)
    return points.reshape(-1)


def pose_is_valid(keypoints, keypoint_confidence=None, min_visible=8, min_confidence=0.25):
    points = np.asarray(keypoints).reshape(NUM_KEYPOINTS, 2)
    nonzero = np.logical_or(points[:, 0] > 0, points[:, 1] > 0)
    if keypoint_confidence is not None:
        confidence = np.asarray(keypoint_confidence).reshape(-1)[:NUM_KEYPOINTS]
        nonzero &= confidence >= min_confidence
    return int(nonzero.sum()) >= min_visible


def body_relative_pose(pose):
    """Express joints relative to hip center and torso length, independent of framing."""
    points = np.asarray(pose, dtype=np.float32).reshape(NUM_KEYPOINTS, 2)
    visible = np.any(points != 0.0, axis=1)
    if not (visible[5] and visible[6] and visible[11] and visible[12]):
        return np.zeros(POSE_DIM, dtype=np.float32)
    shoulder_center = (points[5] + points[6]) / 2
    hip_center = (points[11] + points[12]) / 2
    torso_length = float(np.linalg.norm(shoulder_center - hip_center))
    if torso_length < 1e-4:
        return np.zeros(POSE_DIM, dtype=np.float32)
    relative = np.zeros_like(points)
    relative[visible] = (points[visible] - hip_center) / torso_length
    return relative.reshape(-1)


def make_feature(current_pose, previous_valid_pose, feature_profile="image_relative_v1"):
    """Return 34 normalized coordinates plus velocity; never create a velocity spike after loss."""
    current = np.asarray(current_pose, dtype=np.float32).reshape(POSE_DIM)
    if feature_profile == BODY_RELATIVE_PROFILE:
        current = body_relative_pose(current)
        previous = None if previous_valid_pose is None else body_relative_pose(previous_valid_pose)
    else:
        previous = previous_valid_pose
    if previous_valid_pose is None:
        velocity = np.zeros(POSE_DIM, dtype=np.float32)
    else:
        velocity = current - np.asarray(previous, dtype=np.float32).reshape(POSE_DIM)
    return np.concatenate((current, velocity)).astype(np.float32)


def resample_feature_stream(features, source_fps, target_fps=25.0):
    """Resample pose coordinates and recompute velocity at a fixed temporal rate.

    Returned source positions keep impact-frame labels in the original timeline.
    Interpolation never bridges an invalid pose; such samples remain zero.
    """
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected [frames, {FEATURE_DIM}] features")
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    if not len(values):
        return np.empty((0, FEATURE_DIM), dtype=np.float32), np.empty(0, dtype=np.float32)
    positions = np.arange(0, max(len(values) - 1, 0) + 1e-6, source_fps / target_fps)
    positions = np.minimum(positions, max(len(values) - 1, 0))
    poses = values[:, :POSE_DIM]
    result, previous = [], None
    for position in positions:
        left, right = int(np.floor(position)), int(np.ceil(position))
        if not len(poses) or not np.any(poses[left]) or not np.any(poses[right]):
            result.append(np.zeros(FEATURE_DIM, dtype=np.float32)); previous = None; continue
        fraction = position - left
        pose = poses[left] * (1.0 - fraction) + poses[right] * fraction
        result.append(make_feature(pose, previous)); previous = pose
    return np.asarray(result, dtype=np.float32), positions


def suppress_arm_features(features):
    """Neutralize elbows/wrists so an isolated raised arm cannot drive the classifier."""
    result = np.asarray(features, dtype=np.float32).copy()
    for keypoint in ARM_KEYPOINTS:
        coordinate = keypoint * 2
        result[..., coordinate : coordinate + 2] = 0.0
        result[..., POSE_DIM + coordinate : POSE_DIM + coordinate + 2] = 0.0
    return result


def paper_physics_descriptors(features):
    """Build causal, scale-normalized pose physics and quality descriptors.

    Every temporal derivative is emitted only when both adjacent observations
    are valid. This prevents pose dropouts and detector reacquisition from
    appearing as abrupt body motion.
    """
    base = np.asarray(features, dtype=np.float32)
    if base.ndim < 2 or base.shape[-1] != FEATURE_DIM:
        raise ValueError(f"Expected [..., time, {FEATURE_DIM}] features")
    shape = base.shape
    sequences = base.reshape(-1, shape[-2], FEATURE_DIM)
    output = np.zeros((len(sequences), shape[-2], 12), dtype=np.float32)

    for sequence_index, sequence in enumerate(sequences):
        points = sequence[:, :POSE_DIM].reshape(-1, NUM_KEYPOINTS, 2)
        velocities = sequence[:, POSE_DIM:].reshape(-1, NUM_KEYPOINTS, 2)
        visible = np.any(points != 0.0, axis=2)
        shoulders_valid = visible[:, 5] & visible[:, 6]
        hips_valid = visible[:, 11] & visible[:, 12]
        ankles_valid = visible[:, 15] & visible[:, 16]
        knees_valid = visible[:, 13] & visible[:, 14]
        torso_valid = shoulders_valid & hips_valid

        shoulder = (points[:, 5] + points[:, 6]) / 2.0
        hip = (points[:, 11] + points[:, 12]) / 2.0
        ankle = (points[:, 15] + points[:, 16]) / 2.0
        torso = shoulder - hip
        torso_length = np.linalg.norm(torso, axis=1)
        torso_valid &= torso_length >= 1e-4
        scale = np.where(torso_valid, torso_length, 1.0)

        angle = np.zeros(len(sequence), dtype=np.float32)
        angle[torso_valid] = np.arctan2(torso[torso_valid, 0], -torso[torso_valid, 1]) / np.pi
        angle_velocity = np.zeros_like(angle)
        angle_transition = torso_valid[1:] & torso_valid[:-1]
        wrapped_delta = (angle[1:] - angle[:-1] + 1.0) % 2.0 - 1.0
        angle_velocity[1:] = np.where(angle_transition, wrapped_delta, 0.0)

        support_valid = torso_valid & ankles_valid
        hip_height = np.zeros(len(sequence), dtype=np.float32)
        support_width = np.zeros(len(sequence), dtype=np.float32)
        hip_height[support_valid] = (ankle[support_valid, 1] - hip[support_valid, 1]) / scale[support_valid]
        support_width[support_valid] = np.abs(points[support_valid, 15, 0] - points[support_valid, 16, 0]) / scale[support_valid]

        hip_height_velocity = np.zeros_like(hip_height)
        support_velocity = np.zeros_like(support_width)
        support_transition = support_valid[1:] & support_valid[:-1]
        hip_height_velocity[1:] = np.where(support_transition, hip_height[1:] - hip_height[:-1], 0.0)
        support_velocity[1:] = np.where(support_transition, support_width[1:] - support_width[:-1], 0.0)

        ankle_height_asymmetry = np.zeros(len(sequence), dtype=np.float32)
        ankle_height_asymmetry[support_valid] = (
            np.abs(points[support_valid, 15, 1] - points[support_valid, 16, 1]) / scale[support_valid]
        )
        leg_valid = support_valid & knees_valid
        leg_length_asymmetry = np.zeros(len(sequence), dtype=np.float32)
        left_leg = np.linalg.norm(points[:, 13] - points[:, 15], axis=1)
        right_leg = np.linalg.norm(points[:, 14] - points[:, 16], axis=1)
        leg_length_asymmetry[leg_valid] = np.abs(left_leg[leg_valid] - right_leg[leg_valid]) / scale[leg_valid]

        motion_dispersion = np.zeros(len(sequence), dtype=np.float32)
        for frame_index in range(1, len(sequence)):
            tracked = visible[frame_index] & visible[frame_index - 1]
            if torso_valid[frame_index] and tracked.sum() >= 4:
                joint_velocity = velocities[frame_index, tracked]
                center_velocity = np.median(joint_velocity, axis=0)
                residual = np.linalg.norm(joint_velocity - center_velocity, axis=1)
                motion_dispersion[frame_index] = float(np.median(residual) / scale[frame_index])

        core_visible = visible[:, [5, 6, 11, 12, 13, 14, 15, 16]].mean(axis=1)
        valid_transition = np.zeros(len(sequence), dtype=np.float32)
        valid_transition[1:] = (torso_valid[1:] & torso_valid[:-1]).astype(np.float32)
        output[sequence_index] = np.stack(
            (angle, angle_velocity, hip_height, hip_height_velocity,
             support_width, support_velocity, ankle_height_asymmetry,
             leg_length_asymmetry, visible.mean(axis=1), core_visible,
             motion_dispersion, valid_transition), axis=1,
        )

    # Robust bounds keep isolated detector errors from dominating optimization.
    output[..., 1] = np.clip(output[..., 1], -1.0, 1.0)
    output[..., 2:8] = np.clip(output[..., 2:8], -5.0, 5.0)
    output[..., 10] = np.clip(output[..., 10], 0.0, 5.0)
    return output.reshape(shape[:-1] + (12,))


def inject_core_geometry(features):
    """Replace arm slots with fall-relevant torso geometry and its frame-to-frame change."""
    result = suppress_arm_features(features)
    flat = result.reshape(-1, FEATURE_DIM)
    points = flat[:, :POSE_DIM].reshape(-1, NUM_KEYPOINTS, 2)
    geometry = np.zeros((len(flat), 8), dtype=np.float32)
    for index, pose in enumerate(points):
        visible = np.any(pose != 0.0, axis=1)
        if not (visible[5] and visible[6] and visible[11] and visible[12]):
            continue
        shoulder = (pose[5] + pose[6]) / 2
        hip = (pose[11] + pose[12]) / 2
        torso = shoulder - hip
        torso_length = float(np.linalg.norm(torso))
        if torso_length < 1e-4:
            continue
        scale_points = pose[[5, 6, 11, 12, 15, 16]][visible[[5, 6, 11, 12, 15, 16]]]
        body_width = float(scale_points[:, 0].max() - scale_points[:, 0].min())
        body_height = float(scale_points[:, 1].max() - scale_points[:, 1].min())
        ankle = (pose[15] + pose[16]) / 2 if visible[15] and visible[16] else hip
        standing_height = max(float(np.linalg.norm(shoulder - ankle)), torso_length)
        geometry[index] = (
            torso[0] / torso_length, torso[1] / torso_length,
            torso_length / standing_height,
            float(np.linalg.norm(pose[5] - pose[6])) / torso_length,
            float(np.linalg.norm(pose[11] - pose[12])) / torso_length,
            body_width / standing_height,
            body_height / standing_height,
            (shoulder[1] - ankle[1]) / standing_height,
        )
    geometry_delta = np.zeros_like(geometry)
    geometry_delta[1:] = geometry[1:] - geometry[:-1]
    arm_coordinates = [14, 15, 16, 17, 18, 19, 20, 21]
    arm_velocity = [48, 49, 50, 51, 52, 53, 54, 55]
    flat[:, arm_coordinates] = geometry
    flat[:, arm_velocity] = geometry_delta
    return result


def prepare_features(features, feature_profile):
    if feature_profile == PAPER_RAW_PROFILE:
        return np.asarray(features, dtype=np.float32)
    if feature_profile == PAPER_PHYSICS_PROFILE:
        base = np.asarray(features, dtype=np.float32)
        return np.concatenate((base, paper_physics_descriptors(base)), axis=-1)
    if feature_profile == PAPER_PHYSICS_NO_QUALITY_PROFILE:
        base = np.asarray(features, dtype=np.float32)
        # Keep the eight causal body-motion descriptors and remove the four
        # pose-validity/motion-quality descriptors for the matched ablation.
        return np.concatenate((base, paper_physics_descriptors(base)[..., :8]), axis=-1)
    if feature_profile == REFERENCE_YOLO26_PROFILE:
        return np.asarray(features, dtype=np.float32)[..., REFERENCE_YOLO26_COORDINATES]
    if feature_profile == GRAPH_POSE_PROFILE:
        return np.asarray(features, dtype=np.float32)
    if feature_profile == CORE_GEOMETRY_PROFILE:
        return inject_core_geometry(features)
    if feature_profile == CORE_FUSION_PROFILE:
        geometry_only = inject_core_geometry(features)
        geometry_slots = [14, 15, 16, 17, 18, 19, 20, 21, 48, 49, 50, 51, 52, 53, 54, 55]
        return np.concatenate((np.asarray(features, dtype=np.float32), geometry_only[..., geometry_slots]), axis=-1)
    if feature_profile == LONG_MOTION_PROFILE:
        base = np.asarray(features, dtype=np.float32)
        shape = base.shape
        flat = base.reshape(-1, shape[-2], FEATURE_DIM)
        descriptors = np.zeros((len(flat), shape[-2], 6), dtype=np.float32)
        for sequence_index, sequence in enumerate(flat):
            points = sequence[:, :POSE_DIM].reshape(-1, NUM_KEYPOINTS, 2)
            visible = np.any(points != 0.0, axis=2)
            torso_angle = np.zeros(len(sequence), dtype=np.float32)
            hip_y = np.zeros(len(sequence), dtype=np.float32)
            for frame_index, pose in enumerate(points):
                descriptors[sequence_index, frame_index, 5] = visible[frame_index].mean()
                if visible[frame_index, 5] and visible[frame_index, 6] and visible[frame_index, 11] and visible[frame_index, 12]:
                    shoulder = (pose[5] + pose[6]) / 2; hip = (pose[11] + pose[12]) / 2
                    torso = shoulder - hip
                    torso_angle[frame_index] = np.arctan2(torso[1], torso[0]) / np.pi
                    hip_y[frame_index] = hip[1]
            hip_velocity = np.diff(hip_y, prepend=hip_y[0])
            descriptors[sequence_index, :, 0] = torso_angle
            descriptors[sequence_index, :, 1] = np.diff(torso_angle, prepend=torso_angle[0])
            descriptors[sequence_index, :, 2] = hip_y
            descriptors[sequence_index, :, 3] = hip_velocity
            descriptors[sequence_index, :, 4] = np.diff(hip_velocity, prepend=hip_velocity[0])
        return np.concatenate((base, descriptors.reshape(shape[:-1] + (6,))), axis=-1)
    if feature_profile == STABILITY_MOTION_PROFILE:
        base = np.asarray(features, dtype=np.float32)
        shape = base.shape
        flat = base.reshape(-1, shape[-2], FEATURE_DIM)
        descriptors = np.zeros((len(flat), shape[-2], 12), dtype=np.float32)
        for sequence_index, sequence in enumerate(flat):
            points = sequence[:, :POSE_DIM].reshape(-1, NUM_KEYPOINTS, 2)
            visible = np.any(points != 0.0, axis=2)
            torso_angle = np.zeros(len(sequence), dtype=np.float32)
            hip_x = np.zeros(len(sequence), dtype=np.float32)
            hip_y = np.zeros(len(sequence), dtype=np.float32)
            support_width = np.zeros(len(sequence), dtype=np.float32)
            hip_support_gap = np.zeros(len(sequence), dtype=np.float32)
            for frame_index, pose in enumerate(points):
                if visible[frame_index, 11] and visible[frame_index, 12]:
                    hip = (pose[11] + pose[12]) / 2; hip_x[frame_index], hip_y[frame_index] = hip
                if visible[frame_index, 5] and visible[frame_index, 6] and visible[frame_index, 11] and visible[frame_index, 12]:
                    shoulder = (pose[5] + pose[6]) / 2; hip = (pose[11] + pose[12]) / 2
                    torso = shoulder - hip; torso_angle[frame_index] = np.arctan2(torso[1], torso[0]) / np.pi
                if visible[frame_index, 15] and visible[frame_index, 16]:
                    ankles = (pose[15] + pose[16]) / 2
                    support_width[frame_index] = abs(float(pose[15, 0] - pose[16, 0]))
                    hip_support_gap[frame_index] = float(hip_y[frame_index] - ankles[1])
            hip_velocity = np.diff(hip_y, prepend=hip_y[0]); hip_accel = np.diff(hip_velocity, prepend=hip_velocity[0])
            descriptors[sequence_index] = np.stack((torso_angle, np.diff(torso_angle, prepend=torso_angle[0]), hip_x, hip_y, hip_velocity, hip_accel, support_width, hip_support_gap, np.diff(support_width, prepend=support_width[0]), visible.mean(axis=1), np.diff(hip_x, prepend=hip_x[0]), np.diff(hip_support_gap, prepend=hip_support_gap[0])), axis=1)
        return np.concatenate((base, descriptors.reshape(shape[:-1] + (12,))), axis=-1)
    return suppress_arm_features(features)
