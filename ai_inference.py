import argparse
from enum import Enum
from typing import Iterator, List

import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-ball-detection.pt')

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 60
CONFIG = SoccerPitchConfiguration()

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    thickness=2,
    edges=CONFIG.edges,
)
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    base=20,
    height=15,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


class Mode(Enum):
    """
    Enum class representing different modes of operation for Soccer AI video analysis.
    """
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    BALL_DETECTION = 'BALL_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR = 'RADAR'

def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]

def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.array,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Resolve the team IDs for detected goalkeepers based on the proximity to team
    centroids.
    """
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    
    players_0 = players_xy[players_team_id == 0]
    players_1 = players_xy[players_team_id == 1]
    
    team_0_centroid = players_0.mean(axis=0) if len(players_0) > 0 else np.array([0, 0])
    team_1_centroid = players_1.mean(axis=0) if len(players_1) > 0 else np.array([0, 0])
    
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)

def render_radar(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray
) -> np.ndarray:
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)

    radar = draw_pitch(config=CONFIG)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 0],
        face_color=sv.Color.from_hex(COLORS[0]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 1],
        face_color=sv.Color.from_hex(COLORS[1]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 2],
        face_color=sv.Color.from_hex(COLORS[2]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 3],
        face_color=sv.Color.from_hex(COLORS[3]), radius=20, pitch=radar)
    return radar


def run_analysis(source_video_path: str, target_video_path: str, device: str, modes: List[Mode]) -> None:
    """
    Run combined soccer analysis on a video based on requested modes.
    """
    enable_radar = Mode.RADAR in modes
    enable_team = Mode.TEAM_CLASSIFICATION in modes or enable_radar
    enable_tracking = Mode.PLAYER_TRACKING in modes or enable_team
    enable_pitch = Mode.PITCH_DETECTION in modes or enable_radar
    enable_ball = Mode.BALL_DETECTION in modes
    enable_player = Mode.PLAYER_DETECTION in modes or enable_tracking
    
    # Load required models
    print("Loading models...")
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device) if enable_player else None
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device) if enable_pitch else None
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device) if enable_ball else None
    
    # Initialize trackers and classifier if needed
    tracker = sv.ByteTrack(minimum_consecutive_frames=3) if enable_tracking else None
    team_classifier = TeamClassifier(device=device) if enable_team else None
    ball_tracker = BallTracker(buffer_size=20) if enable_ball else None
    ball_annotator = BallAnnotator(radius=6, buffer_size=10) if enable_ball else None

    # Pre-train team classifier if team classification or radar is enabled
    if enable_team:
        print("Pre-training Team Classifier...")
        frame_generator = sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE)
        crops = []
        for frame in tqdm(frame_generator, desc='collecting crops'):
            result = player_detection_model(frame, imgsz=640, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])
            if len(crops) > 200:
                break
        if len(crops) < 20:
            if len(crops) == 0:
                crops = [np.zeros((100, 100, 3), dtype=np.uint8), np.ones((100, 100, 3), dtype=np.uint8)*255]
            crops = crops * int(30 / len(crops) + 1)
        team_classifier.fit(crops)
    
    # Inference setup
    print("Running analysis pipeline...")
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    
    # State variables
    team_colors_cache = {}
    player_numbers = {}
    color_counters = {0: 0, 1: 0, 2: 0, 3: 0}
    
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in tqdm(frame_generator, desc="Processing frames"):
            annotated_frame = frame.copy()
            
            # --- PITCH DETECTION ---
            keypoints = None
            if enable_pitch:
                pitch_result = pitch_detection_model(frame, verbose=False)[0]
                keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
                # Draw pitch if strictly requested (and radar doesn't override its UI, or maybe we want both)
                # But radar doesn't draw points on the main frame, it draws a minimap.
                # So if PITCH_DETECTION is requested, draw vertex labels.
                if Mode.PITCH_DETECTION in modes:
                    annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(annotated_frame, keypoints, CONFIG.labels)

            # --- BALL DETECTION ---
            if enable_ball:
                ball_result = ball_detection_model(frame, imgsz=640, verbose=False)[0]
                ball_det = sv.Detections.from_ultralytics(ball_result)
                ball_det = ball_tracker.update(ball_det)
                annotated_frame = ball_annotator.annotate(annotated_frame, ball_det)

            # --- PLAYER DETECTION / TRACKING / CLASSIFICATION ---
            detections = None
            if enable_player:
                player_result = player_detection_model(frame, imgsz=640, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(player_result)

                if enable_tracking:
                    detections = tracker.update_with_detections(detections)
                    
                    if enable_team:
                        players = detections[detections.class_id == PLAYER_CLASS_ID]
                        
                        crops_to_predict = []
                        indices_to_predict = []
                        for i, tracker_id in enumerate(players.tracker_id):
                            if tracker_id not in team_colors_cache:
                                crops_to_predict.append(get_crops(frame, players[i:i+1])[0])
                                indices_to_predict.append(i)
                                
                        if crops_to_predict:
                            predicted_teams = team_classifier.predict(crops_to_predict)
                            for idx, team_id in zip(indices_to_predict, predicted_teams):
                                tracker_id = players.tracker_id[idx]
                                team_colors_cache[tracker_id] = team_id
                                
                        players_team_id = np.array([team_colors_cache.get(tracker_id, 0) for tracker_id in players.tracker_id])
                        
                        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
                        goalkeepers_team_id = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)
                        
                        referees = detections[detections.class_id == REFEREE_CLASS_ID]
                        
                        # Merge back with ordered IDs
                        detections = sv.Detections.merge([players, goalkeepers, referees])
                        color_lookup = np.array(
                            players_team_id.tolist() +
                            goalkeepers_team_id.tolist() +
                            [REFEREE_CLASS_ID] * len(referees)
                        )
                        
                        labels = []
                        for tracker_id, color_id in zip(detections.tracker_id, color_lookup):
                            if color_id == REFEREE_CLASS_ID:
                                labels.append("Sedzia")
                            else:
                                if tracker_id not in player_numbers:
                                    color_counters[color_id] += 1
                                    player_numbers[tracker_id] = color_counters[color_id]
                                labels.append(str(player_numbers[tracker_id]))
                                
                        # Only draw tracking / classification if requested
                        if Mode.PLAYER_TRACKING in modes or Mode.TEAM_CLASSIFICATION in modes or Mode.RADAR in modes:
                            annotated_frame = ELLIPSE_ANNOTATOR.annotate(
                                annotated_frame, detections, custom_color_lookup=color_lookup)
                            annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
                                annotated_frame, detections, labels, custom_color_lookup=color_lookup)
                                
                        # --- RADAR ---
                        if enable_radar and keypoints is not None:
                            h, w, _ = frame.shape
                            radar = render_radar(detections, keypoints, color_lookup)
                            radar = sv.resize_image(radar, (w // 2, h // 2))
                            radar_h, radar_w, _ = radar.shape
                            rect = sv.Rect(
                                x=w // 2 - radar_w // 2,
                                y=h - radar_h,
                                width=radar_w,
                                height=radar_h
                            )
                            annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)
                            
                    else:
                        # Tracking enabled, Team classification disabled
                        if Mode.PLAYER_TRACKING in modes:
                            labels = [str(tracker_id) for tracker_id in detections.tracker_id]
                            annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
                            annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(annotated_frame, detections, labels=labels)

                else:
                    # Generic player detection without tracking
                    if Mode.PLAYER_DETECTION in modes:
                        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
                        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)

            sink.write_frame(annotated_frame)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Soccer AI Video Analysis')
    parser.add_argument('--source_video_path', type=str, required=True)
    parser.add_argument('--target_video_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--modes', nargs='+', type=str, required=True, 
                        help='List of modes to run: PITCH_DETECTION PLAYER_DETECTION BALL_DETECTION PLAYER_TRACKING TEAM_CLASSIFICATION RADAR')
    
    args = parser.parse_args()
    
    # Parse passed modes strings to Enum
    selected_modes = []
    for m in args.modes:
        try:
            selected_modes.append(Mode(m.upper()))
        except ValueError:
            print(f"Warning: Mode '{m}' is not recognized and will be ignored.")
            
    if not selected_modes:
        print("No valid modes provided. Exiting.")
        exit(1)
        
    run_analysis(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        modes=selected_modes
    )
