# feature_extractor.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import av
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from video_mae_for_pretraining2 import CompatibleVideoMAEForPreTraining, load_compatible_model

class VideoMAEFeatureExtractor:
    def __init__(self, model_path: str, device: str = 'auto', target_fps: float = None):
        """
        Fixed VideoMAE Feature Extractor with FPS control
        
        Args:
            model_path: Path to trained model
            device: Device to use ('auto', 'cuda', 'cpu')
            target_fps: Target FPS for frame sampling (None for original sampling)
        """
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path)
        self.num_frames = 16
        self.image_size = 224
        self.target_fps = target_fps
        
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"VideoMAE feature extractor has been initialized")
        print(f"Device: {self.device}")
        if target_fps:
            print(f"Target FPS: {target_fps}")
        else:
            print(f"Target FPS: Auto (uniform sampling)")
    
    def _setup_device(self, device: str) -> torch.device:
        """Device setup"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Model loading"""
        
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            print("Using pretrained model...")
            
            # Fallback: pretrained model
            model = CompatibleVideoMAEForPreTraining("MCG-NJU/videomae-base")
        else:
            # Compatible model loading
            model = load_compatible_model(model_path, self.device)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def _calculate_frame_indices_with_fps(self, start_frame: int, end_frame: int, 
                                        original_fps: float, target_fps: float = None) -> np.ndarray:
        """
        Calculate frame indices based on target FPS
        
        Args:
            start_frame: Start frame index
            end_frame: End frame index
            original_fps: Original video FPS
            target_fps: Target FPS for sampling (None for uniform sampling)
            
        Returns:
            Array of frame indices to extract
        """
        total_available_frames = end_frame - start_frame
        
        if target_fps is None:
            # Original uniform sampling method
            if total_available_frames >= self.num_frames:
                frame_indices = np.linspace(start_frame, end_frame - 1, self.num_frames, dtype=int)
            else:
                frame_indices = np.arange(start_frame, end_frame)
        else:
            # FPS-based sampling
            duration_seconds = total_available_frames / original_fps
            target_frames_needed = int(duration_seconds * target_fps)
            
            if target_frames_needed <= 0:
                # If target FPS is too low, use at least 1 frame
                frame_indices = np.array([start_frame])
            elif target_frames_needed >= total_available_frames:
                # If target FPS is higher than original, use all available frames
                frame_indices = np.arange(start_frame, end_frame)
            else:
                # Sample frames at target FPS
                frame_step = original_fps / target_fps
                frame_indices = []
                
                current_frame_float = start_frame
                while current_frame_float < end_frame and len(frame_indices) < self.num_frames:
                    frame_indices.append(int(round(current_frame_float)))
                    current_frame_float += frame_step
                
                frame_indices = np.array(frame_indices)
                # Remove duplicates and sort
                frame_indices = np.unique(frame_indices)
                frame_indices = frame_indices[frame_indices < end_frame]
        
        return frame_indices
    
    def extract_clip_features(self, video_frames: torch.Tensor) -> torch.Tensor:
        """Extract features from video clip"""
        with torch.no_grad():
            if video_frames.dim() == 4:
                video_frames = video_frames.unsqueeze(0)
            
            video_frames = video_frames.to(self.device)
            
            # Extract features with VideoMAE encoder
            outputs = self.model.videomae(
                pixel_values=video_frames,
                return_dict=True
            )
            
            features = outputs.last_hidden_state.squeeze(0)
            return features.cpu()
    
    def extract_global_features(self, video_frames: torch.Tensor) -> torch.Tensor:
        """Extract global features"""
        clip_features = self.extract_clip_features(video_frames)
        global_features = clip_features.mean(dim=0)
        return global_features
    
    def get_video_info(self, video_path: str) -> Dict:
        """Get video information including FPS"""
        try:
            container = av.open(video_path)
            video_stream = container.streams.video[0]
            
            fps = float(video_stream.average_rate)
            total_frames = video_stream.frames
            duration = float(container.duration / av.time_base) if container.duration else None
            
            # If duration is not available, calculate from frames and FPS
            if duration is None and total_frames and fps:
                duration = total_frames / fps
            
            container.close()
            
            info = {
                'fps': fps,
                'total_frames': total_frames,
                'duration': duration,
                'width': video_stream.width,
                'height': video_stream.height
            }
            
            return info
            
        except Exception as e:
            raise RuntimeError(f"Failed to get video info from {video_path}: {e}")
    
    def load_video_frames(self, 
                          video_path: str, 
                          start_time: float = 0.0, 
                          duration: Optional[float] = None,
                          target_fps: float = None) -> Tuple[torch.Tensor, Dict]:
        """
        Load frames from video file with FPS control
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            duration: Duration in seconds (None for auto-calculation)
            target_fps: Target FPS for sampling (overrides instance target_fps)
            
        Returns:
            Tuple of (frames tensor, sampling info)
        """
        try:
            # Get video information
            video_info = self.get_video_info(video_path)
            original_fps = video_info['fps']
            total_frames = video_info['total_frames']
            
            # Use target_fps parameter or instance target_fps
            effective_target_fps = target_fps if target_fps is not None else self.target_fps
            
            # Calculate duration if not provided
            if duration is None:
                if effective_target_fps:
                    # Duration to get num_frames at target FPS
                    duration = self.num_frames / effective_target_fps
                else:
                    # Original method: duration based on original FPS
                    duration = self.num_frames / original_fps
            
            # Calculate frame range
            start_frame = int(start_time * original_fps)
            end_frame = int((start_time + duration) * original_fps)
            end_frame = min(end_frame, total_frames)
            
            total_target_frames = end_frame - start_frame
            if total_target_frames <= 0:
                raise ValueError(f"Invalid time range: start_time={start_time}, duration={duration}")
            
            # Calculate frame indices based on FPS
            frame_indices = self._calculate_frame_indices_with_fps(
                start_frame, end_frame, original_fps, effective_target_fps
            )
            
            # Open video for frame extraction
            container = av.open(video_path)
            
            frames = []
            frame_count = 0
            target_idx = 0
            
            # Seek to start time
            if start_time > 0:
                container.seek(int(start_time * av.time_base))
            
            # Extract frames
            for frame in container.decode(video=0):
                current_frame_idx = start_frame + frame_count
                
                if target_idx < len(frame_indices) and current_frame_idx == frame_indices[target_idx]:
                    pil_frame = frame.to_image()
                    tensor_frame = self.transform(pil_frame)
                    frames.append(tensor_frame)
                    target_idx += 1
                    
                    if len(frames) >= self.num_frames:
                        break
                
                frame_count += 1
                
                if current_frame_idx >= end_frame:
                    break
            
            container.close()
            
            # Pad frames if necessary
            if len(frames) < self.num_frames:
                if frames:
                    # Repeat last frame to reach num_frames
                    while len(frames) < self.num_frames:
                        frames.append(frames[-1].clone())
                else:
                    raise ValueError(f"Failed to load frames: {video_path}")
            
            # Create sampling info
            actual_fps = len(frame_indices) / duration if duration > 0 else 0
            sampling_info = {
                'original_fps': original_fps,
                'target_fps': effective_target_fps,
                'actual_fps': actual_fps,
                'total_frames_extracted': len(frames),
                'frame_indices': frame_indices.tolist(),
                'duration': duration,
                'start_time': start_time,
                'fps_ratio': effective_target_fps / original_fps if effective_target_fps else None
            }
            
            return torch.stack(frames[:self.num_frames]), sampling_info
            
        except Exception as e:
            raise RuntimeError(f"Video loading error ({video_path}): {e}")
    
    def extract_features_from_video(self, video_path: str, 
                                    clip_duration: float = 2.0,
                                    overlap: float = 0.0,
                                    target_fps: float = None) -> List[Dict]:
        """
        Extract features per clip from video file with FPS control
        
        Args:
            video_path: Path to video file
            clip_duration: Duration of each clip in seconds
            overlap: Overlap between clips in seconds
            target_fps: Target FPS for sampling (overrides instance target_fps)
            
        Returns:
            List of dictionaries containing features and metadata
        """
        print(f"Feature extraction started: {video_path}")
        
        # Get video information
        video_info = self.get_video_info(video_path)
        duration = video_info['duration']
        original_fps = video_info['fps']
        
        # Use target_fps parameter or instance target_fps
        effective_target_fps = target_fps if target_fps is not None else self.target_fps
        
        print(f"Video info:")
        print(f"  Total duration: {duration:.2f} seconds")
        print(f"  Original FPS: {original_fps:.2f}")
        if effective_target_fps:
            print(f"  Target FPS: {effective_target_fps:.2f}")
            print(f"  FPS ratio: {effective_target_fps/original_fps:.3f}")
            print(f"  Frame sampling: every {original_fps/effective_target_fps:.1f} frames")
        else:
            print(f"  Sampling method: Uniform")
        
        results = []
        current_time = 0.0
        clip_index = 0
        
        step = clip_duration - overlap
        
        while current_time + clip_duration <= duration:
            try:
                frames, sampling_info = self.load_video_frames(
                    video_path, current_time, clip_duration, effective_target_fps
                )
                """
                print(f"  Clip {clip_index}: frames shape {frames.shape}, "
                      f"actual FPS: {sampling_info['actual_fps']:.2f}")
                """
                clip_features = self.extract_clip_features(frames)
                global_features = self.extract_global_features(frames)
                
                result = {
                    'clip_index': clip_index,
                    'start_time': current_time,
                    'end_time': current_time + clip_duration,
                    'clip_features': clip_features,
                    'global_features': global_features,
                    'feature_shape': clip_features.shape,
                    'video_path': video_path,
                    'sampling_info': sampling_info
                }
                
                results.append(result)
                
#                print(f"  Clip {clip_index}: {current_time:.2f}s - {current_time + clip_duration:.2f}s")
                
                current_time += step
                clip_index += 1
                
            except Exception as e:
                print(f"  Error (clip {clip_index}): {e}")
                current_time += step
                clip_index += 1
                continue
        
        print(f"Feature extraction completed: {len(results)} clips")
        return results

def main():
    """Usage example with FPS control"""
    
    # Path to trained model
    model_path = "videomae_pretrained.pth"
    
    # Test different FPS settings
    fps_configs = [
        None,    # Original uniform sampling
        5.0,     # 5 FPS
        10.0,    # 10 FPS
        15.0,    # 15 FPS
    ]
    
    try:
        video_dir = "/data/MICCAI2025_OSS/task1_2/videos"
        if os.path.exists(video_dir):
            video_files = list(Path(video_dir).glob("*.mp4"))[:1]
            
            if video_files:
                video_path = str(video_files[0])
                print(f"Test video: {video_path}")
                
                for target_fps in fps_configs:
                    print(f"\n{'='*60}")
                    print(f"Testing with target FPS: {target_fps}")
                    print('='*60)
                    
                    # Initialize feature extractor with specific FPS
                    extractor = VideoMAEFeatureExtractor(
                        model_path, 
                        device='auto',
                        target_fps=target_fps
                    )
                    
                    # Run feature extraction
                    results = extractor.extract_features_from_video(
                        video_path,
                        clip_duration=2.0,
                        overlap=0.5
                    )
                    
                    if results:
                        print(f"\n=== Extraction Results ===")
                        print(f"Total number of clips: {len(results)}")
                        print(f"Feature shape: {results[0]['feature_shape']}")
                        print(f"Global feature shape: {results[0]['global_features'].shape}")
                        
                        # Show sampling info for first clip
                        sampling_info = results[0]['sampling_info']
                        print(f"\nSampling info (first clip):")
                        for key, value in sampling_info.items():
                            if isinstance(value, float):
                                print(f"  {key}: {value:.3f}")
                            elif isinstance(value, list):
                                print(f"  {key}: [showing first 5] {value[:5]}")
                            else:
                                print(f"  {key}: {value}")
                        
                        # Save features
                        fps_suffix = f"_fps{target_fps}" if target_fps else "_uniform"
                        output_path = f"extracted_video_features{fps_suffix}.pth"
                        torch.save(results, output_path)
                        print(f"Features saved: {output_path}")
                    else:
                        print("Feature extraction failed")
            else:
                print("Test video file not found")
        else:
            print("Video directory not found")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

