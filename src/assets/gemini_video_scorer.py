#!/usr/bin/env python3
"""
Video Character Authenticity Scorer using Gemini API
Evaluates generated videos on 4 aspects:
1. Identity Preservation
2. Character-Faithful Motion
3. Style Preservation
4. Multi-Character Interaction Naturalness
"""

import os
import json
import time
import base64
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import google.generativeai as genai
from tqdm import tqdm
import cv2
import tempfile

# Configuration for scoring criteria
SCORING_CRITERIA = {
    "identity_preservation": {
        "description": "How well the character's visual identity is preserved",
        "prompt": """Score from 1-10 how well this video preserves the character's identity.
        Consider:
        - Facial features consistency
        - Body proportions
        - Distinctive characteristics (e.g., Jerry's mouse ears, Tom's cat features)
        - Color scheme accuracy
        10 = Perfect identity preservation
        1 = Character unrecognizable""",
        "weight": 0.3
    },
    "character_faithful_motion": {
        "description": "How authentic the character's movements and behaviors are",
        "prompt": """Score from 1-10 how authentic the character's motion/behavior is.
        Consider:
        - Movement patterns typical of the character
        - Behavioral traits (e.g., Jerry's quick scurrying, Tom's sneaky movements)
        - Personality expression through motion
        - Consistency with character's known mannerisms
        10 = Perfectly authentic motion
        1 = Completely out of character""",
        "weight": 0.3
    },
    "style_preservation": {
        "description": "How well the original artistic style is maintained",
        "prompt": """Score from 1-10 how well the original artistic style is preserved.
        Consider:
        - Animation style (cartoon, realistic, etc.)
        - Visual aesthetics matching source material
        - Art direction consistency
        - Rendering quality appropriate to style
        10 = Perfect style match
        1 = Completely different style""",
        "weight": 0.25
    },
    "interaction_naturalness": {
        "description": "For multi-character scenes, how natural the interactions are",
        "prompt": """Score from 1-10 how natural the character interactions are.
        If single character, score based on interaction with environment.
        Consider:
        - Spatial relationships between characters
        - Timing and coordination of interactions
        - Believable reactions and responses
        - Physical plausibility of interactions
        10 = Perfectly natural interactions
        1 = Unnatural or broken interactions""",
        "weight": 0.15
    }
}

class GeminiVideoScorer:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """Initialize Gemini API client"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.generation_config = {
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
        }

    def extract_frames(self, video_path: str, num_frames: int = 8) -> List[str]:
        """Extract frames from video for analysis"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            raise ValueError(f"Cannot read video: {video_path}")

        # Sample frames evenly throughout the video
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        frames_base64 = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert frame to base64
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                frames_base64.append(frame_base64)

        cap.release()
        return frames_base64

    def upload_video_to_gemini(self, video_path: str) -> Optional[str]:
        """Upload video file to Gemini for processing"""
        try:
            video_file = genai.upload_file(video_path)

            # Wait for file to be processed
            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = genai.get_file(video_file.name)

            if video_file.state.name == "FAILED":
                raise ValueError(f"Video processing failed: {video_file.state.name}")

            return video_file
        except Exception as e:
            print(f"Error uploading video: {e}")
            return None

    def score_video_aspect(self, video_file, aspect_name: str, aspect_info: dict,
                          character_name: str, is_multi: bool = False) -> Dict:
        """Score a single aspect of the video"""

        # Build the prompt
        prompt = f"""You are evaluating a generated video of {character_name}.

        {aspect_info['prompt']}

        {'Note: This is a multi-character video.' if is_multi else 'Note: This is a single-character video.'}

        Provide your response in the following JSON format:
        {{
            "score": <integer from 1-10>,
            "reasoning": "<brief explanation of the score>",
            "strengths": ["<strength 1>", "<strength 2>"],
            "weaknesses": ["<weakness 1>", "<weakness 2>"]
        }}
        """

        try:
            response = self.model.generate_content(
                [video_file, prompt],
                generation_config=self.generation_config
            )

            # Parse JSON response
            response_text = response.text.strip()
            # Extract JSON from response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            result = json.loads(response_text)
            result['aspect'] = aspect_name
            return result

        except Exception as e:
            print(f"Error scoring {aspect_name}: {e}")
            return {
                "aspect": aspect_name,
                "score": 0,
                "reasoning": f"Error: {str(e)}",
                "strengths": [],
                "weaknesses": []
            }

    def score_video(self, video_path: str, character_name: str,
                   is_multi: bool = False) -> Dict:
        """Score a video across all aspects"""

        video_path = Path(video_path)
        if not video_path.exists():
            return {"error": f"Video not found: {video_path}"}

        print(f"Processing: {video_path.name}")

        # Upload video to Gemini
        video_file = self.upload_video_to_gemini(str(video_path))
        if not video_file:
            return {"error": "Failed to upload video"}

        # Score each aspect
        scores = {}
        for aspect_name, aspect_info in SCORING_CRITERIA.items():
            # Skip interaction scoring for single character if not relevant
            if aspect_name == "interaction_naturalness" and not is_multi:
                aspect_info = aspect_info.copy()
                aspect_info['prompt'] = aspect_info['prompt'].replace(
                    "how natural the character interactions are",
                    "how natural the character's interaction with the environment is"
                )

            score_result = self.score_video_aspect(
                video_file, aspect_name, aspect_info, character_name, is_multi
            )
            scores[aspect_name] = score_result

            # Rate limiting
            time.sleep(2)

        # Calculate weighted overall score
        total_score = 0
        total_weight = 0
        for aspect_name, score_data in scores.items():
            if 'score' in score_data and score_data['score'] > 0:
                weight = SCORING_CRITERIA[aspect_name]['weight']
                total_score += score_data['score'] * weight
                total_weight += weight

        overall_score = total_score / total_weight if total_weight > 0 else 0

        # Delete uploaded file to save storage
        try:
            genai.delete_file(video_file.name)
        except:
            pass

        return {
            "video_path": str(video_path),
            "character": character_name,
            "is_multi": is_multi,
            "scores": scores,
            "overall_score": overall_score,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

def parse_character_from_filename(filename: str) -> Tuple[str, bool]:
    """Extract character name and detect if multi-character from filename"""
    filename_lower = filename.lower()

    # Known character patterns
    characters = {
        'tom': 'Tom (Tom and Jerry)',
        'jerry': 'Jerry (Tom and Jerry)',
        'mrbean': 'Mr. Bean',
        'sheldon': 'Sheldon Cooper',
        'penny': 'Penny',
        'george': 'George',
        'grizzly': 'Grizzly Bear',
        'icebear': 'Ice Bear',
        'panda': 'Panda Bear',
        'marry': 'Mary'
    }

    detected_chars = []
    for char_key, char_name in characters.items():
        if char_key in filename_lower:
            detected_chars.append(char_name)

    # Check if multi-character
    is_multi = len(detected_chars) > 1 or 'multi' in filename_lower or 'interaction' in filename_lower

    # Format character name(s)
    if detected_chars:
        character_name = ' & '.join(detected_chars) if is_multi else detected_chars[0]
    else:
        character_name = "Unknown Character"

    return character_name, is_multi

def main():
    parser = argparse.ArgumentParser(description="Score videos using Gemini API")
    parser.add_argument("--api_key", type=str, required=True, help="Gemini API key")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing videos")
    parser.add_argument("--output_dir", type=str, default="./vlm_metric/results",
                       help="Output directory for results")
    parser.add_argument("--method_name", type=str, required=True,
                       help="Name of the method being evaluated")
    parser.add_argument("--batch_size", type=int, default=5,
                       help="Number of videos to process before saving")

    args = parser.parse_args()

    # Initialize scorer
    scorer = GeminiVideoScorer(args.api_key)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all video files
    video_dir = Path(args.video_dir)
    video_files = list(video_dir.glob("*.mp4"))

    # Sort for consistent ordering
    video_files = sorted(video_files)

    print(f"Found {len(video_files)} videos to process")

    # Process videos
    results = []
    for i, video_file in enumerate(tqdm(video_files, desc="Scoring videos")):
        # Extract character info from filename
        character_name, is_multi = parse_character_from_filename(video_file.name)

        # Score video
        result = scorer.score_video(video_file, character_name, is_multi)
        result['method'] = args.method_name
        results.append(result)

        # Save intermediate results
        if (i + 1) % args.batch_size == 0:
            output_file = output_dir / f"{args.method_name}_batch_{i//args.batch_size}.json"
            with open(output_file, 'w') as f:
                json.dump(results[-args.batch_size:], f, indent=2)
            print(f"Saved batch to {output_file}")

    # Save final results
    output_file = output_dir / f"{args.method_name}_all_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nCompleted! Results saved to {output_file}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for aspect in SCORING_CRITERIA.keys():
        scores = [r['scores'][aspect]['score'] for r in results
                 if aspect in r.get('scores', {}) and r['scores'][aspect]['score'] > 0]
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"{aspect}: {avg_score:.2f}")

    overall_scores = [r['overall_score'] for r in results if r.get('overall_score', 0) > 0]
    if overall_scores:
        print(f"\nOverall Average: {sum(overall_scores)/len(overall_scores):.2f}")

if __name__ == "__main__":
    main()