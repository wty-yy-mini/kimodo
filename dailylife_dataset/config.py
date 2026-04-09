from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "dailylife_data"

# Generation requirements:
# seed=1, model=Kimodo-G1-RP-v1, denoising_steps=1000, num_samples=5, duration=5s
MODEL_NAME = "Kimodo-G1-RP-v1"
SEED = 1
DENOISING_STEPS = 1000
NUM_SAMPLES_PER_PROMPT = 5
DURATION_SECONDS = 5.0
NUM_TRANSITION_FRAMES = 5

# Dataset coverage by keyword (explicitly listed here for easy management).
KEYWORDS_BY_CATEGORY = {
    "locomotion": ["walk", "run", "jump"],
    "transition": ["stand", "turn", "sit", "squat"],
    "load": ["lift", "push", "throw"],
}

KEYWORD_TO_PROMPTS = {
    "walk": DATA_ROOT / "locomotion" / "walk" / "prompts.txt",
    "run": DATA_ROOT / "locomotion" / "run" / "prompts.txt",
    "jump": DATA_ROOT / "locomotion" / "jump" / "prompts.txt",
    "stand": DATA_ROOT / "transition" / "stand" / "prompts.txt",
    "turn": DATA_ROOT / "transition" / "turn" / "prompts.txt",
    "sit": DATA_ROOT / "transition" / "sit" / "prompts.txt",
    "squat": DATA_ROOT / "transition" / "squat" / "prompts.txt",
    "lift": DATA_ROOT / "load" / "lift" / "prompts.txt",
    "push": DATA_ROOT / "load" / "push" / "prompts.txt",
    "throw": DATA_ROOT / "load" / "throw" / "prompts.txt",
}

DEFAULT_KEYWORDS = [
    "walk",
    "run",
    "jump",
    "stand",
    "turn",
    "sit",
    "squat",
    "lift",
    "push",
    "throw",
]

# Output file name in each prompt folder: "{prompt_line}_{sample_id}.npz"
NPZ_NAME_PATTERN = "{prompt_id}_{sample_id}.npz"
