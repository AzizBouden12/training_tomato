import json
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from train_tomato_models import TARGET_CLASSES, build_transfer_model, make_transforms


CLASS_DETAILS = {
    "Tomato___healthy": {
        "display_name": "Tomato healthy",
        "emoji": "🟢",
        "description": "Feuille saine. Aucune intervention necessaire.",
    },
    "Tomato___Early_blight": {
        "display_name": "Tomato Early blight",
        "emoji": "🟡",
        "description": "Alternariose precoce. Surveiller et traiter rapidement.",
    },
    "Tomato___Late_blight": {
        "display_name": "Tomato Late blight",
        "emoji": "🔴",
        "description": "Mildiou tardif. Cas plus critique, isoler la plante.",
    },
    "Tomato___Leaf_Mold": {
        "display_name": "Tomato Leaf Mold",
        "emoji": "🟠",
        "description": "Moisissure foliaire. Ameliorer l'aeration et surveiller.",
    },
}


@dataclass
class PredictionResult:
    class_name: str
    display_name: str
    confidence: float
    probabilities: dict[str, float]
    emoji: str
    description: str


class TomatoDiseasePredictor:
    def __init__(self, artifact_dir: Path | str = "artifacts_final", image_size: int = 128) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.model_dir = self.artifact_dir / "mobilenet_v3_transfer"
        self.model_path = self.model_dir / "model.pt"
        self.summary_path = self.artifact_dir / "summary.json"
        if not self.model_path.exists():
            raise FileNotFoundError(f"model not found: {self.model_path}")
        if not self.summary_path.exists():
            raise FileNotFoundError(f"summary not found: {self.summary_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_transfer_model(num_classes=len(TARGET_CLASSES))
        state = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        _, self.eval_transform = make_transforms(image_size)
        self.summary = json.loads(self.summary_path.read_text(encoding="utf-8"))

    def predict_image(self, image: Image.Image) -> PredictionResult:
        rgb = image.convert("RGB")
        tensor = self.eval_transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probas = torch.softmax(logits, dim=1).cpu().numpy()[0]

        best_idx = int(probas.argmax())
        class_name = TARGET_CLASSES[best_idx]
        meta = CLASS_DETAILS[class_name]
        probabilities = {
            CLASS_DETAILS[name]["display_name"]: float(probas[idx])
            for idx, name in enumerate(TARGET_CLASSES)
        }
        return PredictionResult(
            class_name=class_name,
            display_name=meta["display_name"],
            confidence=float(probas[best_idx]),
            probabilities=probabilities,
            emoji=meta["emoji"],
            description=meta["description"],
        )

    def predict_path(self, image_path: Path | str) -> PredictionResult:
        with Image.open(image_path) as image:
            return self.predict_image(image)

    def metric_summary(self) -> dict[str, float]:
        metrics = self.summary["mobilenet_v3_transfer"]
        return {
            "accuracy": float(metrics["test_accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "weighted_f1": float(metrics["weighted_f1"]),
        }
