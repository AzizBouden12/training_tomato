import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from tomato_inference import TomatoDiseasePredictor


class TomatoInterfaceApp:
    def __init__(self, root: tk.Tk, artifact_dir: Path) -> None:
        self.root = root
        self.root.title("Tomato Disease Classifier")
        self.root.geometry("980x640")
        self.root.minsize(900, 560)

        self.predictor = TomatoDiseasePredictor(artifact_dir=artifact_dir)
        self.selected_path: Path | None = None
        self.preview_image = None

        self.metrics = self.predictor.metric_summary()
        self._build_layout()

    def _build_layout(self) -> None:
        self.root.configure(bg="#f3f8f3")

        header = tk.Frame(self.root, bg="#226b45", padx=18, pady=16)
        header.pack(fill="x")
        tk.Label(
            header,
            text="Tomato Disease Classifier",
            font=("Segoe UI", 20, "bold"),
            fg="white",
            bg="#226b45",
        ).pack(anchor="w")
        tk.Label(
            header,
            text=(
                f"MobileNetV3 Transfer  |  Accuracy {self.metrics['accuracy']:.2%}  |  "
                f"Weighted F1 {self.metrics['weighted_f1']:.2%}"
            ),
            font=("Segoe UI", 11),
            fg="#d9f0e1",
            bg="#226b45",
        ).pack(anchor="w", pady=(4, 0))

        body = tk.Frame(self.root, bg="#f3f8f3", padx=16, pady=16)
        body.pack(fill="both", expand=True)

        left = tk.Frame(body, bg="white", highlightbackground="#d8e6da", highlightthickness=1)
        left.pack(side="left", fill="y")
        right = tk.Frame(body, bg="white", highlightbackground="#d8e6da", highlightthickness=1)
        right.pack(side="left", fill="both", expand=True, padx=(16, 0))

        self.image_label = tk.Label(
            left,
            text="No image selected",
            width=42,
            height=22,
            bg="#eef5ef",
            fg="#67806e",
            font=("Segoe UI", 11),
            compound="top",
        )
        self.image_label.pack(padx=16, pady=16)

        controls = tk.Frame(left, bg="white")
        controls.pack(fill="x", padx=16, pady=(0, 16))

        tk.Button(
            controls,
            text="Choose image",
            command=self.choose_image,
            bg="#226b45",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            padx=10,
            pady=8,
        ).pack(fill="x")

        self.model_choice = ttk.Combobox(
            controls,
            values=["MobileNetV3 Transfer (validated)"],
            state="readonly",
            font=("Segoe UI", 10),
        )
        self.model_choice.current(0)
        self.model_choice.pack(fill="x", pady=(12, 8))

        tk.Button(
            controls,
            text="Analyze",
            command=self.analyze,
            bg="#2f8f5b",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            padx=10,
            pady=8,
        ).pack(fill="x")

        tk.Button(
            controls,
            text="Reset",
            command=self.reset,
            bg="#6c757d",
            fg="white",
            font=("Segoe UI", 10),
            relief="flat",
            padx=10,
            pady=6,
        ).pack(fill="x", pady=(8, 0))

        self.file_label = tk.Label(
            controls,
            text="No file loaded",
            wraplength=260,
            justify="left",
            bg="white",
            fg="#506257",
            font=("Segoe UI", 9),
        )
        self.file_label.pack(fill="x", pady=(12, 0))

        title = tk.Label(
            right,
            text="Prediction",
            font=("Segoe UI", 18, "bold"),
            bg="white",
            fg="#224233",
        )
        title.pack(anchor="w", padx=20, pady=(20, 8))

        self.result_title = tk.Label(
            right,
            text="Load a leaf image and click Analyze",
            font=("Segoe UI", 15, "bold"),
            bg="white",
            fg="#355747",
        )
        self.result_title.pack(anchor="w", padx=20)

        self.result_desc = tk.Label(
            right,
            text="",
            font=("Segoe UI", 11),
            bg="white",
            fg="#5b6d62",
            wraplength=520,
            justify="left",
        )
        self.result_desc.pack(anchor="w", padx=20, pady=(10, 14))

        self.confidence_label = tk.Label(
            right,
            text="",
            font=("Segoe UI", 22, "bold"),
            bg="white",
            fg="#226b45",
        )
        self.confidence_label.pack(anchor="w", padx=20, pady=(0, 10))

        prob_frame = tk.Frame(right, bg="white")
        prob_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        self.prob_frame = prob_frame

    def choose_image(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select a plant leaf image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")],
        )
        if not filename:
            return
        self.selected_path = Path(filename)
        self.file_label.config(text=str(self.selected_path))
        self._update_preview(self.selected_path)

    def _update_preview(self, path: Path) -> None:
        image = Image.open(path).convert("RGB")
        preview = image.copy()
        preview.thumbnail((320, 320))
        self.preview_image = ImageTk.PhotoImage(preview)
        self.image_label.config(image=self.preview_image, text="")

    def analyze(self) -> None:
        if self.selected_path is None:
            messagebox.showwarning("Missing image", "Choose an image first.")
            return
        try:
            result = self.predictor.predict_path(self.selected_path)
        except Exception as exc:
            messagebox.showerror("Prediction error", str(exc))
            return

        self.result_title.config(text=f"{result.emoji}  {result.display_name}")
        self.result_desc.config(text=result.description)
        self.confidence_label.config(text=f"Confidence: {result.confidence:.1%}")
        for child in self.prob_frame.winfo_children():
            child.destroy()

        for class_name, prob in sorted(result.probabilities.items(), key=lambda item: item[1], reverse=True):
            row = tk.Frame(self.prob_frame, bg="white")
            row.pack(fill="x", pady=6)
            tk.Label(
                row,
                text=class_name,
                font=("Segoe UI", 10),
                bg="white",
                fg="#30483b",
                width=24,
                anchor="w",
            ).pack(side="left")
            bar = ttk.Progressbar(row, maximum=100, value=prob * 100, length=240)
            bar.pack(side="left", padx=(8, 8))
            tk.Label(
                row,
                text=f"{prob:.1%}",
                font=("Segoe UI", 10, "bold"),
                bg="white",
                fg="#226b45",
                width=8,
                anchor="e",
            ).pack(side="left")

    def reset(self) -> None:
        self.selected_path = None
        self.preview_image = None
        self.image_label.config(image="", text="No image selected")
        self.file_label.config(text="No file loaded")
        self.result_title.config(text="Load a leaf image and click Analyze")
        self.result_desc.config(text="")
        self.confidence_label.config(text="")
        for child in self.prob_frame.winfo_children():
            child.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the tomato disease GUI.")
    parser.add_argument("--artifact-dir", default="artifacts_final")
    parser.add_argument("--smoke-test", type=str, help="Run one prediction on an image and print the result.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = TomatoDiseasePredictor(artifact_dir=args.artifact_dir)
    if args.smoke_test:
        result = predictor.predict_path(args.smoke_test)
        print(f"class={result.class_name}")
        print(f"display={result.display_name}")
        print(f"confidence={result.confidence:.6f}")
        for class_name, prob in sorted(result.probabilities.items(), key=lambda item: item[1], reverse=True):
            print(f"{class_name}: {prob:.6f}")
        return

    root = tk.Tk()
    app = TomatoInterfaceApp(root=root, artifact_dir=Path(args.artifact_dir))
    root.mainloop()


if __name__ == "__main__":
    main()
