"""Desktop gallery for evaluating labelled pet images with the trained model."""
import csv
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

import config
from inference import predict
from model import get_device, load_class_names, load_trained_model


class PetBreedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pet Breed Classifier")
        self.root.geometry("1180x820")
        self.root.minsize(820, 600)

        self.external_dir = config.EXTERNAL_DIR
        self.results = []
        self.events = queue.Queue()
        self.running = False

        self._build_ui()
        self.root.after(100, lambda: self.run_evaluation(self.external_dir))
        self.root.after(100, self._process_events)

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#17243a", padx=24, pady=18)
        header.pack(fill="x")
        tk.Label(
            header,
            text="Pet Breed Classifier",
            bg="#17243a",
            fg="white",
            font=("Segoe UI", 22, "bold"),
        ).pack(anchor="w")
        tk.Label(
            header,
            text="Visual evaluation of your labelled cat and dog photos",
            bg="#17243a",
            fg="#cbd5e1",
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(3, 0))

        controls = ttk.Frame(self.root, padding=(20, 12))
        controls.pack(fill="x")
        self.folder_label = ttk.Label(controls, text=f"Folder: {self.external_dir}")
        self.folder_label.pack(side="left", fill="x", expand=True)
        self.browse_button = ttk.Button(
            controls, text="Choose folder...", command=self.choose_folder
        )
        self.browse_button.pack(side="right", padx=(8, 0))
        self.run_button = ttk.Button(
            controls, text="Run evaluation", command=self.refresh
        )
        self.run_button.pack(side="right")

        summary = ttk.Frame(self.root, padding=(20, 0, 20, 8))
        summary.pack(fill="x")
        self.metric_labels = {}
        for column, (key, title) in enumerate(
            (("accuracy", "External accuracy"), ("correct", "Correct"), ("tested", "Images tested"))
        ):
            card = tk.Frame(summary, bg="#f1f5f9", padx=16, pady=10)
            card.grid(row=0, column=column, sticky="ew", padx=(0, 10))
            summary.columnconfigure(column, weight=1)
            tk.Label(
                card, text=title, bg="#f1f5f9", fg="#475569",
                font=("Segoe UI", 9),
            ).pack(anchor="w")
            value = tk.Label(
                card, text="—", bg="#f1f5f9", fg="#17243a",
                font=("Segoe UI", 18, "bold"),
            )
            value.pack(anchor="w", pady=(2, 0))
            self.metric_labels[key] = value

        status_bar = ttk.Frame(self.root, padding=(20, 4, 20, 8))
        status_bar.pack(fill="x")
        self.status_label = ttk.Label(status_bar, text="Ready")
        self.status_label.pack(side="left", fill="x", expand=True)
        self.filter_var = tk.StringVar(value="All images")
        self.filter_box = ttk.Combobox(
            status_bar,
            textvariable=self.filter_var,
            values=("All images", "Incorrect only"),
            state="readonly",
            width=16,
        )
        self.filter_box.pack(side="right")
        self.filter_box.bind("<<ComboboxSelected>>", lambda _event: self._render_gallery())

        gallery = ttk.Frame(self.root, padding=(20, 0, 20, 16))
        gallery.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(gallery, bg="#e9eef5", highlightthickness=0)
        scrollbar = ttk.Scrollbar(gallery, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.gallery_frame = tk.Frame(self.canvas, bg="#e9eef5", padx=10, pady=10)
        self.gallery_window = self.canvas.create_window(
            (0, 0), window=self.gallery_frame, anchor="nw"
        )
        self.gallery_frame.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfigure(self.gallery_window, width=event.width),
        )
        self.canvas.bind("<MouseWheel>", self._scroll_gallery)

    def _scroll_gallery(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def choose_folder(self):
        directory = filedialog.askdirectory(
            title="Choose the folder containing Cats and Dogs",
            initialdir=self.external_dir,
        )
        if directory:
            self.external_dir = Path(directory)
            self.folder_label.configure(text=f"Folder: {self.external_dir}")
            self.run_evaluation(self.external_dir)

    def refresh(self):
        self.run_evaluation(self.external_dir)

    def run_evaluation(self, external_dir):
        if self.running:
            return
        self.running = True
        self.run_button.configure(state="disabled")
        self.browse_button.configure(state="disabled")
        self.status_label.configure(text="Loading model and evaluating images...")
        self.results = []
        self._update_metrics()
        self._render_gallery()
        threading.Thread(
            target=self._evaluate, args=(Path(external_dir),), daemon=True
        ).start()

    def _evaluate(self, external_dir):
        try:
            class_names = load_class_names()
            name_lookup = {name.lower(): name for name in class_names}
            image_paths = []
            skipped = 0

            for folder_name in ("Cats", "Dogs"):
                folder = external_dir / folder_name
                if not folder.is_dir():
                    continue
                for image_path in sorted(folder.iterdir()):
                    if not image_path.is_file() or image_path.suffix.lower() not in config.VALID_EXTENSIONS:
                        continue
                    actual_key = image_path.stem.rsplit("_", 1)[0].lower()
                    if actual_key not in name_lookup:
                        skipped += 1
                        continue
                    image_paths.append((image_path, name_lookup[actual_key]))

            if not image_paths:
                raise FileNotFoundError(
                    f"No labelled images found in {external_dir}. Put supported images in "
                    "'Cats' or 'Dogs' subfolders and name each file Breed_number.ext."
                )

            device = get_device()
            model = load_trained_model(len(class_names), device=device)
            results = []
            total = len(image_paths)
            for index, (image_path, actual_breed) in enumerate(image_paths, start=1):
                predicted_breed, confidence = predict(
                    model, image_path, class_names, device, top_k=1
                )[0]
                results.append({
                    "filename": image_path.name,
                    "actual_breed": actual_breed,
                    "predicted_breed": predicted_breed,
                    "confidence": confidence,
                    "correct": actual_breed == predicted_breed,
                    "image_path": image_path,
                })
                self.events.put(("progress", index, total, image_path.name))

            config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            with config.EXTERNAL_PREDICTIONS_PATH.open(
                "w", newline="", encoding="utf-8"
            ) as output_file:
                writer = csv.DictWriter(
                    output_file,
                    fieldnames=("filename", "actual_breed", "predicted_breed", "correct"),
                )
                writer.writeheader()
                for result in results:
                    writer.writerow({key: result[key] for key in writer.fieldnames})

            self.events.put(("complete", results, skipped))
        except Exception as error:
            self.events.put(("error", str(error)))

    def _process_events(self):
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "progress":
                    _, current, total, filename = event
                    self.status_label.configure(
                        text=f"Evaluating {current} of {total}: {filename}"
                    )
                elif event[0] == "complete":
                    _, self.results, skipped = event
                    self.running = False
                    self.run_button.configure(state="normal")
                    self.browse_button.configure(state="normal")
                    correct = sum(result["correct"] for result in self.results)
                    note = f" | Skipped {skipped} unrecognized filename(s)" if skipped else ""
                    self.status_label.configure(
                        text=f"Finished. CSV saved to {config.EXTERNAL_PREDICTIONS_PATH}{note}"
                    )
                    self._update_metrics()
                    self._render_gallery()
                elif event[0] == "error":
                    _, error_message = event
                    self.running = False
                    self.run_button.configure(state="normal")
                    self.browse_button.configure(state="normal")
                    self.status_label.configure(text="Evaluation could not be completed.")
                    messagebox.showerror("Evaluation failed", error_message, parent=self.root)
        except queue.Empty:
            pass
        self.root.after(100, self._process_events)

    def _update_metrics(self):
        if not self.results:
            for label in self.metric_labels.values():
                label.configure(text="—")
            return
        correct = sum(result["correct"] for result in self.results)
        accuracy = correct / len(self.results) * 100
        self.metric_labels["accuracy"].configure(text=f"{accuracy:.2f}%")
        self.metric_labels["correct"].configure(text=f"{correct} / {len(self.results)}")
        self.metric_labels["tested"].configure(text=str(len(self.results)))

    def _render_gallery(self):
        for child in self.gallery_frame.winfo_children():
            child.destroy()

        visible_results = self.results
        if self.filter_var.get() == "Incorrect only":
            visible_results = [result for result in self.results if not result["correct"]]
        if not visible_results:
            text = "No results yet. Start an evaluation to see your images."
            if self.results:
                text = "No incorrect predictions in this evaluation."
            tk.Label(
                self.gallery_frame,
                text=text,
                bg="#e9eef5",
                fg="#475569",
                font=("Segoe UI", 12),
                padx=20,
                pady=30,
            ).grid(row=0, column=0, sticky="w")
            return

        for index, result in enumerate(visible_results):
            row, column = divmod(index, 3)
            card = tk.Frame(
                self.gallery_frame,
                bg="white",
                padx=12,
                pady=12,
                highlightbackground="#d6deea",
                highlightthickness=1,
            )
            card.grid(row=row, column=column, sticky="nsew", padx=8, pady=8)
            try:
                with Image.open(result["image_path"]) as source:
                    preview = source.convert("RGB")
                    preview.thumbnail((300, 180))
                    photo = ImageTk.PhotoImage(preview)
            except OSError as error:
                tk.Label(
                    card, text=f"Could not display image:\n{error}",
                    bg="white", fg="#b42318", wraplength=300,
                ).pack(fill="x", pady=12)
            else:
                image_label = tk.Label(card, image=photo, bg="white")
                image_label.image = photo
                image_label.pack(pady=(0, 8))

            correct = result["correct"]
            status_color = "#137547" if correct else "#b42318"
            status_text = "CORRECT" if correct else "INCORRECT"
            tk.Label(
                card, text=result["filename"], bg="white", fg="#17243a",
                font=("Segoe UI", 10, "bold"), anchor="w",
            ).pack(fill="x")
            tk.Label(
                card, text=f"Actual: {result['actual_breed']}", bg="white",
                fg="#475569", anchor="w",
            ).pack(fill="x", pady=(5, 0))
            tk.Label(
                card, text=f"Predicted: {result['predicted_breed']}",
                bg="white", fg=status_color, font=("Segoe UI", 10, "bold"),
                anchor="w",
            ).pack(fill="x", pady=(2, 0))
            tk.Label(
                card,
                text=f"Model confidence (predicted label): {result['confidence'] * 100:.2f}%  |  {status_text}",
                bg="white", fg=status_color, anchor="w", wraplength=330,
            ).pack(fill="x", pady=(4, 0))

        for column in range(3):
            self.gallery_frame.columnconfigure(column, weight=1)


def main():
    root = tk.Tk()
    PetBreedApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
