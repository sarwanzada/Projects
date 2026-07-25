import cv2
import tkinter as tk
from ttkbootstrap import Style
from tkinter import ttk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import pyttsx3
import datetime
import speech_recognition as sr
from googletrans import Translator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import json
import os
import mediapipe as mp

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Translator")
        self.root.geometry("1200x800")
        self.theme = "flatly"
        self.style = Style(self.theme)

        self.engine = pyttsx3.init()
        self.cap = None
        self.last_frame = None
        self.translation_text = tk.StringVar(value="")
        self.translation_history = []
        self.selected_lang = tk.StringVar(value='en')
        self.translator = Translator()
        self.auto_speak_enabled = False

        if not os.path.exists("asl_mlp_final_trained.keras"):
            messagebox.showerror("Error", "Model file 'asl_mlp_final_trained.keras' not found!")
            self.root.quit()
            return

        self.model = Sequential([
            Dense(128, activation='relu', input_shape=(63,)),
            Dropout(0.4),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(28, activation='softmax')
        ])
        self.model.load_weights("asl_weights.h5")

        with open("class_indices_dataset1.json", "r") as f:
            class_indices = json.load(f)
        self.labels = [label for index, label in sorted(class_indices.items(), key=lambda x: int(x[0]))]

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

        self.start_screen()

    def start_screen(self):
        self.clear_screen()
        self.root.configure(bg="#3399ff")
        frame = ttk.Frame(self.root, padding=50)
        frame.place(relx=0.5, rely=0.5, anchor="center")

        ttk.Label(frame, text="Sign Language Translator",
                  font=("Segoe UI", 26, "bold")).pack(pady=10)
        ttk.Label(frame, text="Click below to begin",
                  font=("Segoe UI", 14)).pack(pady=5)
        ttk.Button(frame, text="Start App",
                   command=self.ask_permission,
                   bootstyle="purple").pack(pady=20)

    def ask_permission(self):
        if messagebox.askyesno("Camera Access", "This app requires webcam. Allow?"):
            self.setup_ui()
            self.cap = cv2.VideoCapture(0)
            self.show_frame()
        else:
            self.root.quit()

    def setup_ui(self):
        self.clear_screen()
        self.root.configure(bg="SystemButtonFace")

        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill="both", expand=True)

        top_bar = ttk.Frame(main_frame)
        top_bar.pack(fill="x", pady=(0, 10))
        ttk.Label(top_bar, text="Powered by Google Translate").pack(side="left", padx=10)
        ttk.Button(top_bar, text="Toggle Theme",
                   command=self.toggle_theme).pack(side="right", padx=10)

        body_frame = ttk.Frame(main_frame)
        body_frame.pack(fill="both", expand=True)

        left_frame = ttk.Frame(body_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=10)

        self.video_label = ttk.Label(left_frame)
        self.video_label.pack(pady=10)

        ttk.Label(left_frame, text="Translation Output",
                  font=("Segoe UI", 14, "bold")).pack()
        self.translation_display = ttk.Entry(
            left_frame, textvariable=self.translation_text,
            font=("Segoe UI", 13), width=70, justify="center")
        self.translation_display.pack(pady=10)

        ttk.Label(left_frame, text="Select Language:",
                  font=("Segoe UI", 12)).pack()
        ttk.Combobox(
            left_frame, textvariable=self.selected_lang,
            values=["en", "ur", "fr", "es", "zh-cn", "ar"],
            width=20).pack(pady=5)

        right_frame = ttk.Frame(body_frame)
        right_frame.pack(side="right", fill="y", padx=20)

        ttk.Button(right_frame, text="Capture",
                   command=self.capture_frame, bootstyle="info").pack(pady=10, fill="x")
        ttk.Button(right_frame, text="Speak",
                   command=self.speak_translation, bootstyle="warning").pack(pady=10, fill="x")
        ttk.Button(right_frame, text="Voice Input",
                   command=self.voice_input, bootstyle="primary").pack(pady=10, fill="x")
        ttk.Button(right_frame, text="History",
                   command=self.show_history, bootstyle="light").pack(pady=10, fill="x")
        ttk.Button(right_frame, text="Toggle Auto-Speak",
                   command=self.toggle_auto_speak, bootstyle="secondary").pack(pady=10, fill="x")
        ttk.Button(right_frame, text="Exit",
                   command=self.exit_app, bootstyle="danger").pack(pady=10, fill="x")

    def show_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.last_frame = frame
                frame_flipped = cv2.flip(frame, 1)
                frame_flipped = cv2.resize(frame_flipped, (900, 500))

                img_rgb = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
                results = self.hands.process(img_rgb)

                predicted_letter = ""
                if results.multi_hand_landmarks:
                    landmarks = results.multi_hand_landmarks[0].landmark
                    features = np.array([coord for lm in landmarks for coord in (lm.x, lm.y, lm.z)]).reshape(-1, 3)
                    wrist = features[0]
                    features -= wrist
                    max_val = np.max(np.linalg.norm(features, axis=1))
                    if max_val > 0:
                        features /= max_val
                    features = features.flatten().reshape(1, -1)
                    if features.shape[1] == 63:
                        preds = self.model.predict(features, verbose=0)
                        class_idx = preds.argmax(axis=1)[0]
                        predicted_letter = self.labels[class_idx]

                translated = self.translate_text(predicted_letter, self.selected_lang.get()) if predicted_letter else ""
                self.translation_text.set(translated)

                if translated:
                    self.translation_history.append(translated)

                if self.auto_speak_enabled and translated:
                    self.engine.say(translated)
                    self.engine.runAndWait()

                img_pil = Image.fromarray(cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img_pil)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            self.root.after(20, self.show_frame)

    def translate_text(self, text, lang):
        try:
            return self.translator.translate(text, dest=lang).text
        except:
            return text

    def capture_frame(self):
        if self.last_frame is not None:
            name = f"capture_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(name, self.last_frame)
            messagebox.showinfo("Image Saved", f"Saved as {name}")
        else:
            messagebox.showwarning("No Frame", "Nothing to save")

    def speak_translation(self):
        text = self.translation_text.get()
        if text:
            self.engine.say(text)
            self.engine.runAndWait()
        else:
            messagebox.showinfo("No Text", "Nothing to speak")

    def voice_input(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            messagebox.showinfo("Voice Input", "Start speaking...")
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio)
                self.translation_text.set(text)
                self.translation_history.append(text)
            except sr.UnknownValueError:
                messagebox.showwarning("Voice Error", "Sorry, could not understand")
            except sr.RequestError:
                messagebox.showerror("API Error", "Could not reach speech API")

    def show_history(self):
        win = tk.Toplevel(self.root)
        win.title("Translation History")
        win.geometry("400x300")
        ttk.Label(win, text="Your Translations",
                  font=("Segoe UI", 13, "bold")).pack(pady=10)
        text_area = tk.Text(win, font=("Segoe UI", 11))
        text_area.pack(expand=True, fill="both", padx=10, pady=10)
        for entry in self.translation_history:
            text_area.insert(tk.END, entry + "\n")
        ttk.Button(win, text="Close",
                   command=win.destroy).pack(pady=10)

    def toggle_theme(self):
        self.theme = "darkly" if self.theme == "flatly" else "flatly"
        self.style = Style(self.theme)
        self.setup_ui()

    def toggle_auto_speak(self):
        self.auto_speak_enabled = not self.auto_speak_enabled
        state = "enabled" if self.auto_speak_enabled else "disabled"
        messagebox.showinfo("Auto-Speak Mode", f"Auto-speak {state}")

    def exit_app(self):
        if self.cap:
            self.cap.release()
        self.root.quit()

    def clear_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()