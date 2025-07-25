import customtkinter as ctk
import joblib
import os
import threading
import time

# === Appearance Configuration ===
ctk.set_appearance_mode("light")  # Choose between "light", "dark", "system"
ctk.set_default_color_theme("blue")

# === Load Model and Vectorizer ===
MODEL_PATH = r"C:\Users\gaura\OneDrive\Desktop\aiassignment\models\naive_bayes_model.joblib"
VECTORIZER_PATH = r"C:\Users\gaura\OneDrive\Desktop\aiassignment\models\tfidf_vectorizer.joblib"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# === Main App Class ===
class SpamClassifierApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Spam Email Classifier")
        self.state("zoomed")  # Fullscreen window
        self.resizable(True, True)

        # === Layout Container ===
        self.main_frame = ctk.CTkFrame(self, corner_radius=20)
        self.main_frame.pack(padx=60, pady=40, fill="both", expand=True)

        # === Title ===
        self.title_label = ctk.CTkLabel(self.main_frame,
                                        text="📨 Spam Email Classifier",
                                        font=("Helvetica", 36, "bold"))
        self.title_label.pack(pady=(20, 10))

        # === Description Text ===
        self.description_label = ctk.CTkLabel(
            self.main_frame,
            text="This tool uses Machine Learning to identify whether a message is SPAM or NOT SPAM (Ham).\n"
                 "It analyzes your input text and shows not just the result but also why it was classified that way.",
            font=("Helvetica", 16),
            wraplength=1000,
            justify="center",
            text_color="#333"
        )
        self.description_label.pack(pady=10)

        # === Input Box ===
        self.input_label = ctk.CTkLabel(self.main_frame, text="📥 Enter Email or Message:", font=("Helvetica", 18, "bold"))
        self.input_label.pack(pady=(30, 10))

        self.input_textbox = ctk.CTkTextbox(self.main_frame, height=150, font=("Helvetica", 16), corner_radius=10)
        self.input_textbox.pack(padx=40, pady=10, fill="x")

        # === Action Buttons ===
        self.button_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.button_frame.pack(pady=15)

        self.classify_button = ctk.CTkButton(self.button_frame, text="🔍 Classify", width=160,
                                             font=("Helvetica", 16, "bold"), command=self.animate_classification)
        self.classify_button.grid(row=0, column=0, padx=15)

        self.clear_button = ctk.CTkButton(self.button_frame, text="🧹 Clear", width=120,
                                          font=("Helvetica", 16), command=self.clear_all)
        self.clear_button.grid(row=0, column=1, padx=15)

        # === Output Result ===
        self.output_label = ctk.CTkLabel(self.main_frame, text="", font=("Helvetica", 22, "bold"))
        self.output_label.pack(pady=30)

        self.explanation_label = ctk.CTkLabel(self.main_frame, text="", font=("Helvetica", 16), wraplength=1000,
                                              justify="center", text_color="#444")
        self.explanation_label.pack(pady=(0, 20))

        # === Footer ===
        self.footer = ctk.CTkLabel(self.main_frame,
                                   text="Developed with ❤️ by Gaurav | Spam Detection using Machine Learning (Naive Bayes)",
                                   font=("Helvetica", 12), text_color="gray")
        self.footer.pack(side="bottom", pady=10)

    # === Classification Logic with Animation ===
    def animate_classification(self):
        threading.Thread(target=self._animate_and_classify).start()

    def _animate_and_classify(self):
        message = self.input_textbox.get("1.0", "end").strip()

        if not message:
            self.output_label.configure(text="⚠️ Please enter a message!", text_color="orange")
            self.explanation_label.configure(text="")
            return

        # Simulate typing animation
        self.output_label.configure(text="Classifying...", text_color="#0066cc")
        self.explanation_label.configure(text="")
        time.sleep(1.2)

        input_vector = vectorizer.transform([message])
        prediction = model.predict(input_vector)[0]
        probs = model.predict_proba(input_vector)[0]

        # Format probabilities for better understanding
        ham_prob = probs[1] * 100
        spam_prob = probs[0] * 100

        # Show prediction and explanation
        if prediction == 1:
            self.output_label.configure(text="✅ Prediction: Not Spam (Ham)", text_color="green")
            explanation = (
                f"🔍 This message is likely NOT spam based on its content and word patterns.\n"
                f"🟢 Confidence: {ham_prob:.2f}% Ham\n"
                f"🔴 Spam Probability: {spam_prob:.2f}%"
            )
        else:
            self.output_label.configure(text="🚫 Prediction: Spam", text_color="red")
            explanation = (
                f"⚠️ This message contains features commonly found in spam messages like promotions, links, or urgency.\n"
                f"🔴 Confidence: {spam_prob:.2f}% Spam\n"
                f"🟢 Ham Probability: {ham_prob:.2f}%"
            )

        self.explanation_label.configure(text=explanation)

    # === Clear Fields ===
    def clear_all(self):
        self.input_textbox.delete("1.0", "end")
        self.output_label.configure(text="")
        self.explanation_label.configure(text="")

# === Run App ===
if __name__ == "__main__":
    app = SpamClassifierApp()
    app.mainloop()
