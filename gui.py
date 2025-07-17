import customtkinter as ctk
import joblib
import os

# Set appearance
ctk.set_appearance_mode("light")  # Change to "dark" for dark mode
ctk.set_default_color_theme("blue")

# Load model and vectorizer
model_path = r"C:\Users\gaura\OneDrive\Desktop\aiassignment\models\naive_bayes_model.joblib"
vectorizer_path = r"C:\Users\gaura\OneDrive\Desktop\aiassignment\models\tfidf_vectorizer.joblib"

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# App class
class SpamClassifierApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("📨 Spam Email Classifier")
        self.geometry("700x500")
        self.resizable(False, False)

        # Layout frame (for better alignment)
        self.frame = ctk.CTkFrame(self, width=640, height=460, corner_radius=15)
        self.frame.pack(pady=20)

        # Heading
        self.heading = ctk.CTkLabel(self.frame, text="Spam Email Classifier", font=("Helvetica", 28, "bold"))
        self.heading.pack(pady=(20, 10))

        # Subheading
        self.subheading = ctk.CTkLabel(self.frame, text="Enter a message below to check if it's spam or not.", font=("Helvetica", 14))
        self.subheading.pack(pady=(0, 10))

        # Input field
        self.message_entry = ctk.CTkTextbox(self.frame, width=550, height=120, font=("Helvetica", 14), corner_radius=10)
        self.message_entry.pack(pady=(10, 10))

        # Buttons frame
        self.button_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        self.button_frame.pack(pady=10)

        self.predict_button = ctk.CTkButton(self.button_frame, text="🔍 Classify", command=self.classify_message, font=("Helvetica", 14), width=140)
        self.predict_button.grid(row=0, column=0, padx=10)

        self.clear_button = ctk.CTkButton(self.button_frame, text="🧹 Clear", command=self.clear_input, font=("Helvetica", 14), width=100)
        self.clear_button.grid(row=0, column=1, padx=10)

        # Result label
        self.result_label = ctk.CTkLabel(self.frame, text="", font=("Helvetica", 18, "bold"), text_color="#222")
        self.result_label.pack(pady=20)

        # Footer
        self.footer = ctk.CTkLabel(self, text="Made with ❤️ by Gaurav", font=("Helvetica", 12), text_color="gray")
        self.footer.pack(side="bottom", pady=5)

    def classify_message(self):
        input_text = self.message_entry.get("1.0", "end").strip()
        if not input_text:
            self.result_label.configure(text="⚠️ Please enter a message.")
            return

        input_data = vectorizer.transform([input_text])
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            self.result_label.configure(text=" Prediction: Ham", text_color="green")
        else:
            self.result_label.configure(text="  Prediction: Spam", text_color="red")

    def clear_input(self):
        self.message_entry.delete("1.0", "end")
        self.result_label.configure(text="")

# Run the app
if __name__ == "__main__":
    app = SpamClassifierApp()
    app.mainloop()
