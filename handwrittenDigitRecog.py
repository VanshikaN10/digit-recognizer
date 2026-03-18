import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps
import tkinter as tk
from tkinter import Label, Button

# ── 1. Load & preprocess data ──────────────────────────────────────────────
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = X_train / 255.0
X_test  = X_test  / 255.0
X_train = X_train.reshape(-1, 784)
X_test  = X_test.reshape(-1, 784)

# ── 2. Build ANN model ─────────────────────────────────────────────────────
model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10,  activation='softmax')
])

# ── 3. Compile & Train ─────────────────────────────────────────────────────
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("⏳ Training model, please wait...")
model.fit(X_train, y_train, epochs=15, batch_size=32,
          validation_split=0.1, verbose=1)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")

model.save('digit_recognizer.h5')
print("✅ Model saved!")

# ── 4. Real-Time Drawing GUI ───────────────────────────────────────────────
class DigitRecognizerApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.root.title("✏️ Draw a Digit (0-9)")
        self.root.resizable(False, False)

        # Canvas to draw on
        self.canvas_size = 280
        self.canvas = tk.Canvas(root, width=self.canvas_size,
                                height=self.canvas_size, bg='black', cursor='cross')
        self.canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

        # PIL image to capture drawing
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw  = ImageDraw.Draw(self.image)

        # Prediction label
        self.pred_label = Label(root, text="Draw a digit above!",
                                font=("Helvetica", 20, "bold"), fg="blue")
        self.pred_label.grid(row=1, column=0, columnspan=3, pady=5)

        # Confidence label
        self.conf_label = Label(root, text="", font=("Helvetica", 13), fg="gray")
        self.conf_label.grid(row=2, column=0, columnspan=3, pady=2)

        # Buttons
        Button(root, text="🔍 Predict", font=("Helvetica", 13),
               bg="green", fg="white", command=self.predict).grid(row=3, column=0, padx=10, pady=10)
        Button(root, text="🗑️ Clear", font=("Helvetica", 13),
               bg="red", fg="white", command=self.clear).grid(row=3, column=1, padx=10, pady=10)
        Button(root, text="❌ Exit", font=("Helvetica", 13),
               bg="gray", fg="white", command=root.quit).grid(row=3, column=2, padx=10, pady=10)

        # Mouse events
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.auto_predict)

        self.last_x = None
        self.last_y = None

    def paint(self, event):
        brush = 12
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                    width=brush*2, fill='white',
                                    capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_x, self.last_y, x, y],
                           fill=255, width=brush*2)
        self.last_x = x
        self.last_y = y

    def auto_predict(self, event):
        self.last_x = None
        self.last_y = None
        self.predict()

    def predict(self):
        # Resize to 28x28 like MNIST
        img = self.image.resize((28, 28), Image.LANCZOS)
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 784)

        # Predict
        prediction = self.model.predict(img_array, verbose=0)
        digit      = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        self.pred_label.config(text=f"Prediction: {digit} 🎯")
        self.conf_label.config(text=f"Confidence: {confidence:.1f}%")

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw  = ImageDraw.Draw(self.image)
        self.pred_label.config(text="Draw a digit above!")
        self.conf_label.config(text="")

# ── 5. Launch GUI ──────────────────────────────────────────────────────────
root = tk.Tk()
app  = DigitRecognizerApp(root, model)
root.mainloop()