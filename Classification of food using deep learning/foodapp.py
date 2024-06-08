import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Label, Toplevel
import cv2
import os
from datetime import datetime
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Load your trained model
model = load_model('model_v1_inceptionV3 (2).h5')

# Define the classes and calories (update this with your actual class names)
category = {
    0: ['apple_pie','Apple Pie'], 1: ['cannoli','Cannoli'], 2: ['chicken_curry','Chicken Curry'],
    3: ['chocolate_cake','Chocolate Cake'], 4: ['cup_cake','Cup Cake'], 5: ['donuts','Donuts'],
    6: ['dumplings','Dumplings'], 7: ['french_fries','French Fries'], 8: ['fried_rice','Fried Rice'], 9: ['hamburger','Hamburger'],
    10: ['hot_and_sour_soup','Hot and Sour Soup'], 11: ['hot_dog','Hot Dog'], 12: ['ice_cream','Ice Cream'],
    13: ['nachos','Nachos'], 14: ['omlette','Omlette'], 15: ['pizza','Pizza'],
    16: ['ramen','Ramen'], 17: ['samosa','Samosa'], 18: ['spring_rolls','Spring Rolls'], 19: ['waffles','Waffles']
}
calories = {
    0: 237,  # Apple Pie
    1: 267,  # Cannoli
    2: 220,  # Chicken Curry
    3: 350,  # Chocolate Cake
    4: 250,  # Cup Cake
    5: 195,  # Donuts
    6: 41,   # Dumplings
    7: 312,  # French Fries
    8: 333,  # Fried Rice
    9: 250,  # Hamburger
    10: 95,  # Hot and Sour Soup
    11: 290,  # Hot Dog
    12: 137,  # Ice Cream
    13: 300,  # Nachos
    14: 154,  # Omlette
    15: 285,  # Pizza
    16: 436,  # Ramen
    17: 100,  # Samosa
    18: 100,  # Spring Rolls
    19: 82    # Waffles
}

photo_folder = ""

def take_photo():
    global img_label
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam")
        return

    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Failed to capture image")
        cap.release()
        return

    # Release the camera
    cap.release()

    # Convert the frame to RGB and PIL format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame)
    
    # Display the captured photo
    display_image(pil_image)

    # Save the image to the selected folder with a unique filename
    filename = os.path.join(photo_folder, f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    pil_image.save(filename)
    
    # Predict the image
    predict_image(filename, model)

def upload_photo():
    global img_label
    file_path = filedialog.askopenfilename(
        title="Select a Photo",
        filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*"))
    )
    if file_path:
        pil_image = Image.open(file_path)
        
        # Display the uploaded photo
        display_image(pil_image)

        # Save the uploaded photo to the selected folder with a unique filename
        dest_path = os.path.join(photo_folder, os.path.basename(file_path))
        pil_image.save(dest_path)
        
        # Predict the image
        predict_image(dest_path, model)

def predict_image(filename, model):
    img_ = load_img(filename, target_size=(299, 299))
    img_array = img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.

    prediction = model.predict(img_processed)
    index = np.argmax(prediction)

    plt.title(f"Prediction - {category[index][1]} (Calories: {calories[index]})")
    plt.imshow(img_array.astype('uint8'))
    plt.axis('off')

    # Display the matplotlib plot in a new Toplevel window
    display_plot()

    print(f"Food Category - {category[index][1]}, Calories - {calories[index]}")

def display_plot():
    top = Toplevel(root)
    top.title("Prediction Result")

    fig = plt.gcf()
    canvas = plt.FigureCanvasTkAgg(fig, master=top)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
    canvas.draw()

    plt.close(fig)

def display_image(pil_image):
    img = pil_image.resize((250, 250))  # Resize to fit the GUI
    tk_image = ImageTk.PhotoImage(img)
    
    img_label.config(image=tk_image)
    img_label.image = tk_image

def select_photo_folder():
    global photo_folder
    folder_selected = filedialog.askdirectory(title="Select Folder to Save Photos")
    if folder_selected:
        photo_folder = folder_selected
        folder_label.config(text=f"Selected folder: {photo_folder}")

# Initialize the main window
root = tk.Tk()
root.title("Photo Capture and Upload")

photo_folder = ""

# Create the GUI elements
frame = tk.Frame(root)
frame.pack(pady=20)

folder_button = tk.Button(frame, text="Select Folder to Save Photos", command=select_photo_folder)
folder_button.pack(pady=10)

folder_label = tk.Label(frame, text="No folder selected")
folder_label.pack(pady=5)

take_photo_button = tk.Button(frame, text="Take Photo with Webcam", command=take_photo)
take_photo_button.pack(pady=10)

upload_photo_button = tk.Button(frame, text="Upload Photo from Filesystem", command=upload_photo)
upload_photo_button.pack(pady=10)

img_label = Label(root)
img_label.pack(pady=10)

result_label = Label(root, text="Prediction: ", font=("Helvetica", 16))
result_label.pack(pady=10)

# Run the GUI event loop
root.mainloop()

