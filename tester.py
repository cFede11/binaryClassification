import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import csv
from PIL import Image, ImageDraw, ImageFont
import shutil

def predict_image(image_path, model_path, class_names):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model_path.predict(img_array)
    predicted_class_index = int(np.round(prediction[0][0]))
    predicted_class = class_names[predicted_class_index]
    return predicted_class, prediction[0][0]

def results_csv(data):
    with open("Results\\Results.csv", 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)

def overview_image(image_path, prediction_result, class_name):
    img = Image.open(image_path)

    width, height = img.size
    max_width = 400
    max_height = 400
    new_width = width
    new_height = height
    if width > max_width or height > max_height:
        ratio = min(max_width / width, max_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    final_width = 430
    final_height = 500
    final_img = Image.new("RGB", (final_width, final_height), color="white")
    offset = (final_width - new_width) // 2
    final_img.paste(img, (offset, 15))

    draw = ImageDraw.Draw(final_img)
    font = ImageFont.load_default()
    image_name = os.path.basename(image_path)

    if prediction_result[0] == class_name:
        text = f"Image Result: {prediction_result[0]}"
        fill_color = "black"
        background_color = "green"
    else:
        text = f"Image Result: {prediction_result[0]} ({class_name})"
        fill_color = "black"
        background_color = "red"

    text_length = draw.textlength(text, font=font)
    
    draw.text((offset, new_height + 20), f"Image Name: {image_name}", fill="black", font=font)
    draw.rectangle([(offset, new_height + 40), (offset + text_length, new_height + 40 + 12)], fill=background_color)
    draw.text((offset, new_height + 40), text, fill=fill_color, font=font)
    draw.text((offset, new_height + 60), f"Image Percentage: {prediction_result[1]}", fill="black", font=font)

    output_path = "Results\\" + os.path.basename(image_path)
    final_img.save(output_path)

def reset_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' deleted.")

    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created successfully.")

def main():

    model_path = tf.keras.models.load_model('image_classification_model.h5')
    test_data_dir = 'imageset\\test'

    class_names = []
    correct_predictions = 0
    total_predictions = 0

    for entry in os.listdir(test_data_dir):
        entry_path = os.path.join(test_data_dir, entry)
        if os.path.isdir(entry_path):
            class_names.append(entry)

    reset_folder("Results")

    for class_name in class_names:
        class_dir = os.path.join(test_data_dir, class_name)
        for img_name in os.listdir(class_dir):
            result = "FAILED"
            img_path = os.path.join(class_dir, img_name)
            prediction_result = predict_image(img_path, model_path, class_names)
            if prediction_result[0] == class_name:
                correct_predictions += 1
                result = "CORRECT"
            result_data = [img_name, result, prediction_result[1]]
            results_csv(result_data)
            overview_image(img_path, prediction_result, class_name)
            total_predictions += 1

    accuracy = correct_predictions / total_predictions
    print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()