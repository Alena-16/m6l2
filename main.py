from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

import telebot
import os
def predict_custom(name_file):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("keras_Model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r", encoding='utf-8').readlines()

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(name_file).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)
    return class_name[2:], confidence_score
TOKEN = '8144729124:AAGIr48S5toUomSyAmENaxSymlXyCYkFASg'
bot = telebot.TeleBot(TOKEN)

SAVE_DIR = 'images'
os.makedirs(SAVE_DIR, exist_ok=True)

@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):
    bot.reply_to(message, 'Приветствую, это ии бот где ты отправляешь картинки и он их различает и говорит кто на картинке. На данный момент доступны животные: синички, голуби, хомяки. Приятного времяпровождения!')

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

     
        file_name = f"{message.chat.id}_{file_info.file_unique_id}.jpg"
        file_path = os.path.join(SAVE_DIR, file_name)

        with open(file_path, 'wb') as f:
            f.write(downloaded_file)
        ans = predict_custom(file_path)

        # bot.reply_to(message, "✅ Изображение успешно сохранено.")
        bot.send_message(message.chat.id, ans[0])
    except Exception as e:
        bot.reply_to(message, f"❌ Ошибка: {e}")



print("✅ Бот запущен")
bot.infinity_polling()

