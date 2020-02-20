import telebot, cv2, os, hashlib, random

from gan import GAN

random.seed(10)

width, heigth = 128, 128
TOKEN = "1078723988:AAGEqQnbEKEkeRenB09SXeDZmWeJyIsCN88"
keyboard1 = telebot.types.ReplyKeyboardMarkup()
keyboard1.row('тян')
proxy = telebot.apihelper.proxy = {'https': 'socks5h://54.38.195.161:56153',}

self_path = os.path.dirname(os.path.abspath(__file__))
temp_path = os.path.join(self_path, "temp")

if not os.path.exists(temp_path):
    os.makedirs(temp_path)

bot = telebot.TeleBot(TOKEN)

generator = GAN(buff_size=10035, batch_size=16, epochs=5000, imgs_size=(width, heigth))

@bot.message_handler(commands=['start', 'help'])
def start_handler(message):
    bot.send_message(message.chat.id, "Данный бот может генерировать аниме тян. Чтобы использовать напиши тян.", reply_markup=keyboard1)

@bot.message_handler(content_types=['text'])
def msg_handler(message):
    if message.text == "тян":
        bot.send_message(message.chat.id, "А ну чичас, грузица")
        image = generator.generate_image()
        img_name = os.path.join(temp_path, hashlib.sha256((str(message.message_id)+str(random.random())).encode()).hexdigest() + ".jpg")
        cv2.imwrite(img_name, image)
        image = open(img_name, 'rb')
        bot.send_photo(message.chat.id, image)
        image.close()
        os.remove(img_name)
    else:
        bot.send_message(message.chat.id, "Каво? Ты что дядя хочешь от дедушки. Я знаю команды: Тян мне, тян мне, тян, Тян")


bot.polling(none_stop=True)