import config

from utils import *

from toxicity_detector import NNClassifier

import telebot
from telebot.types import Message, User


# init telegram bot
bot = telebot.TeleBot(token=config.telegram_token, threaded=True)

# init toxicity classifier
classifier = NNClassifier(gpu=config.GPU_mode,
                          message_toxicity_threshold=config.message_toxicity_threshold,
                          model_path=config.path_to_model,
                          navec_path=config.path_to_navec)


@bot.message_handler(commands=['start'])
def start(message: Message) -> None:
    send_message(bot, message.chat.id, 'Привет!\n')


@bot.message_handler(content_types=['text'])
def moderate(message: Message):
    print(message.chat.id)
    print(classifier.check_is_toxic(message.text))
    
    if check_the_message_is_not_from_the_group(message):
        send_message(bot, message.chat.id, "Toxic probability: " + classifier.check_is_toxic(message.text))
        return

    chat_id = str(message.chat.id)
    # get user
    user: User = message.from_user
    # get user id
    user_id = str(user.id)
    if classifier.check_is_toxic(message.text) > message_toxicity_threshold:
        send_message(bot, message.chat.id, f'Пользователь @{user.username} Написал токсичное сообщение!')

bot.infinity_polling(timeout=60)
