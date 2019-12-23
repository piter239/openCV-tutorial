#!/usr/bin/env python
# -*- coding: utf-8 -*-
print ("Привет")

import pyttsx3

tts = pyttsx3.init()

voices = tts.getProperty('voices')


for voice in voices:

    print('=======')

    print('Name: %s' % voice.name)

    print('ID: %s' % voice.id)

    print('Lang: %s' % voice.languages)

    print('Sex: %s' % voice.gender)

    print('Age %s' % voice.age)


tts.setProperty('voice', 'ru')

# Попробовать установить предпочтительный голос

for voice in voices:

    if voice.name == 'Aleksandr':
        x = None
        tts.setProperty('voice', voice.id)

tts.say('Командный голос вырабатываю, товарищ генерал-полковник!')

tts.runAndWait()
