import numpy as np
import argparse
import time
import cv2 as cv
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageDraw, ImageFont
import re
import os

names = ['Agni Suktam', 'agnisUktam', 'Advaita Shatakam', 'A no bhadrAH Suktam',
         'UdakashAnti Mantra', 'Rigveda Mandala 1', 'Rigveda Mandala 2', 'Rigveda Mandala 3',
         'Rigveda Mandala 4', 'Rigveda Mandala 5', 'Rigveda Mandala 6', 'Rigveda Mandala 7',
         'Rigveda Mandala 8', 'Rigveda Mandala 9', 'Rigveda Mandala 10', 'Selected verses from Rigveda',
         'oShadhIsUktam', 'Kumara Suktam', 'khilas 1', 'Ganapati sUkta from Rigveda',
         'Shri Ganapati Atharvashirsha Upanishat or Ganapati Upanishat with Accents',
         'Gosthasukta', 'Go Samuha Suktam', 'First mantra of each Veda', 'Chamakaprashna',
         'Taittiriya AraNyaka', 'Taittiriya Brahmanam', 'Taittiriya Samhita 1', 'Taittiriya Samhita',
         'TaittiriyAranyakam aruNaprashnaH', 'Trisuparna Suktam', 'durgAsUktam', 'devIsukta (Rigveda)',
         'dhanurveda', 'Dhruvasuktam Rigveda', 'Vedokta Sabija Navagraha Mantra Japa Prayogah', 'naShTa dravya prApti sUktam',
         'NakShatrasukta', 'Narayanasukta', 'nAsadIya sUkta (Rigveda )', 'PavamAnasukta', 'Pitrisuktam', 'Purushasukta',
         'Purushasukta from Shuklayajurveda', 'Krityaapariharanasuktam or Bagalamukhisuktam', 'brahmaNaspatisUktam sasvara',
         'Bhagya Suktam or Pratah Suktam', 'shrIbhUsUktam', 'bhUsUktam', 'Mantrapushpa', 'mantrapuShpAnjali', 'Manyu Suktam',
         'Maha Sauram', 'Medhasukta', 'rakShoghna sUkta Rigveda Mandala 4 and 10', 'rAtrisUktam', 'Rashtra Suktam',
         'rudram (praise of Lord Shiva) namakam and chamakam', 'Rudrapanchakam', 'Rudraprashna',
         'Shri Shuklayajurvediya Sasvara Rudrashtadhyayi', 'Shri Shuklayajurvediya Rudrashtadhyayi', 'Varuna Suktam 1',
         'Varuna Suktam 2', 'Vastu Suktam', 'vishvakarmasUktam', 'Shri Vishnu Suktam 2', 'Vishnusuktam', 'Vedamantramanjari 1',
         'Vedamantramanjari 2', 'Vedamantramanjari 3', 'Praise of Vedas from Shrimad Bhagavata Purana Skandha 10 Adhyaya 87',
         'Shantipatha', 'Shasta Suktam', 'Shivapujana Vaidika Shodashopachara', 'Shraddha Suktam', 'shrI sUkta (Rigveda)',
         'Samvada or Akhyana sukta from Rigveda Samhita Mandala 10', 'sanj~nAnasUkta', 'Rigvediya Sandhya Vandana',
         'Shukla YajurvedIya SandhyA Morning-Noon-Evening', 'Samaveda Samhita Kauthuma ShAkha', 'Suryasukta from Rigveda',
         'SaubhagyalakShmi Upanishad', 'Svasti Suktam', 'hiraNyagarbhasUktam']

# Load the txt file and read the file line by line
def read_file_lines(filename):
    lines = []
    try:
        with open(filename, 'r', encoding='utf8') as file:
            for line in file:
                lines.append(line.strip())  # Remove trailing newline characters
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return lines

dir = r'C:\Users\Varun Gopal\Desktop\TeluguOCR_MLProject\KLA_Intern' 
text_file_paths = [os.path.join(dir, 'html_transcriptions', f'{x}.txt') for x in names]
font_paths = [os.path.join(dir, 'fonts', x) for x in os.listdir(os.path.join(dir, 'fonts'))]

f_out = open(os.path.join(dir,'Dataset','labels.txt'), 'w', encoding='utf8')

for transcription in text_file_paths:
    print(transcription)
    lines = read_file_lines(transcription)
    
    for x in lines:
        f_out.write(x + "\n")
f_out.close()

def get_text_dimensions(text_string, font):
    if font.getmask(text_string).getbbox() is None:
        return [0, 0]
    text_width = font.getmask(text_string).getbbox()[2]
    text_height = font.getmask(text_string).getbbox()[3]
    return [text_width, text_height]

def draw_telugu_text(text, font_path, font_size, text_color=(0, 0, 0)):
    font = ImageFont.truetype(font_path, font_size)

    # Get the size of the text
    text_size = get_text_dimensions(text, font)

    height = text_size[1] + 100  # Image height plus buffer for clear image
    width = text_size[0] + 20  # Image width plus buffer for clear image
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Calculate the position to center the text
    x = (img.width - text_size[0]) // 2
    y = (img.height - text_size[1]) // 2

    draw.text((x, y), text, fill=text_color, font=font)
    
    # Remove the excess white space on top and bottom
    image = np.array(img)
    non_zeros = [i for i in range(image.shape[0]) if np.sum(image[i]) != 255 * image.shape[1] * 3]
    
    image = image[non_zeros[0] - 5:non_zeros[-1] + 5, :, :]
    img = Image.fromarray(image)
  
    return img

acchulu = ['అ', 'ఆ', 'ఇ', 'ఈ', 'ఉ', 'ఊ', 'ఋ', 'ౠ', 'ఌ', 'ౡ', 'ఎ', 'ఏ', 'ఐ', 'ఒ', 'ఓ', 'ఔ', 'అం', 'అః']
hallulu = ['క', 'ఖ', 'గ', 'ఘ', 'ఙ',
           'చ', 'ఛ', 'జ', 'ఝ', 'ఞ',
           'ట', 'ఠ', 'డ', 'ఢ', 'ణ',
           'త', 'థ', 'ద', 'ధ', 'న',
           'ప', 'ఫ', 'బ', 'భ', 'మ',
           'య', 'ర', 'ల', 'వ', 'శ', 'ష', 'స', 'హ', 'ళ', 'క్ష', 'ఱ']
vallulu = ['ా', 'ి', 'ీ', 'ు', 'ూ', 'ృ', 'ౄ', 'ె', 'ే', 'ై', 'ొ', 'ో', 'ౌ', 'ం', 'ః', 'ఁ', 'ౕ', 'ౖ', 'ౢ']
connector = ['్']
numbers = ['౦', '౧', '౨', '౩', '౪', '౫', '౬', '౭', '౮', '౯']

varnmala = acchulu + hallulu + vallulu + connector + numbers + [' '] 

def cleaning_the_text_2(string):
    # Remove leading and trailing spaces
    string = string.strip()
    for x in string:
        if x in varnmala:
            continue
        else:
            string = string.replace(x, '')
    return string

lines = read_file_lines(os.path.join(dir, 'Dataset', 'labels.txt'))
f_str = open(os.path.join(dir, 'Dataset', 'strings.txt'), 'w', encoding='utf8')

number = 0
for s in lines:
    s = cleaning_the_text_2(s)
    s = re.sub("\s\s+", " ", s)
    if s == '' or s is None or s == ' ':
        continue
    number += 1
    f_str.write(s + '\n')
f_str.close()

print(number)
print(len(lines))

start = 0
lines = read_file_lines(os.path.join(dir, 'Dataset', 'strings.txt'))
f_final = open(os.path.join(dir, 'Dataset', 'final_strings.txt'), 'w', encoding='utf8')

indx = 1
print("Haa",len(lines))
for s in lines:
    # Random choice of fonts
    for font_path in random.sample(font_paths, 3):
        Img = draw_telugu_text(text=s, font_path=font_path, font_size=64)
        m = Img.size[1] // 40
        if m == 0 or Img.size[0] // m == 0:
            continue
        Img = Img.resize((Img.size[0] // m, 40))
        cv.imwrite(os.path.join(dir, 'Dataset', 'Images', f'Image{indx}.png'), np.array(Img))
        f_final.write(s + '\n')
        del Img
        del m
        indx += 1
    print(indx, end='\r')

