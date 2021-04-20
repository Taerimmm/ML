import cv2
from PIL import Image

img = Image.open('C:\Study\project/team\style-transfer/03_style_image/SpongeBob.png').convert('RGB')

img = img.resize((500, 284), Image.ANTIALIAS)

img.save('SpongeBob.jpg')