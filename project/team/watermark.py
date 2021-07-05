from PIL import Image, ImageDraw, ImageFont

# Image Load
for i in range(1320, 1523):
    image = Image.open('./project/team/data/frame/{}.jpg'.format(i))
    width, height = image.size
    print(image)

    # BoxBlur
    draw = ImageDraw.Draw(image)
    text = 'AnimeGAN-3'

    font = ImageFont.truetype('arial.ttf', 30)
    textwidth, textheight = draw.textsize(text, font)
    print(textwidth, textheight)

    margin = 10
    x = width - textwidth - margin
    y = margin

    center_x = (width - textwidth) / 2

    # Text apply
    draw.rectangle([(center_x - margin * 1.5, 0),(center_x + textwidth + margin * 1.5, textheight + margin * 2)], fill=(255,255,255), outline=(0,0,0), width=4)
    draw.text((center_x,y), text, font=font, fill=(0,0,0))
    
    # Save
    image.save('./project/team/data/frame/{}.jpg'.format(i))