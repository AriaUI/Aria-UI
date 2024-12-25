from PIL import Image, ImageDraw

def draw_coord(image, coords):
    width, height = image.size
    draw = ImageDraw.Draw(image)

    abs_coords = [coords[0]/1000*width, coords[1]/1000*height]

    draw.circle(abs_coords, 10, fill="red")
    return image

def resize_image(image, long_size=1920):
    width, height = image.size
    if width > height:
        new_width = long_size
        new_height = int(height * long_size / width)
    else:
        new_height = long_size
        new_width = int(width * long_size / height)
    return image.resize((new_width, new_height))