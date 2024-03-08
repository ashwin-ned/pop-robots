import cv2
import numpy as np
import openai
import base64 
import random

def process_image(image, randomize=False, save=True):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold input greyscale image otsu's method
    ret, thresh_image = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow("Threshold", thresh_image)
    cv2.waitKey(0)
    if randomize:
        color1 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        color2 = (0, 0, 0)
    else:
        color1 = (0, 165, 255)
        color2 = (0, 0, 0)
    temp = np.stack((thresh_image,) * 3, axis=-1)
    # turn the whites to color1
    temp[np.all(temp == (255, 255, 255), axis=-1)] = color1
    # turn the blacks to color2
    temp[np.all(temp == (0, 0, 0), axis=-1)] = color2
    final = cv2.resize(temp, (1024, 1024))
    cv2.imshow("robbot", final)
    cv2.waitKey(0)
    
    if save:
        rand = random.randint(0, 20)
        cv2.imwrite(f"./robot_abstract{rand}.png", final)

    return final

def generate_image(object, style, color):
    openai.api_key = "YOUR API KEY HERE"
    prompt = f"A painting of a {object} in the style of {style} on a {color} background."
    response = openai.Image.create(prompt=prompt, model="dall-e-2")
    with open("image.jpg", "wb") as f:
        f.write(base64.b64decode(response["data"]))

    return response

def load_pregenerated_image(image_path, show=True):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if show:
        cv2.imshow("Generated Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return image

if __name__ == "__main__":
    # Genereate image from prompt
    generate_image("robot", "abstract", "red and yellow")
    # Load pregenerated image
    image_path = r"C:/Users/Ashwin/Desktop/CVPR24Art/media/robot1.png"
    image = load_pregenerated_image(image_path)
    # process Image
    processed = process_image(image)
