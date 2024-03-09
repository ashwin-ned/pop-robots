import cv2
import numpy as np
import openai
import base64
import os 
import random

def process_image(image, randomize=False, save=True, show=False):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold input greyscale image otsu's method
    ret, thresh_image = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if show:
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

def blend_images(image_list):
    crop_list = []
    for image in image_list:
        cropped = image[0:1024, 250:600]
        crop_list.append(cropped)
    result = np.concatenate((crop_list[0], crop_list[1], crop_list[2], crop_list[3]), axis=1)
   
    print(result.shape)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    return result
if __name__ == "__main__":
    # Genereate image from prompt
    #generate_image("robot", "abstract", "red and yellow")
    # Load pregenerated image
    image_folder = r"C:/Users/Ashwin/Desktop/CVPR24Art/pop-robots/media/"
    processed_images = []
    for file in os.listdir(image_folder):
        if file.endswith(".png") or file.endswith(".jpg"):
            image_path = os.path.join(image_folder, file)
            image = load_pregenerated_image(image_path, show=False)
            processed = process_image(image, save=False)
            processed_images.append(processed)
    blended_image = blend_images(processed_images)
