import numpy as np
import PIL.Image as Image
from matplotlib import pyplot as plt
import imageio
import random
import math
import os
import shutil
import copy

import paint


# Function to display an image
from scipy.signal import convolve2d
from skimage import color


def display_image(array):
    plt.imshow(array)
    plt.show()


def make_gif(images):

    with imageio.get_writer('resultgif.gif', mode='I', duration=0.1) as writer:
        for image in images:
            writer.append_data(image)


# Gaussian int with an upper and lower bound
def trunc_gauss(mu, sigma, bottom, top):
    a = int(random.gauss(mu, sigma))
    while (bottom <= a <= top) == False:
        a = int(random.gauss(mu, sigma))
    return a

def is_active_convolution(agent, canvas, target_color, alpha):
    # image size
    h, w = canvas.original_image.shape

    # agent location
    x, y = agent[0], agent[1]

    # convolution mask
    scharr = np.array([[-3 - 3j, 0 - 10j, +3 - 3j],
                       [-10 + 0j, 0 + 0j, +10 + 0j],
                       [-3 + 3j, 0 + 10j, +3 + 3j]])

    # indexes of block around agent position
    indexes = np.array([[(x - 1, y + 1), (x, y + 1), (x + 1, y + 1)],
                        [(x - 1, y), (x, y), (x + 1, y)],
                        [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1)]])
    block = np.full((3, 3), 0.5)

    # fill the block with values
    for i in range(3):
        for j in range(3):
            ind_x, ind_y = indexes[i, j]
            if ind_x != 0 and ind_x != h and ind_y != 0 and ind_y != w:
                block[i, j] = canvas.original_image[ind_x, ind_y]

    # convolve the block with mask and return the value of position of the agent
    test = convolve2d(block, scharr, boundary='symm', mode='same')
    test = np.absolute(test)
    activity = test[1, 1] > alpha
    return activity


def is_active_multiplication(agent, canvas, target_color, alpha):
    # image size
    h, w = canvas.original_image.shape

    # location of agent
    x, y = agent[0], agent[1]

    # multiplication masks
    gx = [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]
    gy = [[-3, 10, -3], [0, 0, 0], [3, 10, 3]]

    # indexes of block around agent position
    indexes = np.array([[(x - 1, y + 1), (x, y + 1), (x + 1, y + 1)],
                        [(x - 1, y), (x, y), (x + 1, y)],
                        [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1)]])

    # fill the block with values
    block = np.full((3, 3), 0.5)
    for i in range(3):
        for j in range(3):
            ind_x, ind_y = indexes[i, j]
            if ind_x != 0 and ind_x != h and ind_y != 0 and ind_y != w:
                block[i, j] = canvas.original_image[ind_x, ind_y]

    # multiply the block with mask and take the sum
    s1 = np.sum(np.sum(np.multiply(gx, block)))
    s2 = np.sum(np.sum(np.multiply(gy, block)))

    # RMSE of both horizontal and vertical mask result
    mag = np.sqrt(s1 ** 2 + s2 ** 2)
    return mag

# Check whether an agent is active depending on the color distance
def is_active(agent, canvas, target_color, alpha):
    return math.sqrt((canvas.original_image[agent[0]][agent[1]][0] - target_color[0]) ** 2 + (
        canvas.original_image[agent[0]][agent[1]][1] - target_color[1]) ** 2 + (
            canvas.original_image[agent[0]][agent[1]][2] - target_color[2]) ** 2) < alpha


# The big function
def SDS(agent_locs, num_agents, target_color, alpha, canvas, epochs, brush):

    height, width = np.shape(canvas.get_image())[:2]
    active_agents = 0

    initial_brush_size = brush.size

    for epoch in range(epochs):
        if active_agents == num_agents:
            break
        print("epoch: ", epoch)
        for agent in agent_locs:
            if is_active_convolution(agent, canvas, target_color, alpha):
                agent[2] = True
                active_agents += 1

        brush.resize(initial_brush_size * (1 - int(active_agents / num_agents)))
        # brush_size = brush_size * (1 - int(active_agents / num_agents))
        # brush.resize(brush_size)


        for i, agent in enumerate(agent_locs):
            if not agent[2]:
                numbers = list(range(0, i)) + list(range(i + 1, len(agent_locs)))
                r = random.choice(numbers)
                if agent_locs[r][2]:
                    active_agent_x = agent_locs[r][1]
                    active_agent_y = agent_locs[r][0]
                    agent[0] = trunc_gauss(active_agent_y, 5, 0, height - 1)
                    agent[1] = trunc_gauss(active_agent_x, 5, 0, width - 1)

                    # Paint
                    # if is_active_convolution(agent, canvas, target_color, alpha):
                    brush.color = [255,255,255]
                    brush.stroke(canvas, agent[0], agent[1])

                else:
                    agent[0] = random.choice(range(height))
                    agent[1] = random.choice(range(width))

        #Add image for later creating a process gif
        gif_images.append(copy.deepcopy(np.copy(canvas.get_image())))

    brush.resize(initial_brush_size)

    return canvas


def color_distance(color1, color2):
    if not (len(color1) == len(color2) and len(color1) == 3):
        raise TypeError(f"Either of {color1} or {color2} is not a valid color.")
    return math.sqrt(sum([(color1[i] - color2[i])**2 for i in range(len(color1))]))
#
# def rgb2gray(rgb):
#     return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


if __name__ == "__main__":

    # Load input image
    input_img = color.rgb2gray(imageio.imread('Input_Images/Frogger_750.jpg'))
    height, width = input_img.shape[0], input_img.shape[1]

    # any brush strokes with a size above this value are not guaranteed to paint. Adjust as you see fit
    MAX_BRUSH_SIZE = 400

    # paper uses w*h / 5
    num_agents = int((width * height) / 50)
    # maximum value of color distance that makes an agent happy
    alpha = 2
    brush_size = 5
    # epochs per target color
    epochs = 50

    canvas = paint.Canvas(input_img, max_brush_size=MAX_BRUSH_SIZE)


    # Initialize brush
    brush = paint.BrushRound(brush_size, [0, 0, 0], opacity=1)
    # brush = paint.BrushSquare(brush_size, [0, 0, 0], opacity=1)
    # brush = paint.BrushSquare(brush_size, [0, 0, 0], opacity=0.2, opacity_falloff='cornered')
    # brush = paint.BrushSquare(brush_size, [0, 0, 0], opacity=1, opacity_falloff='linear')

    gif_images = []

    # Use a seed to be able to reproduce results
    seed = random.randint(0, 1_000_000_000)
    print(f"Using seed {seed}.")
    random.seed(seed)


    # Initialize agents
    agent_locs = [[x, y, False] for _ in range(num_agents) for x in random.choices(range(height)) for y in
                  random.choices(range(width))]

    target_color = [255,255,255]

    """Running SDS"""
    canvas = SDS(agent_locs, num_agents, target_color, alpha, canvas, epochs, brush)

    display_image(canvas.get_image())
    imageio.imwrite("result.png", canvas.get_image())

    make_gif(gif_images)
