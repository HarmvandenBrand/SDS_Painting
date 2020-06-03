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
            if is_active(agent, canvas, target_color, alpha):
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
                    brush.color = canvas.original_image[active_agent_y, active_agent_x]
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


if __name__ == "__main__":

    # Load input image
    input_img = imageio.imread('mao.jpg')
    height, width = input_img.shape[0], input_img.shape[1]

    # any brush strokes with a size above this value are not guaranteed to paint. Adjust as you see fit
    MAX_BRUSH_SIZE = 400

    # paper uses w*h / 5
    num_agents = int((width * height) / 500)
    # maximum value of color distance that makes an agent happy
    alpha = 20
    brush_size = 12
    # epochs per target color
    epochs = 2
    # number of colors to target and run SDS on
    num_colors = 750

    canvas = paint.Canvas(input_img, max_brush_size=MAX_BRUSH_SIZE)


    # Extension toggles
    brushsize_annealing = True
    used_colors_alpha = 0 # The minimum color distance from all used colors for a new target color to be accepted. Set to <0 if repeats are okay.


    if brushsize_annealing:
        brush_size *= 2

    # Initialize brush
    brush = paint.BrushRound(brush_size, [0, 0, 0], opacity=1)
    # brush = paint.BrushSquare(brush_size, [0, 0, 0], opacity=1)
    # brush = paint.BrushSquare(brush_size, [0, 0, 0], opacity=0.2, opacity_falloff='cornered')
    # brush = paint.BrushSquare(brush_size, [0, 0, 0], opacity=1, opacity_falloff='linear')

    used_colors = []
    gif_images = []

    # Use a seed to be able to reproduce results
    seed = random.randint(0, 1_000_000_000)
    print(f"Using seed {seed}.")
    random.seed(seed)


    for i in range(num_colors):

        # Initialize agents
        agent_locs = [[x, y, False] for _ in range(num_agents) for x in random.choices(range(height)) for y in
                      random.choices(range(width))]

        """Target color selection"""

        # Color is randomly sampled from the input image
        y = random.randint(0, height - 1)
        x = random.randint(0, width - 1)
        target_color = input_img[y][x]

        while not all(color_distance(target_color, used_color) > used_colors_alpha for used_color in used_colors):
            y = random.randint(0, height - 1)
            x = random.randint(0, width - 1)
            target_color = input_img[y][x]
        else:
            used_colors.append(target_color)

        """Running SDS"""

        canvas = SDS(agent_locs, num_agents, target_color, alpha, canvas, epochs, brush)

        print(f"Painted color {i+1}: {target_color}.")

        if brushsize_annealing:
            brush.resize(int(brush_size * (1 - (i/num_colors))))

    display_image(canvas.get_image())
    imageio.imwrite("result_mao1.png", canvas.get_image())

    make_gif(gif_images)