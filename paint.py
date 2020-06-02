import numpy as np

def merge_colors(bg, color, opacity):
    """Blend colors to emulate opacity"""

    color3 = [0, 0, 0]
    for i in range(3):
        color3[i] = int(bg[i] + opacity * (color[i] - bg[i]))
    return color3

def merge_colors2(bg: np.ndarray, color: np.ndarray, opacity):
    return (1-opacity)*bg + opacity*color


def cornercheck(x, y, l):
    """Checks whether a given x,y-position indicates a corner"""

    if (x == 0 and y == 0):
        return True
    if (x == 0 and y == l):
        return True
    if (x == l and y == 0):
        return True
    if (x == l and y == l):
        return True
    return False


def paint_round(target, y, x, color, brush_size):

    h, w = target.shape[0], target.shape[1]
    target[y][x] = color
    if brush_size > 1:
        for i in range(brush_size - 1):
            vierkant = i + 1
            x_range = np.arange(x - vierkant, x + vierkant + 1)
            y_range = np.arange(y - vierkant, y + vierkant + 1)
            lengte = len(x_range)
            for ix in range(len(x_range)):
                for iy in range(len(y_range)):
                    if (ix == 0) or (iy == 0) or (ix == lengte - 1) or (iy == lengte - 1):
                        if (y_range[iy] > 0) & (y_range[iy] < h) & (x_range[ix] > 0) & (x_range[ix] < w):
                            if (x-x_range[ix])**2 + (y-y_range[iy])**2 <= brush_size**2:
                                target[y_range[iy]][x_range[ix]] = color


def paint_opacity(target, y, x, color, brush_size):
    """Paint a specific pixel of a target image. Ensure that the color is an RGB triplet of integers. Returns true to be able to short-circuit it a lambday function"""

    h, w = target.shape[0], target.shape[1]
    opacity = 1
    target[y][x] = merge_colors(target[y][x], color, opacity)
    reduce_opacity = 1 / brush_size
    new_color = [0, 0, 0]

    if brush_size > 1:
        for i in range(brush_size):
            opacity = opacity - reduce_opacity
            for j in range(3):
                new_color[j] = 255 - opacity * (255 - color[j])
            vierkant = i + 1
            x_range = np.arange(x - vierkant, x + vierkant + 1)
            y_range = np.arange(y - vierkant, y + vierkant + 1)
            lengte = len(x_range)
            for ix in range(len(x_range)):
                for iy in range(len(y_range)):
                    if (ix == 0) or (iy == 0) or (ix == lengte - 1) or (iy == lengte - 1):
                        if (y_range[iy] > 0) & (y_range[iy] < h) & (x_range[ix] > 0) & (x_range[ix] < w):
                            if cornercheck(ix, iy, lengte - 1):
                                target[y_range[iy]][x_range[ix]] = merge_colors(target[y_range[iy]][x_range[ix]],
                                                                                new_color, opacity - reduce_opacity)
                            else:
                                target[y_range[iy]][x_range[ix]] = merge_colors(target[y_range[iy]][x_range[ix]],
                                                                                new_color, opacity)


class Canvas:
    """This class is a wrapper for image arrays. Its purpose is to prevent out-of-bounds exceptions while painting"""

    def __init__(self, image, max_brush_size=300):
        self.original_image = image
        self.max_brush_size = max_brush_size

        y = np.shape(image)[0]
        x = np.shape(image)[1]

        self.canvas = np.zeros((y + (max_brush_size // 2) * 2, x + (max_brush_size // 2) * 2, 3), dtype=int)
        self.canvas_slice = np.s_[max_brush_size // 2: y + (max_brush_size // 2),
                            max_brush_size // 2: x + (max_brush_size // 2)]

    def get_image(self):
        return self.canvas[self.canvas_slice]


class Brush:

    def __init__(self, size, color, opacity=1, opacity_falloff=None):
        self.size = size//2
        self.color = color
        self.opacity = opacity
        self.opacity_falloff = opacity_falloff

    def _blend(self, area):
        raise NotImplementedError

    def stroke(self, canvas: Canvas, y, x):
        # correct for canvas border offset
        y += canvas.max_brush_size//2
        x += canvas.max_brush_size//2

        # determine slice of target to be painted in
        size = self.size
        canvas_slice = np.s_[y - size: y + size, x - size: x + size]
        area = canvas.canvas[canvas_slice]

        canvas.canvas[canvas_slice] = self._blend(area)


class BrushRound(Brush):

    def _blend(self, area):
        if self.opacity_falloff is None:

            # a and b are center of the circle. n is circle diameter, r is circle radius
            a, b = self.size, self.size
            n = self.size*2
            r = self.size

            # black numpy magic to create a circle
            y, x = np.ogrid[-a:n - a, -b:n - b]
            mask = x * x + y * y <= r * r
            stroke = np.copy(area)
            stroke[mask] = self.color

            # put it all together
            stroke = self.opacity * stroke
            area = (1 - self.opacity) * area
            area = area + stroke

            return np.array(area, dtype=int)

        else:
            raise NotImplementedError


class BrushSquare(Brush):

    def _blend(self, area):
        if self.opacity_falloff is None:

            area = (1 - self.opacity) * area

            stroke = np.tile(self.color, (*np.shape(area)[:2], 1))
            stroke = self.opacity * stroke

            area = area + stroke

            return np.array(area, dtype=int)


        elif self.opacity_falloff == "cornered":
            """This mode started out as a bug, but I kind of like it ^^."""

            opacity_step = self.opacity / np.shape(area)[0]

            for y in range(np.shape(area)[0]):
                for x in range(np.shape(area)[1]):
                    opacity = opacity_step * (min(y, x) + 1)
                    area[y][x] = merge_colors2(area[y][x], self.color, opacity)

            return np.array(area, dtype=int)


        elif self.opacity_falloff == "linear":

            opacity_step = self.opacity / (np.shape(area)[0] // 2)
            center = np.shape(area)[0]//2

            # linear
            for y in range(np.shape(area)[0]):
                for x in range(np.shape(area)[1]):
                    # The distance function used below is, I have been informed, the 'Chebyshev Distance'.
                    opacity = self.opacity - opacity_step * (max(abs(y - center), abs(x - center)))
                    area[y][x] = merge_colors2(area[y][x], self.color, opacity)

            return np.array(area, dtype=int)


        else:
            raise NotImplementedError