import numpy as np


def merge_colors(bg: np.ndarray, color: np.ndarray, opacity):
    """Blend colors to emulate opacity"""
    return (1-opacity)*bg + opacity*color


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
        return np.array(self.canvas[self.canvas_slice], dtype=np.uint8)


class Brush:

    def __init__(self, size, color, opacity=1, opacity_falloff=None):
        self.size = size
        self.color = color
        self.opacity = opacity
        self.opacity_falloff = opacity_falloff

    def _blend(self, area):
        raise NotImplementedError

    def resize(self, new_size):
        self.size = new_size

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
                    area[y][x] = merge_colors(area[y][x], self.color, opacity)

            return np.array(area, dtype=int)


        elif self.opacity_falloff == "linear":

            opacity_step = self.opacity / (np.shape(area)[0] // 2)
            center = np.shape(area)[0]//2

            # linear
            for y in range(np.shape(area)[0]):
                for x in range(np.shape(area)[1]):
                    # The distance function used below is, I have been informed, the 'Chebyshev Distance'.
                    opacity = self.opacity - opacity_step * (max(abs(y - center), abs(x - center)))
                    area[y][x] = merge_colors(area[y][x], self.color, opacity)

            return np.array(area, dtype=int)


        else:
            raise NotImplementedError
