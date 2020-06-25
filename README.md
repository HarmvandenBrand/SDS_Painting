# SDS_Painting

Repository for a Natural Computing project


___

### Instructions

To generate an image, you can directly run Main.py using Python3.

There are some parameters in the main function of Main.py that you may want to play with:

`input_img` this is the input image for the algorithm. Sample images are included in the `Input_Images` folder.

`num_agents` this is the number of agents used in the algorithm. For this value, the original paper uses the product of the width and height of the input image, divided by five. This parameter greatly affects the running time of the algorithm.

`alpha` this value indicates how close a color has to be to the target color in order to be deemed 'satisfactory'. The distance function this value is used in is equivalent to the distance between two colors in 3-dimensional RGB space.

`brush_size` this value sets the baseline brush size. The brush size will vary through execution of the algorithm, partly depending on which extension is used.

`epochs` this value indicates the number of epochs SDS is run for before the agents are reinitialized and SDS is run with a different target color.

`num_colors` this value indicates the number of target colors the algorithm cycles through before finishing. Setting this parameter to, for example, 5 will make the algorithm select 5 random colors from the input image to use as target colors for 5 cycles of SDS. This parameter has a big effect on the duration of the algorithm.

`brushsize_annealing` this toggle indicates whether or not brushsize annealing should be used. If set to `True`, brush size will linearly diminish during the algorithm.
