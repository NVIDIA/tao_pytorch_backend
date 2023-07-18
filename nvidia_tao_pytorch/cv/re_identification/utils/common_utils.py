# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils for re-identification."""

import os
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def check_and_create(d):
    """
    Check if a directory path is valid and create it.

    Args:
        d (str): The path of the directory to create.

    Forward:
        If the directory doesn't exist, it will be created.
    """
    if not os.path.isdir(d):
        os.makedirs(d)


def data_to_device(data):
    """
    Transfer numpy data to GPU.

    Args:
        data (Numpy): Image data.

    Returns:
        data (Tensor): Processed image data.

    Forward:
        Transfers the input data to the GPU memory. If the input data is a list,
        each item will be individually transferred.
    """
    if isinstance(data, list):
        cuda_data = []
        for item in data:
            cuda_item = item.cuda(non_blocking=True)
            cuda_data.append(cuda_item)
    else:
        cuda_data = data.cuda(non_blocking=True)

    return cuda_data


def read_image(img_path):
    """
    Check if the image path is valid and read the image.

    Args:
        img_path (str): Image path.

    Returns:
        img (Pillow): Image data.

    Forward:
        Reads an image file from the provided path and converts it to RGB format.
        If the file does not exist, a FileNotFoundError will be raised.
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError("{} does not exist".format(img_path))
    img = Image.open(img_path).convert('RGB')
    return img


def plot_evaluation_results(num_queries, query_maps, max_rank, output_file):
    """
    Plot evaluation results from queries.

    This method will plot a Mx(N+1) grid for images from query & gallery folders.
    Query images will be randomly sampled from their folder. The closest matching
    N gallery images will be plotted besides the query images.

    M = num_queries
    N = max_rank

    The image in the first column comes from the query image folder.
    The image in the rest of the columns will come from the nearest
    matches from the gallery folder.

    A blue border is drawn over the images in the first column.

    A green border over an image indicates a true positive match.
    A red border over an image indicates a false positive match.

    This plot is saved using matplotlib at output_file location.

    Args:
        num_queries (int): Number of queries to plot.
        query_maps (list(list)): List of query images mapped with test images with their corresponding match status.
        max_rank (int): Max rank to plot.
        output_file (str): Output file to plot.

    Forward:
        Plots a grid of images showcasing the matches found for each query image.
        The grid will have a width of max_rank + 1 and a height of num_queries.
        Images are color-coded based on their match status. The plot is then saved to the specified output file.
    """
    # Create a Mx(N+1) grid.
    fig, ax = plt.subplots(num_queries, max_rank + 1)
    fig.suptitle('Sampled Matches')

    # Shuffle the data for creating a sampled plot
    random.shuffle(query_maps)
    query_maps = query_maps[:num_queries]

    # Iterate through query_maps
    for row, collections in enumerate(query_maps):
        for col, collection in enumerate(collections):

            # Images belongs to column no. 2 to N
            if col != 0:
                img_path, keep = collection
                string = "Rank " + str(col)
                if keep:  # Correct match
                    outline = "green"
                else:  # Incorrect match
                    outline = "red"

            # Image belongs in the 1st column
            else:
                img_path, _ = collection
                outline = "blue"
                string = "Query"
            img = read_image(img_path)
            draw = ImageDraw.Draw(img)
            width, height = img.size
            draw.rectangle([(0, 0), (width, height)], fill=None, outline=outline, width=10)
            ax[row, col].imshow(img)
            ax[row, col].tick_params(top=False, bottom=False, left=False, right=False,
                                     labelleft=False, labelbottom=False)
            if row == len(query_maps) - 1:
                # Beautify the text
                ax[row, col].set_xlabel(string, rotation=80)

    # Beautify the grid
    plt.gcf().subplots_adjust(bottom=0.2)

    # Save the plot
    plt.savefig(output_file)
