import math
from typing import Union, List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import random


def cluster_barplot(
    arr: np.ndarray,
    node_names: list,
    figs_folder: str,
    figure_name: str,
    cluster_name: str,
    per_page_num: int = 50,
    sorted: bool = False,
    color: str = "red",
    y_offset: float = 0.1,
    per_page_length: float = 14,
):
    """This function plots a bar plot for each node in the cluster.

    Args:
        arr (np.ndarray): Array of values to plot.
        node_names (list): List of node names.
        figs_folder (str): Folder to save the figure.
        figure_name (str): Name of the figure.
        cluster_name (str): Name of the cluster.
        per_page_num (int, optional): Number of nodes to plot per page. Defaults to 50.
        sorted (bool, optional): If True, the nodes are sorted by the values. Defaults to False.
        color (str, optional): Color of the bars. Defaults to "red".
        y_offset (float, optional): Offset for the y-axis. Defaults to 0.1.
        per_page_length (float, optional): Length of the figure. Defaults to 14.
    """
    num_pages = math.ceil(len(arr) / per_page_num)
    fig, axs = plt.subplots(num_pages, 1, figsize=(per_page_length, 5 * num_pages))

    if type(axs) == plt.Axes:
        axs = [axs]

    if sorted:
        sorted_indices = np.argsort(arr)
        modified_num_array = np.array(arr)[sorted_indices]
        modified_name_array = np.array(node_names)[sorted_indices]
    else:
        modified_num_array = arr
        modified_name_array = node_names

    for page in range(num_pages):
        l = per_page_num * page
        if page == num_pages - 1:
            u = len(modified_num_array)
        else:
            u = l + per_page_num

        axs[page].bar(
            [i for i in range(u - l)],
            modified_num_array[l:u],
            color=color,
        )

        axs[page].set_xticks(range(u - l))
        axs[page].set_xticklabels(
            modified_name_array[l:u], rotation=45, ha="right", fontsize=6
        )

        axs[page].set_ylim(top=max(modified_num_array) + y_offset)

        axs[page].set_title(
            f"{cluster_name} {figure_name.capitalize()} [{l + 1} - {u}]"
        )
        axs[page].set_ylabel("# Associations")

    fig.tight_layout()

    file_name = f"{figs_folder}/{figure_name.lower().replace(' ', '_')}_{cluster_name.lower()}.pdf"
    plt.savefig(file_name)
    print(f"Figure Saved: {file_name}")


def plot_per_group_associations(
    figs_folder: Union[str],
    node_names: list,
    cluster_name: str,
    num_list: list,
    num_pos_list: list,
    c_pos: str,
    c_neg: str,
) -> None:
    """This function plots the number of positive and negative associations per each node in the cluster.

    Args:
        figs_folder (Union[str]): Folder to save the figure.
        node_names (list): List of node names.
        cluster_name (str): Name of the cluster.
        num_list (list): List of number of associations. (len(node_names) * 1)
        num_pos_list (list): List of number of positive associations. (len(node_names) * 1)
        c_pos (str): Color for the positive associations.
        c_neg (str): Color for the negative associations.
    """
    per_page_num = 50
    num_pages = math.ceil(len(num_list) / per_page_num)
    fig, axs = plt.subplots(num_pages, 1, figsize=(14, 5 * num_pages))

    if num_pages == 1:
        axs = [axs]

    sorted_indices = np.argsort(num_list)[::-1]
    sorted_num = np.array(num_list)[sorted_indices]
    sorted_num_pos = np.array(num_pos_list)[sorted_indices]
    sorted_names = np.array(node_names)[sorted_indices]

    for page in range(num_pages):
        l = per_page_num * page
        if page == num_pages - 1:
            u = len(sorted_num)
        else:
            u = l + per_page_num

        axs[page].bar(
            [i for i in range(u - l)],
            sorted_num[l:u],
            color=c_neg,
        )
        axs[page].bar(
            [i for i in range(u - l)],
            sorted_num_pos[l:u],
            color=c_pos,
        )

        cluster_a_patch_train = mpatches.Patch(
            color=c_pos, label="positive associations"
        )
        cluster_a_neg_patch_train = mpatches.Patch(
            color=c_neg, label="negative associations"
        )
        axs[page].legend(handles=[cluster_a_patch_train, cluster_a_neg_patch_train])

        axs[page].set_xticks(range(u - l))
        axs[page].set_xticklabels(
            sorted_names[l:u], rotation=45, ha="right", fontsize=6
        )

        axs[page].set_ylim(top=max(sorted_num) + 5)

        axs[page].set_title(
            f"Per-{cluster_name.lower()} Number of Associations [{l + 1} - {u}]"
        )
        axs[page].set_xlabel(f"{cluster_name.capitalize()} Name")
        axs[page].set_ylabel("# Associations")

    fig.tight_layout()

    plt.savefig(f"{figs_folder}/per_{cluster_name.lower()}_num_associations.pdf")
    print(
        f"Figure Saved: {figs_folder}/per_{cluster_name.lower()}_num_associations.pdf"
    )


def plot_x_vs_y_dist(
    y_list: Union[List[float], np.ndarray],
    y_list_list: Union[List[List[float]], np.ndarray],
    x_list: Union[List[float], np.ndarray],
    figs_folder: str,
    cold_color: str,
    warm_color: str,
    x_name: str = "Entropy",
    y_name: str = "AUC",
    title: str = "",
    xlim_left: float = -0.1,
    xlim_right: float = 1.1,
    ylim_up: float = 1.0,
    ylim_down: float = 0.4,
    fig_width: float = 10,
    fig_height: float = 8,
    violon_width: float = 0.03,
    max_k: Union[None, int] = None,
):
    """This function plots a line and its distribution in x_list using violin plot.

    Args:
        y_list (Union[List[float], np.ndarray]): List of y values.
        y_list_list (Union[List[List[float]], np.ndarray]): (len = len(x_list) * p) List of lists of y values. Each list will be plotted as a violin plot.
        x_list (Union[List[float], np.ndarray]): List of x values.
        figs_folder (str): Folder to save the figure.
        cold_color (str): Color for the lines.
        warm_color (str): Color for the violin plot.
        x_name (str, optional): x-axis label. Defaults to "Entropy".
        y_name (str, optional): y-axis label. Defaults to "AUC".
        title (str, optional): Title of the plot. Defaults to "".
        xlim_left (float, optional): Left limit of x-axis. Defaults to -0.1.
        xlim_right (float, optional): Right limit of x-axis. Defaults to 1.1.
        ylim_up (float, optional): Upper limit of y-axis. Defaults to 1.0.
        ylim_down (float, optional): Lower limit of y-axis. Defaults to 0.4.
        fig_width (float, optional): Width of the figure. Defaults to 10.
        fig_height (float, optional): Height of the figure. Defaults to 8.
        violon_width (float, optional): Width of the violin plot. Defaults to 0.03.
        max_k (Union[None, int], optional): Maximum number of points to plot. Defaults to None. If None, all k points are plotted.
    """
    if max_k is None:
        max_k = len(x_list)

    fig, axe = plt.subplots(figsize=(fig_width, fig_height))
    axe.plot(x_list[:max_k], y_list[:max_k], color=cold_color, marker="o")

    violins = axe.violinplot(
        y_list_list[:max_k],
        positions=x_list[:max_k],
        widths=violon_width,
        showmeans=False,
        showextrema=True,
    )

    # Customize violin plot colors
    for pc in violins["bodies"]:
        pc.set_facecolor(warm_color)
        pc.set_edgecolor(cold_color)
        pc.set_alpha(0.4)

    violins["cbars"].set_color(cold_color)
    violins["cmins"].set_color(cold_color)
    violins["cmaxes"].set_color(cold_color)

    # Set plot properties
    axe.set_xlabel(x_name)
    axe.set_ylabel(y_name)
    axe.set_title(title)
    axe.set_xlim([xlim_left, xlim_right])
    axe.set_ylim([ylim_down, ylim_up])

    # Add grid and legend
    axe.grid(axis="y", linestyle="--", alpha=0.7)

    # Tight layout and save
    fig.tight_layout()
    file_name = f"{figs_folder}/{title.lower().replace(' ', '_')}_plot_x_vs_y_dist.pdf"
    plt.savefig(file_name)
    print(f"\nFigure Saved: {file_name}")


def plot_xs_vs_y_dist(
    y_list_list: Union[List[List[float]], np.ndarray],
    x_list: Union[List[float], np.ndarray],
    figs_folder: str,
    cold_color: str,
    warm_color: str,
    max_y_plot: Union[None, int] = None,
    x_name: str = "Entropy",
    y_name: str = "AUC",
    title: str = "",
    xlim_left: float = -0.1,
    xlim_right: float = 1.1,
    ylim_up: float = 1.0,
    ylim_down: float = 0.4,
    fig_width: float = 10,
    fig_height: float = 8,
    violon_width: float = 0.03,
    max_k: Union[None, int] = None,
    seed: int = 42,
):
    """This function plots different lines which are in y_list_list and their distribution in x_list using violin plot.

    Args:
        y_list_list (Union[List[List[float]], np.ndarray]): (len = p * len(x_list))List of lists of y values. Each list is a line.
        x_list (Union[List[float], np.ndarray]): List of x values.
        figs_folder (str): Folder to save the figure.
        cold_color (str): Color for the lines.
        warm_color (str): Color for the violin plot.
        max_y_plot (Union[None, int], optional): Maximum number of lines to plot. Defaults to None. If None, all lines are plotted.
        x_name (str, optional): x-axis label. Defaults to "Entropy".
        y_name (str, optional): y-axis label. Defaults to "AUC".
        title (str, optional): Title of the plot. Defaults to "".
        xlim_left (float, optional): Left limit of x-axis. Defaults to -0.1.
        xlim_right (float, optional): Right limit of x-axis. Defaults to 1.1.
        ylim_up (float, optional): Upper limit of y-axis. Defaults to 1.0.
        ylim_down (float, optional): Lower limit of y-axis. Defaults to 0.4.
        fig_width (float, optional): Width of the figure. Defaults to 10.
        fig_height (float, optional): Height of the figure. Defaults to 8.
        violon_width (float, optional): Width of the violin plot. Defaults to 0.03.
        seed (int, optional): Seed for random sampling. Defaults to 42.
        max_k (Union[None, int], optional): Maximum number of points to plot. Defaults to None. If None, all k points are plotted.
    """
    if max_k is None:
        max_k = len(x_list)

    x_list = x_list[:max_k]
    y_list_list = [y_list[:max_k] for y_list in y_list_list]

    if max_y_plot is None:
        max_y_plot = len(y_list_list)

    fig, axe = plt.subplots(figsize=(fig_width, fig_height))

    random.seed(seed)
    samps = random.sample(range(len(y_list_list)), max_y_plot)
    for i in samps:
        y_list = y_list_list[i]
        axe.plot(x_list, y_list + np.random.normal(0, 0.02, len(y_list)))

    violon_data = []
    for i in range(len(y_list_list[0])):
        temp = []
        for j in range(len(y_list_list)):
            temp.append(y_list_list[j][i])
        violon_data.append(temp)

    violins = axe.violinplot(
        violon_data,
        positions=x_list,
        widths=violon_width,
        showmeans=True,
        showextrema=True,
    )

    # Customize violin plot colors
    for pc in violins["bodies"]:
        pc.set_facecolor(warm_color)
        pc.set_edgecolor(cold_color)
        pc.set_alpha(0.4)

    violins["cbars"].set_color(cold_color)
    violins["cmins"].set_color(cold_color)
    violins["cmaxes"].set_color(cold_color)

    # Set plot properties
    axe.set_xlabel(x_name)
    axe.set_ylabel(y_name)
    axe.set_title(title)
    axe.set_xlim([xlim_left, xlim_right])
    axe.set_ylim([ylim_down, ylim_up])

    # Add grid and legend
    axe.grid(axis="y", linestyle="--", alpha=0.7)

    # Tight layout and save
    fig.tight_layout()
    file_name = f"{figs_folder}/{title.lower().replace(' ', '_')}_plot_xs_vs_y_dist.pdf"
    plt.savefig(file_name)
    print(f"\nFigure Saved: {file_name}")
