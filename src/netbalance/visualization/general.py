import math
import random
from typing import List, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Circle, Wedge
from matplotlib.ticker import MaxNLocator

plt.rcParams.update(
    {
        "font.weight": "normal",  # options: 'normal', 'light', 'regular'
        "axes.labelsize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 8,
        "axes.labelweight": "regular",
        "axes.titleweight": "regular",
    }
)


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

    file_name = f"{figs_folder}/{figure_name.lower().replace(' ', '_')}_{cluster_name.lower()}.svg"
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
    max_k: Union[None, int] = None,
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
        max_k (Union[None, int], optional): Maximum number of nodes to plot. Defaults to None. If None, all nodes are plotted.
    """
    per_page_num = 50
    if max_k is None or max_k > len(num_list):
        max_k = len(num_list)
    num_pages = math.ceil(max_k / per_page_num)
    fig, axs = plt.subplots(num_pages, 1, figsize=(5.5, 2 * num_pages))

    if num_pages == 1:
        axs = [axs]

    sorted_indices = np.argsort(num_list)[::-1]
    sorted_num = np.array(num_list)[sorted_indices]
    sorted_num_pos = np.array(num_pos_list)[sorted_indices]
    sorted_names = np.array(node_names)[sorted_indices]

    gap = int(len(sorted_num) / max_k)
    if gap == 0:
        gap = 1

    s_indecies = []
    for i in range(max_k):
        s_indecies.append(gap * i)
    sorted_num = sorted_num[s_indecies]
    sorted_num_pos = sorted_num_pos[s_indecies]
    sorted_names = sorted_names[s_indecies]

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
            alpha=0.9,
        )
        axs[page].bar(
            [i for i in range(u - l)],
            sorted_num_pos[l:u],
            color=c_pos,
            alpha=0.9,
        )

        # cluster_a_patch_train = mpatches.Patch(
        #     color=c_pos, label="Positive Associations"
        # )
        # cluster_a_neg_patch_train = mpatches.Patch(
        #     color=c_neg, label="Negative Associations"
        # )
        # axs[page].legend(handles=[cluster_a_patch_train, cluster_a_neg_patch_train])

        axs[page].set_xticks(range(u - l))
        axs[page].set_xticklabels(
            ["" for _ in range(len(sorted_names[l:u]))], rotation=90, ha="right"
        )
        # axs[page].set_xticklabels(sorted_names[l:u], rotation=90, ha="right")

        axs[page].set_ylim(top=max(max(sorted_num) + 2, 10))
        axs[page].spines["top"].set_visible(False)
        axs[page].spines["right"].set_visible(False)
        axs[page].yaxis.set_major_locator(MaxNLocator(integer=True))

        # axs[page].set_title(
        #     f"Per-{cluster_name.lower()} Number of Associations [{l + 1} - {u}]",
        #     fontweight="bold",
        # # )
        # axs[page].set_xlabel(f"{cluster_name.capitalize()} Name", fontweight="bold")
        # axs[page].set_ylabel("# Associations", fontweight="bold")

    fig.tight_layout()

    file_name = f"{figs_folder}/per_{cluster_name.lower()}_num_associations_{max_k}.svg"
    plt.savefig(file_name)
    print(file_name)


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
    xlim_left: Union[float, None] = None,
    xlim_right: Union[float, None] = None,
    ylim_up: float = 1.0,
    ylim_down: float = 0.4,
    fig_width: float = 10,
    fig_height: float = 8,
    violon_width: Union[float, None] = None,
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
        xlim_left (Union[float, None], optional): Left limit of x-axis. Defaults to None.
            If None, the first element of x_list is used.
        xlim_right (Union[float, None], optional): Right limit of x-axis. Defaults to None.
            If None, the last element of x_list is used.
        ylim_up (float, optional): Upper limit of y-axis. Defaults to 1.0.
        ylim_down (float, optional): Lower limit of y-axis. Defaults to 0.4.
        fig_width (float, optional): Width of the figure. Defaults to 10.
        fig_height (float, optional): Height of the figure. Defaults to 8.
        violon_width (Union[float, None], optional): Width of the violin plot. Defaults to None.
            If None, it is calculated based on the number of points.
        max_k (Union[None, int], optional): Maximum number of points to plot. Defaults to None. If None, all k points are plotted.
    """
    if max_k is None:
        max_k = len(x_list)

    y_list_temp = []
    x_list_temp = []
    y_list_list_temp = []
    gap = int(len(x_list) / max_k)
    if gap == 0:
        gap = 1

    if violon_width is None:
        violon_width = gap / 2

    for i in range(max_k):
        x_list_temp.append(x_list[gap * i])
        y_list_temp.append(y_list[gap * i])
        y_list_list_temp.append(y_list_list[gap * i])

    x_list = x_list_temp
    y_list = y_list_temp
    y_list_list = y_list_list_temp

    if xlim_left is None:
        xlim_left = x_list[0] - gap
    if xlim_right is None:
        xlim_right = x_list[-1] + gap

    fig, axe = plt.subplots(figsize=(fig_width, fig_height))
    axe.plot(x_list, y_list, color=cold_color, marker="o")

    violins = axe.violinplot(
        y_list_list,
        positions=x_list,
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
    axe.set_xlabel(x_name, fontweight="bold")
    axe.set_ylabel(y_name, fontweight="bold")
    axe.set_title(title, fontweight="bold")
    axe.set_xlim([xlim_left, xlim_right])
    axe.set_ylim([ylim_down, ylim_up])
    axe.set_xticks(x_list)
    axe.set_xticklabels([round(x, 2) for x in x_list])

    # Add grid and legend
    axe.grid(axis="y", linestyle="--", alpha=0.7)

    # Tight layout and save
    fig.tight_layout()
    file_name = f"{figs_folder}/{title.lower().replace(' ', '_')}_plot_x_vs_y_dist.svg"
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
    xlim_left: Union[float, None] = None,
    xlim_right: Union[float, None] = None,
    ylim_up: float = 1.0,
    ylim_down: float = 0.4,
    fig_width: float = 10,
    fig_height: float = 8,
    violon_width: Union[float, None] = None,
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
        xlim_left (Union[float, None], optional): Left limit of x-axis. Defaults to None.
            If None, the first element of x_list is used.
        xlim_right  (Union[float, None], optional): Right limit of x-axis. Defaults to None.
            If None, the last element of x_list is used.
        ylim_up (float, optional): Upper limit of y-axis. Defaults to 1.0.
        ylim_down (float, optional): Lower limit of y-axis. Defaults to 0.4.
        fig_width (float, optional): Width of the figure. Defaults to 10.
        fig_height (float, optional): Height of the figure. Defaults to 8.
        violon_width (Union[float, None], optional): Width of the violin plot. Defaults to None.
            If None, it is calculated based on the number of points.
        seed (int, optional): Seed for random sampling. Defaults to 42.
        max_k (Union[None, int], optional): Maximum number of points to plot. Defaults to None. If None, all k points are plotted.
    """
    if max_k is None:
        max_k = len(x_list)

    x_list_temp = []
    y_list_list_temp = []
    gap = int(len(x_list) / max_k)
    if gap == 0:
        gap = 1

    if violon_width is None:
        violon_width = gap / 2

    for i in range(max_k):
        x_list_temp.append(x_list[gap * i])
        for y_list in y_list_list:
            y_list_temp = [y_list[gap * k] for k in range(max_k)]
            y_list_list_temp.append(y_list_temp)

    x_list = x_list_temp
    y_list_list = y_list_list_temp

    if max_y_plot is None:
        max_y_plot = len(y_list_list)

    if xlim_left is None:
        xlim_left = x_list[0] - gap
    if xlim_right is None:
        xlim_right = x_list[-1] + gap

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
    axe.set_xlabel(x_name, fontweight="bold")
    axe.set_ylabel(y_name, fontweight="bold")
    axe.set_title(title, fontweight="bold")
    axe.set_xlim([xlim_left, xlim_right])
    axe.set_ylim([ylim_down, ylim_up])
    axe.set_xticks(x_list)
    axe.set_xticklabels(x_list)

    # Add grid and legend
    axe.grid(axis="y", linestyle="--", alpha=0.7)

    # Tight layout and save
    fig.tight_layout()
    file_name = f"{figs_folder}/{title.lower().replace(' ', '_')}_plot_xs_vs_y_dist.svg"
    plt.savefig(file_name)
    print(f"\nFigure Saved: {file_name}")


def draw_bipartite_graph(
    associations,
    cluster_a_names,
    cluster_b_names,
    ax,
    node_radius=0.3,
    ring_width=0.1,
    pos_edge_width=2.0,
    neg_edge_width=2.0,
    a_nodes_position=3.0,
    b_nodes_position=0.0,
    pos_color="#66C2A5",
    neg_color="#D53E4F",
    a_color="#FEE08B",
    b_color="#5E4FA2",
    ring_bg_color="#e0e0e0",
):
    """
    Draw a bipartite graph with A nodes at top and B nodes at bottom,
    with per-node green/red proportion rings. Supports duplicate names
    across groups by using internal unique IDs.

    Args:
        associations (list of tuples): List of (a_idx, b_idx, value) tuples
            where value is 1 for positive and 0 for negative associations.
        cluster_a_names (list): List of names for A nodes.
        cluster_b_names (list): List of names for B nodes.
        ax (matplotlib.axes.Axes): Axes to draw the graph on.
        node_radius (float): Radius of the node circles.
        ring_width (float): Width of the proportion rings.
        pos_edge_width (float): Width of positive edges.
        neg_edge_width (float): Width of negative edges.
        a_nodes_position (float): Y position for A nodes.
        b_nodes_position (float): Y position for B nodes.
        pos_color (str): Color for positive edges.
        neg_color (str): Color for negative edges.
        a_color (str): Color for A nodes.
        b_color (str): Color for B nodes.
        ring_bg_color (str): Background color for the proportion rings.
    """

    a_ids = [("A", i) for i in range(len(cluster_a_names))]
    b_ids = [("B", j) for j in range(len(cluster_b_names))]

    labels = {("A", i): cluster_a_names[i] for i in range(len(cluster_a_names))}
    labels.update({("B", j): cluster_b_names[j] for j in range(len(cluster_b_names))})

    G = nx.Graph()
    G.add_nodes_from(a_ids, bipartite=0)
    G.add_nodes_from(b_ids, bipartite=1)

    pos_edges, neg_edges = [], []
    pos_count = {n: 0 for n in G.nodes()}
    neg_count = {n: 0 for n in G.nodes()}

    for a_idx, b_idx, val in associations:
        a = ("A", int(a_idx))
        b = ("B", int(b_idx))
        G.add_edge(a, b)
        if val == 1:
            pos_edges.append((a, b))
            pos_count[a] += 1
            pos_count[b] += 1
        else:
            neg_edges.append((a, b))
            neg_count[a] += 1
            neg_count[b] += 1

    pos = {}
    for i, node in enumerate(a_ids):
        pos[node] = (i, a_nodes_position)
    for j, node in enumerate(b_ids):
        pos[node] = (j, b_nodes_position)

    nx.draw_networkx_edges(
        G, pos, edgelist=pos_edges, edge_color=pos_color, width=pos_edge_width, ax=ax
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=neg_edges,
        edge_color=neg_color,
        width=neg_edge_width,
        ax=ax,
    )

    def draw_node_with_ring(x, y, fill_color, p_pos, p_neg):
        r_outer = node_radius + ring_width
        # background ring (no seams)
        ax.add_patch(
            Wedge(
                (x, y),
                r_outer,
                0,
                360,
                width=ring_width,
                facecolor=ring_bg_color,
                edgecolor="none",
            )
        )
        if p_pos is not None and p_neg is not None and (p_pos + p_neg) > 0:
            start_angle = 90.0  # start at top, clockwise
            ax.add_patch(
                Wedge(
                    (x, y),
                    r_outer,
                    start_angle,
                    start_angle + 360.0 * p_pos,
                    width=ring_width,
                    facecolor=pos_color,
                    edgecolor="none",
                )
            )
            ax.add_patch(
                Wedge(
                    (x, y),
                    r_outer,
                    start_angle + 360.0 * p_pos,
                    start_angle + 360.0 * (p_pos + p_neg),
                    width=ring_width,
                    facecolor=neg_color,
                    edgecolor="none",
                )
            )
        # node body (no outline so ring looks seamless)
        ax.add_patch(
            Circle((x, y), radius=node_radius, facecolor=fill_color, edgecolor="none")
        )

    for n in a_ids:
        x, y = pos[n]
        p, q = pos_count[n], neg_count[n]
        total = p + q
        if total > 0:
            draw_node_with_ring(x, y, a_color, p / total, q / total)
        else:
            draw_node_with_ring(x, y, a_color, None, None)

    for n in b_ids:
        x, y = pos[n]
        p, q = pos_count[n], neg_count[n]
        total = p + q
        if total > 0:
            draw_node_with_ring(x, y, b_color, p / total, q / total)
        else:
            draw_node_with_ring(x, y, b_color, None, None)

    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, ax=ax)

    ax.set_aspect("equal")
    ax.set_ylim(-0.5, max(a_nodes_position, b_nodes_position) + 0.5)
    ax.axis("off")
