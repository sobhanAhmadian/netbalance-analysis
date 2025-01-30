import math
from typing import Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


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
