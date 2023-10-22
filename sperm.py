"""Module for Sperm class and related functions."""
import os
from dataclasses import dataclass
from dataclasses import field
from dataclasses import InitVar
from typing import Optional
from typing import Any

import cv2
import numpy as np
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt


@dataclass
class Sperm:
    """Sperm class for holding and displaying data related to a sperm in a video defined by an id."""

    id: int = field()
    sperm_overlay_image_shape: InitVar[tuple] = field()
    sperm_overlay_image: np.ndarray = field(init=False)
    p_num_5: list[Any] = field(default_factory=list, repr=False)
    p_num_6: list[Any] = field(default_factory=list, repr=False)
    p_num_7: list[Any] = field(default_factory=list, repr=False)
    p_num_8: list[Any] = field(default_factory=list, repr=False)
    head_angle: list[Any] = field(default_factory=list, repr=False)
    sperm_image: Optional[np.ndarray] = field(default=None)

    def __post_init__(self, sperm_overlay_image_shape):
        self.sperm_overlay_image = np.zeros(sperm_overlay_image_shape)

    def save_sperm_image(self, out_dir: str) -> None:
        """Saves the sperm image to out_dir.

        Args:
            out_dir(str): dir to save image to.

        Returns:
            None"""
        if self.sperm_image is None:
            raise ValueError("sperm image is None")
        cv2.imwrite(
            os.path.join(out_dir, f"id_{self.id}_sperm_image.jpeg"), self.sperm_image
        )

    def save_sperm_overlay_image(self, out_dir: str) -> None:
        """Saves the sperm overlay image to out_dir.

        Args:
            out_dir(str): dir to save image to.

        Returns:
            None"""
        cv2.imwrite(
            os.path.join(out_dir, f"id_{self.id}_sperm_overlay_image.jpeg"),
            self.sperm_overlay_image,
        )

    def save_amplitude_figures(self, out_dir: str) -> None:
        """Creates and saves the amplitude figures of last 4 points of the Sperm.

        Args:
            out_dir(str): dir to image to.

        Returns:
            None"""
        title = f"Signed Amplitude of last 4 points for id {self.id}"
        xlabel = "frame count"
        ylabel = "distance between point and head axis in pixels"
        fig, axes = plt.subplots(2, 2, sharey=True, sharex=True)
        fig.suptitle(title)
        fig.text(0.5, 0.04, xlabel, ha="center")
        fig.text(0.04, 0.5, ylabel, va="center", rotation="vertical")
        for i in range(2):
            for j in range(2):
                axes[i, j].axhline(
                    y=0, color="black", linestyle="-", linewidth=2, label=None
                )
        axes[0, 0].plot(self.p_num_5)
        axes[0, 0].set_title("point 5")
        axes[0, 1].plot(self.p_num_6)
        axes[0, 1].set_title("point 6")
        axes[1, 0].plot(self.p_num_7)
        axes[1, 0].set_title("point 7")
        axes[1, 1].plot(self.p_num_8)
        axes[1, 1].set_title("point 8")
        plt.savefig(fname=f"{os.path.join(out_dir, title)}.jpeg")
        plt.close(fig)

    def save_head_frequency_figure(self, out_dir: str) -> None:
        """Creates and saves the head_frequency figure.

        Args:
            out_dir(str): dir to figure to.

        Returns:
            None"""
        title = f"head angle vs frame for id {self.id}"
        xlabel = "frame count"
        ylabel = "angle"
        fig, ax = plt.subplots()
        fig.suptitle(title)
        fig.text(0.5, 0.04, xlabel, ha="center")
        fig.text(0.04, 0.5, ylabel, va="center", rotation="vertical")
        ax.plot(self.head_angle)
        plt.savefig(fname=f"{os.path.join(out_dir,title)}.jpeg")
        plt.close(fig)

    def save_fft_graph_for_head_frequency(
        self, sampling_rate: int, out_dir: str
    ) -> None:
        """Creates and saves the fft graph for the head frequency and estimates it.

        Args:
            id_num(int): id of the sperm.
            signal(list): list of points of graph in time domain.
            sampling_rate(int): rate of sampling to determine actual frequencies.
            out_dir(str): dir to write figure to.

        Returns:
            (None)"""
        signal = self.head_angle
        n = len(signal)
        normalize = n / 2
        fourier: np.ndarray = np.array(rfft(signal))
        frequency_axis = rfftfreq(n, d=1.0 / sampling_rate)
        norm_amplitude = np.abs(fourier / normalize)
        estimated_frequency = estimate_freq(frequency_axis, norm_amplitude)

        title = f"fourier transform of head frequency for id {self.id}"
        xlabel = "frequencies"
        ylabel = "norm amplitude"
        fig, ax = plt.subplots()
        fig.suptitle(title)
        fig.text(0.5, 0.04, xlabel, ha="center")
        fig.text(0.04, 0.5, ylabel, va="center", rotation="vertical")
        ax.plot(frequency_axis, norm_amplitude)
        ax.axvline(
            x=estimated_frequency,
            color="red",
            linestyle="--",
            label=f"Estimated frequency = {estimated_frequency:.1f}",
        )
        ax.legend()
        # ax.set_xlim(0, 50)
        plt.savefig(fname=f"{os.path.join(out_dir,title)}.jpeg")
        plt.close(fig)


def estimate_freq(frequency_axis: np.ndarray, norm_amplitude: np.ndarray) -> float:
    """Estimates the frequency of the sperm's head by analyzing the graph in the frequency domain.

    Args:
        frequency_axis(np.ndarray): array of all possible frequencies/domain.
        norm_amplitude(np.ndarray): array of amplitudes for corresponding frequencies.

    Returns:
        (float): Estimated frequency of sperm's head"""
    is_increasing = norm_amplitude > np.roll(norm_amplitude, 1)
    is_decreasing = norm_amplitude > np.roll(norm_amplitude, -1)
    is_critical_point = is_increasing & is_decreasing
    is_critical_point[[0, -1]] = False
    norm_amplitude_tmp = np.array(norm_amplitude)
    norm_amplitude_tmp[~is_critical_point] = 0
    amp_index = np.argmax(norm_amplitude_tmp)
    return frequency_axis[amp_index]
