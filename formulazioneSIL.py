from mixbox import *
import numpy as np
import colorsys
from scipy.optimize import differential_evolution

# Definizione della palette e dei nomi dei colori
DEFAULT_PALETTE_RGB = [
    (236, 234, 226),  # Bianco (da LAB: 91.18, -1.10, 3.32)
    (13, 39, 45),  # Verde (da LAB: 11.57, -16.83, -6.38)
    (255, 195, 0),  # Giallo (da LAB: 78.24, -3.12, 97.93)
    (180, 40, 41),  # Rosso (da LAB: 39.53, 61.07, 56.24)
    (0, 20, 141),  # Blu (da LAB: 15.50, 28.46, -54.12)
]


palette_names = [
    "Bianco",
    "Verde",
    "Giallo",
    "Rosso",
    "Blu",
]


def objective_function(
    coefficients, latent_colors, target_rgb, reg_coef, hsv_coef, latent_coef
):
    coefficients = np.clip(coefficients, 0, 1)
    coefficients /= np.sum(coefficients) + 1e-8

    mixture_latent = np.dot(coefficients, latent_colors)
    target_latent = np.array(rgb_to_latent(target_rgb))
    target_hsv = colorsys.rgb_to_hsv(*[x / 255 for x in target_rgb])

    mixture_rgb = np.array(latent_to_rgb(mixture_latent))
    mixture_hsv = colorsys.rgb_to_hsv(*[x / 255 for x in mixture_rgb])

    difference = (
        np.linalg.norm(mixture_rgb - np.array(target_rgb)) / 255
        + latent_coef * np.linalg.norm(mixture_latent - target_latent)
        + hsv_coef * np.linalg.norm(np.array(mixture_hsv) - np.array(target_hsv))
        - reg_coef * np.sum(np.log(coefficients + 1e-8))
    )

    return difference


def optimize_coefficients(
    palette_rgb, target_rgb, reg_coef=0.01, hsv_coef=0.1, latent_coef=0.1
):
    latent_colors = np.array([rgb_to_latent(color) for color in palette_rgb])
    bounds = [(0, 1) for _ in range(len(palette_rgb))]

    result = differential_evolution(
        objective_function,
        bounds,
        args=(latent_colors, target_rgb, reg_coef, hsv_coef, latent_coef),
        popsize=20,
        maxiter=1000,
        tol=1e-7,
        atol=1e-7,
        mutation=(0.5, 1),
        recombination=0.7,
    )

    optimized_coefficients = result.x / np.sum(result.x)
    return optimized_coefficients


def generate_mixture(
    target_rgb,
    palette=DEFAULT_PALETTE_RGB,
    reg_coef=0.01,
    hsv_coef=0.1,
    latent_coef=0.1,
    clipping_perc=0.05,
):
    optimized_coeffs = optimize_coefficients(
        palette, target_rgb, reg_coef, hsv_coef, latent_coef
    )

    optimized_coeffs_clipped = optimized_coeffs.copy()
    optimized_coeffs_clipped[optimized_coeffs_clipped < clipping_perc] = 0
    optimized_coeffs_clipped /= np.sum(optimized_coeffs_clipped) + 1e-8

    optimized_weights = optimized_coeffs_clipped * 1  # Campione su 1 grammo

    latent_colors = np.array([rgb_to_latent(c) for c in palette])
    mixture_latent = np.dot(optimized_coeffs_clipped, latent_colors)
    mixture_rgb = np.array(latent_to_rgb(mixture_latent))

    return {"optimized_weights": optimized_weights, "mixture_rgb": mixture_rgb}


# Esempio di utilizzo
target_rgb = (200, 100, 50)  # Colore target in RGB
result = generate_mixture(target_rgb)
print(
    "Miscela in peso su 1 grammo per ogni colore della palette:",
    result["optimized_weights"],
)
print("Colore finale della miscela in RGB:", result["mixture_rgb"])
