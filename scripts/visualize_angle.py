import numpy as np
import matplotlib.pyplot as plt


def vec_angle(vec_1: np.ndarray, vec_2: np.ndarray) -> float:
    """Returns the angle in degrees between two vectors."""
    vec_1 = vec_1.reshape(-1)
    vec_2 = vec_2.reshape(-1)
    rise = vec_2[1] - vec_1[1]
    run = vec_2[0] - vec_1[0]
    return np.degrees(np.arctan2(rise, run))


# Fixed first point
v1 = np.array([1.5, 1.5])

# Generate v2 points around v1 in a circle (radius = 1)
angles = np.linspace(0, 2 * np.pi, 16)[:-1]  # 16 points
circle_offsets = np.stack([np.cos(angles), np.sin(angles)], axis=1)
# Adjust for CV coordinates (y grows downward)
circle_offsets[:, 1] *= -1
points = v1 + circle_offsets

# Compute angles
computed_angles = [vec_angle(v1, p) for p in points]

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
circle = plt.Circle(v1, 1, color="lightgray", fill=False)
ax.add_artist(circle)

# Draw lines and annotate angles
for p, ang in zip(points, computed_angles):
    ax.plot([v1[0], p[0]], [v1[1], p[1]], marker="o")
    ax.text(p[0] * 1.1 - 0.05, p[1] * 1.1, f"{ang:.1f}°", ha="center", va="center")

# Mark v1
ax.plot(v1[0], v1[1], "ro", label="first keypoint/ top of head (1.5, 1.5)")

# Axis setup (origin top-left, x→ right, y→ down)
ax.set_xlim(0, 3)
ax.set_ylim(0, 3)
ax.set_aspect("equal")
ax.invert_yaxis()
ax.legend()

plt.show()
