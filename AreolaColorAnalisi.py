import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image


# Deve essere il primo comando Streamlit
st.set_page_config(layout="wide")

# CSS per nascondere il menu, il footer e l'header
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_st_style, unsafe_allow_html=True)


def create_circular_mask(h, w, center, radius):
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask


def to_sketch(image, mask, sensitivity=50):
    masked_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
    gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, sensitivity, sensitivity * 2)
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    inverted = cv2.bitwise_not(dilated)
    white_bg = np.ones_like(image) * 255
    sketch = cv2.bitwise_and(white_bg, white_bg, mask=inverted)
    return sketch


def get_dominant_colors(image, mask):
    masked_image = image[mask]
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(masked_image)
    colors = kmeans.cluster_centers_
    colors = colors.round(0).astype(int)
    colors = sorted(
        colors, key=lambda rgb: 0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2]
    )
    return [colors[0], colors[len(colors) // 2], colors[-1]]


def highlight_color(image, color, threshold=30, mask=None):
    distances = np.linalg.norm(image - color, axis=2)
    color_mask = distances < threshold
    if mask is not None:
        color_mask = np.logical_and(color_mask, mask)
    highlighted = image.copy()
    highlighted[~color_mask] = [255, 255, 255]
    return highlighted


def color_sketch(image, color, mask, threshold=30):
    distances = np.linalg.norm(image - color, axis=2)
    color_mask = distances < threshold
    if mask is not None:
        color_mask = np.logical_and(color_mask, mask)
    binary = color_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sketch = np.ones_like(image) * 255
    cv2.drawContours(sketch, contours, -1, (0, 0, 0), 1)
    return sketch


def overlay_sketches(image, colors, mask, threshold=30):
    overlay = np.ones_like(image) * 255
    color_map = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i, color in enumerate(colors):
        distances = np.linalg.norm(image - color, axis=2)
        color_mask = distances < threshold
        if mask is not None:
            color_mask = np.logical_and(color_mask, mask)
        binary = color_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color_map[i], 1)
    return overlay


def get_color_percentage(image, color, mask, threshold=30):
    distances = np.linalg.norm(image - color, axis=2)
    color_mask = distances < threshold
    if mask is not None:
        color_mask = np.logical_and(color_mask, mask)
    return np.sum(color_mask) / np.sum(mask) * 100


def get_download_link(img, filename, text):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href


st.title("Compact Image Deconstruction for Water Colors", anchor=False)

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    uploaded_file = st.file_uploader("Choose image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if "center_x" not in st.session_state:
            st.session_state.center_x = image.shape[1] // 2
        if "center_y" not in st.session_state:
            st.session_state.center_y = image.shape[0] // 2
        if "radius" not in st.session_state:
            st.session_state.radius = min(image.shape[0], image.shape[1]) // 4

        sensitivity = st.slider(
            "Sketch Sensitivity:", 10, 200, 50, 10, key="sensitivity"
        )
        color_threshold = st.slider(
            "Color Threshold:", 10, 100, 30, 5, key="color_threshold"
        )

        st.write("Move and Resize:")
        col_controls = st.columns(3)
        with col_controls[0]:
            st.button(
                "←",
                on_click=lambda: setattr(
                    st.session_state, "center_x", max(st.session_state.center_x - 10, 0)
                ),
            )
            st.button(
                "↑",
                on_click=lambda: setattr(
                    st.session_state, "center_y", max(st.session_state.center_y - 10, 0)
                ),
            )
        with col_controls[1]:
            st.button(
                "→",
                on_click=lambda: setattr(
                    st.session_state,
                    "center_x",
                    min(st.session_state.center_x + 10, image.shape[1]),
                ),
            )
            st.button(
                "↓",
                on_click=lambda: setattr(
                    st.session_state,
                    "center_y",
                    min(st.session_state.center_y + 10, image.shape[0]),
                ),
            )
        with col_controls[2]:
            st.button(
                "＋",
                on_click=lambda: setattr(
                    st.session_state,
                    "radius",
                    min(
                        st.session_state.radius + 10,
                        min(image.shape[0], image.shape[1]) // 2,
                    ),
                ),
            )
            st.button(
                "－",
                on_click=lambda: setattr(
                    st.session_state, "radius", max(st.session_state.radius - 10, 10)
                ),
            )

with col2:
    if uploaded_file is not None:
        mask = create_circular_mask(
            image.shape[0],
            image.shape[1],
            (st.session_state.center_x, st.session_state.center_y),
            st.session_state.radius,
        )
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        gray_image = cv2.addWeighted(
            gray_image, 0.7, np.ones_like(gray_image) * 255, 0.3, 0
        )
        masked_image = np.where(mask[..., None], image, gray_image)
        st.image(masked_image, caption="Selected Region", use_column_width=True)

        colors = get_dominant_colors(image, mask)
        st.write("Dominant Colors:")
        fig, ax = plt.subplots(figsize=(6, 0.5))
        ax.axis("off")
        ax.imshow([colors], aspect="auto")
        st.pyplot(fig)

        sketch = to_sketch(image, mask, sensitivity)
        st.image(sketch, caption="Sketch", use_column_width=True)

        overlay = overlay_sketches(image, colors, mask, color_threshold)
        st.image(overlay, caption="Overlayed Color Distribution", use_column_width=True)

with col3:
    if uploaded_file is not None:
        st.write("Color Highlights and Distribution Sketches:")
        for i, color_name in enumerate(["Darkest", "Middle", "Brightest"]):
            highlighted = highlight_color(image, colors[i], color_threshold, mask)
            st.image(
                highlighted,
                caption=f"{color_name} Color Highlight",
                use_column_width=True,
            )

            color_sketch_img = color_sketch(image, colors[i], mask, color_threshold)
            st.image(
                color_sketch_img,
                caption=f"{color_name} Color Distribution",
                use_column_width=True,
            )

            percentage = get_color_percentage(image, colors[i], mask, color_threshold)
            st.write(f"{color_name} Color:")
            st.write(f"RGB: {colors[i]}")
            st.write(f"Coverage: {percentage:.2f}%")

        st.write("Download Results:")
        st.markdown(
            get_download_link(Image.fromarray(sketch), "sketch.png", "Download Sketch"),
            unsafe_allow_html=True,
        )
        st.markdown(
            get_download_link(
                Image.fromarray(overlay),
                "overlay.png",
                "Download Color Distribution Overlay",
            ),
            unsafe_allow_html=True,
        )
