# app.py
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import io
import zipfile
import os
import random

st.set_page_config(page_title="Image Augmentation: 80 Filters × Rotations × Flip = 640 Images", layout="wide")
st.title("🔄 Image Augmentation Tool")
st.markdown("Upload one or more images. This tool applies **80 unique filters**, then **rotates (0°, 90°, 180°, 270°)** and **flips** them, generating a total of **640 augmented images per image.**")

uploaded_files = st.file_uploader("📁 Upload one or more images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# ------------------- 80 Filter Functions ------------------- #
def get_80_filters():
    def contrast_inc(img): return ImageEnhance.Contrast(img).enhance(1.5)
    def contrast_dec(img): return ImageEnhance.Contrast(img).enhance(0.5)
    def brightness_inc(img): return ImageEnhance.Brightness(img).enhance(1.5)
    def brightness_dec(img): return ImageEnhance.Brightness(img).enhance(0.5)
    def sharpness_inc(img): return ImageEnhance.Sharpness(img).enhance(2.0)
    def sharpness_dec(img): return ImageEnhance.Sharpness(img).enhance(0.5)
    def color_inc(img): return ImageEnhance.Color(img).enhance(2.0)
    def color_dec(img): return ImageEnhance.Color(img).enhance(0.2)

    def blur(img): return img.filter(ImageFilter.BLUR)
    def gaussian_blur(img): return img.filter(ImageFilter.GaussianBlur(radius=2))
    def box_blur(img): return img.filter(ImageFilter.BoxBlur(2))
    def median_filter(img): return img.filter(ImageFilter.MedianFilter(size=3))
    def detail(img): return img.filter(ImageFilter.DETAIL)
    def edge_enhance(img): return img.filter(ImageFilter.EDGE_ENHANCE)
    def edge_enhance_more(img): return img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    def emboss(img): return img.filter(ImageFilter.EMBOSS)
    def find_edges(img): return img.filter(ImageFilter.FIND_EDGES)
    def smooth(img): return img.filter(ImageFilter.SMOOTH)
    def smooth_more(img): return img.filter(ImageFilter.SMOOTH_MORE)
    def contour(img): return img.filter(ImageFilter.CONTOUR)

    def invert(img): return ImageOps.invert(img.convert('RGB'))
    def solarize(img): return ImageOps.solarize(img, threshold=128)
    def equalize(img): return ImageOps.equalize(img)
    def posterize(img): return ImageOps.posterize(img, bits=3)
    def autocontrast(img): return ImageOps.autocontrast(img)
    def grayscale(img): return ImageOps.grayscale(img).convert('RGB')
    
    def sepia(img):
        arr = np.array(img.convert('RGB'))
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        tr = 0.393 * r + 0.769 * g + 0.189 * b
        tg = 0.349 * r + 0.686 * g + 0.168 * b
        tb = 0.272 * r + 0.534 * g + 0.131 * b
        sepia_img = np.stack([tr, tg, tb], axis=-1)
        sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
        return Image.fromarray(sepia_img)

    def add_noise(img, amount=0.05):
        arr = np.array(img.convert('RGB')) / 255.0
        noise = np.random.normal(0, amount, arr.shape)
        noisy = np.clip(arr + noise, 0, 1) * 255
        return Image.fromarray(noisy.astype(np.uint8))

    def identity(img): return img

    filters = [
        identity, contrast_inc, contrast_dec, brightness_inc, brightness_dec,
        sharpness_inc, sharpness_dec, color_inc, color_dec, blur,
        gaussian_blur, box_blur, median_filter, detail, edge_enhance,
        edge_enhance_more, emboss, find_edges, smooth, smooth_more,
        contour, invert, solarize, equalize, posterize,
        autocontrast, grayscale, sepia,
    ]

    filters += [
        lambda img: autocontrast(contrast_inc(img)),
        lambda img: brightness_inc(contrast_dec(img)),
        lambda img: grayscale(blur(img)),
        lambda img: sepia(emboss(img)),
        lambda img: grayscale(edge_enhance(img)),
        lambda img: posterize(sharpness_inc(img)),
        lambda img: invert(color_dec(img)),
        lambda img: solarize(brightness_inc(img)),
        lambda img: smooth_more(median_filter(img)),
        lambda img: add_noise(img, amount=0.03),
        lambda img: add_noise(sharpness_inc(img), amount=0.05),
        lambda img: add_noise(sepia(img), amount=0.07),
    ]

    while len(filters) < 80:
        r = random.uniform(0.4, 2.0)
        filters.append(lambda img, r=r: ImageEnhance.Brightness(img).enhance(r))

    return filters[:80]

# ------------------ Augment Single Image ------------------ #
def augment_image_640(original_img: Image.Image):
    augmented_images = []
    filters = get_80_filters()

    for i, filter_func in enumerate(filters):
        filtered_img = filter_func(original_img)

        for angle in [0, 90, 180, 270]:
            rotated = filtered_img.rotate(angle, expand=True)
            augmented_images.append((f"filter{i+1}_rot{angle}.png", rotated))

            flipped = ImageOps.mirror(rotated)
            augmented_images.append((f"filter{i+1}_rot{angle}_flip.png", flipped))

    return augmented_images

# ------------------ Streamlit Logic ------------------ #
# ------------------ Streamlit Logic ------------------ #
if uploaded_files:
    if st.button("🚀 Generate 640 Augmented Images per Image", key="generate_btn"):
        with st.spinner("Applying filters and displaying results... Please wait..."):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:

                first_file = True  # Flag to display only the first file

                for file in uploaded_files:
                    img = Image.open(file)
                    augmented = augment_image_640(img)
                    base_name = os.path.splitext(file.name)[0]

                    # Display only for the first uploaded file
                    if first_file:
                        st.subheader(
                            f"80 Filtered Images for **{file.name}** (only first image of each filter shown)"
                        )
                        cols = st.columns(8)  # grid layout

                    display_idx = 0
                    for filename, aug_img in augmented:
                        # Save all 640 images to zip
                        foldered_name = f"{base_name}/{filename}"
                        img_bytes = io.BytesIO()
                        aug_img.save(img_bytes, format="PNG")
                        zip_file.writestr(foldered_name, img_bytes.getvalue())

                        # Show preview only for first file and rotation=0 images
                        if first_file and "_rot0.png" in filename:
                            with cols[display_idx % 8]:
                                st.image(
                                    aug_img,
                                    caption=filename.split(".")[0],
                                    use_container_width=True,
                                )
                            display_idx += 1

                    if first_file:
                        st.caption("(Rest pictures will be in similar pattern.)")
                        first_file = False  # Skip displaying for next files

            zip_buffer.seek(0)
            st.success("✅ Augmentation complete! 80 images shown above. Full 640 images available in ZIP.")
            st.download_button(
                "📦 Download All 640 Augmented Images",
                zip_buffer,
                "augmented_images_all.zip",
                "application/zip"
            )
