import os
import streamlit as st
import boto3
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, deltaE_cie76
from botocore import UNSIGNED
from botocore.config import Config
from scipy.ndimage import binary_dilation, label, find_objects, gaussian_filter1d
from skimage.morphology import disk
import matplotlib.pyplot as plt

# Grading match scores
def grade_color(score):
    if score >= 750:
        return "🟢 Excellent (Highly Specific)"
    elif score >= 700:
        return "🟢 Very Good (Specific)"
    elif score >= 650:
        return "🟡 Good (Some off-target)"
    elif score >= 550:
        return "🟠 Moderate (Likely off-targets)"
    elif score >= 440:
        return "🔴 Low Confidence (Mixed Expression)"
    else:
        return "🔴 Poor Match (Diffuse or Weak)"

# Setup S3
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
bucket_name = "janelia-flylight-imagery"

# S3 image lookup
def find_cdm_png(line_name, mode="both"):
    prefix = f"Gen1/CDM/{line_name}/"
    found = {"brain": None, "vnc": None}
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if "CDM_1" in key and key.endswith(".png"):
                    key_lower = key.lower()
                    if "brain" in key_lower:
                        found["brain"] = key
                    elif "vnc" in key_lower:
                        found["vnc"] = key
    except Exception as e:
        st.error(f"Failed to list {prefix}: {e}")
    return found if mode == "both" else found.get(mode)

# Generates match images based on color difference through deltaE_cie76 (can adjust tolerance here. also consider tolerance 15, dilation 0)
def compare_images(img1_path, img2_path, output_path, tolerance=10, dilation_radius=1):
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")
    arr1 = np.asarray(img1)
    arr2 = np.asarray(img2)
    if arr1.shape != arr2.shape:
        raise ValueError("Image sizes differ")
    lab1 = rgb2lab(arr1)
    lab2 = rgb2lab(arr2)
    delta = deltaE_cie76(lab1, lab2)
    background_mask = np.all(arr1 == 0, axis=-1) & np.all(arr2 == 0, axis=-1)
    delta[background_mask] = 0
    match_mask = delta < tolerance
    dilated_mask = binary_dilation(match_mask, structure=disk(dilation_radius))
    result = np.zeros_like(arr1)
    result[dilated_mask] = arr1[dilated_mask]
    Image.fromarray(result).save(output_path)

def generate_match_images(ad, dbd, output_dir="match_output", mode="both"):
    os.makedirs(output_dir, exist_ok=True)
    result_paths = {}
    ad_keys = find_cdm_png(ad, "both") if mode == "both" else {mode: find_cdm_png(ad, mode)}
    dbd_keys = find_cdm_png(dbd, "both") if mode == "both" else {mode: find_cdm_png(dbd, mode)}
    regions = ["brain", "vnc"] if mode == "both" else [mode]

    for region in regions:
        ad_key = ad_keys.get(region)
        dbd_key = dbd_keys.get(region)
        if not ad_key or not dbd_key:
            continue
        ad_path = os.path.join(output_dir, os.path.basename(ad_key))
        dbd_path = os.path.join(output_dir, os.path.basename(dbd_key))
        s3.download_file(bucket_name, ad_key, ad_path)
        s3.download_file(bucket_name, dbd_key, dbd_path)
        match_path = os.path.join(output_dir, f"{ad}-x-{dbd}_{region}_match.png")
        compare_images(ad_path, dbd_path, match_path)
        os.remove(ad_path)
        os.remove(dbd_path)
        result_paths[region] = match_path

    return result_paths

# Analyzes match images based on numpy array. Creates grayscale + clustered image.
def analyze_match_image(match_img_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(match_img_path)
    mode = 'vnc' if 'vnc' in filename.lower() else 'brain'
    img = Image.open(match_img_path).convert("L")
    gray_arr = np.array(img)
    if mode == 'vnc':
        gray_arr[0:75, 314:314+258] = 0
        gray_arr[1080:, 0:573] = 0
    else:
        gray_arr[0:75, 950:950+260] = 0

    grayscale_path = os.path.join(output_dir, filename.replace('.png', '_grayscale.png'))
    Image.fromarray(gray_arr).save(grayscale_path)
    # Defines characteristics
    mean_brightness = np.mean(gray_arr)
    contrast_score = np.std(gray_arr)
    binary_img = gray_arr > 100
    labeled_array, _ = label(binary_img)
    raw_objects = find_objects(labeled_array)
    cluster_sizes = [np.sum(binary_img[obj]) for obj in raw_objects]
    filtered_sizes = [s for s in cluster_sizes if s >= 5]
    avg_cluster_size = np.mean(filtered_sizes) if filtered_sizes else 0
    max_cluster_size = np.max(filtered_sizes) if filtered_sizes else 0
    num_clusters = len(filtered_sizes)
    total_bright_pixels = np.sum(binary_img)
    # Creates intensity histogram
    hist, _ = np.histogram(gray_arr.flatten(), bins=256, range=(0, 255))
    norm_hist = hist / hist.sum()
    smooth_hist = gaussian_filter1d(norm_hist, sigma=2)
    left_peak_idx = np.argmax(smooth_hist[:50])
    right_peak_idx = np.argmax(smooth_hist[200:]) + 200
    dip_depth = 0
    if right_peak_idx > left_peak_idx:
        valley_min = np.min(smooth_hist[left_peak_idx:right_peak_idx + 1])
        dip_depth = 0.5 * (smooth_hist[left_peak_idx] + smooth_hist[right_peak_idx]) - valley_min

    plt.figure(figsize=(6, 4))
    plt.plot(norm_hist, color='black')
    plt.title(f"Grayscale Histogram\n{filename}")
    plt.xlabel("Pixel Intensity (0=black, 255=white)")
    plt.ylabel("Normalized Count")
    plt.yscale('log')
    plt.grid(True)
    hist_path = os.path.join(output_dir, filename.replace('.png', '_hist.png'))
    plt.savefig(hist_path)
    plt.close()

    cluster_img = Image.fromarray((labeled_array > 0).astype(np.uint8) * 255)
    cluster_path = os.path.join(output_dir, filename.replace('.png', '_clusters.png'))
    cluster_img.save(cluster_path)

    capped_total = min(total_bright_pixels, 2000)
    capped_max = min(max_cluster_size, 1000)
    capped_avg = min(avg_cluster_size, 200)
    # Personal scoring formula. Can adjust as needed, this version placed emphasis on what I sought visually.
    score = (
        0.4 * contrast_score +
        0.3 * capped_max +
        0.3 * capped_avg -
        0.1 * num_clusters +
        0.3 * dip_depth -
        0.3 * mean_brightness +
        0.2 * capped_total
    )

    return {
        'match_image': match_img_path,
        'grayscale_image': grayscale_path,
        'cluster_image': cluster_path,
        'histogram_image': hist_path,
        'score': score,
        'contrast': contrast_score,
        'dip_depth': dip_depth,
        'mean_brightness': mean_brightness,
        'num_clusters': num_clusters,
        'avg_cluster_size': avg_cluster_size,
        'max_cluster_size': max_cluster_size,
        'total_bright_pixels': total_bright_pixels
    }

# Streamlit UI
st.title("Split-GAL4 Matcher")

ad = st.text_input("Enter AD Line")
dbd = st.text_input("Enter DBD Line")
run = st.button("Generate Match")

if run and ad and dbd:
    with st.spinner("Processing..."):
        ad_keys = find_cdm_png(ad, "both")
        dbd_keys = find_cdm_png(dbd, "both")

        if not any(ad_keys.values()):
            st.error(f"❌ No images found for AD line: {ad}")
        elif not any(dbd_keys.values()):
            st.error(f"❌ No images found for DBD line: {dbd}")
        else:
            results = generate_match_images(ad, dbd)
            for region, match_path in results.items():
                st.subheader(f"{region.upper()} Match")

                ad_img_path = match_path.replace(f"_{region}_match.png", f"_ad_{region}.png")
                dbd_img_path = match_path.replace(f"_{region}_match.png", f"_dbd_{region}.png")

                ad_key = ad_keys.get(region)
                dbd_key = dbd_keys.get(region)
                s3.download_file(bucket_name, ad_key, ad_img_path)
                s3.download_file(bucket_name, dbd_key, dbd_img_path)

                col1, col2, col3 = st.columns(3)
                col1.image(ad_img_path, caption=f"AD: {ad}", use_container_width=True)
                col2.image(match_path, caption="Match Image", use_container_width=True)
                col3.image(dbd_img_path, caption=f"DBD: {dbd}", use_container_width=True)

                analysis = analyze_match_image(match_path, "match_output")

                col4, col5, col6 = st.columns(3)
                col4.image(analysis['grayscale_image'], caption="Grayscale", use_container_width=True)
                col5.image(analysis['cluster_image'], caption="Post-clustering", use_container_width=True)
                col6.image(analysis['histogram_image'], caption="Pixel Intensity Histogram", use_container_width=True)

                score = analysis['score']
                with st.expander("🔍 View analysis details"):
                    st.markdown(f"""
                        **Score:** {score:.2f} — {grade_color(score)}  
                        - **Contrast:** {analysis['contrast']:.2f}  
                        - **Dip Depth:** {analysis['dip_depth']:.4f}  
                        - **Mean Brightness:** {analysis['mean_brightness']:.2f}  
                        - **Clusters:** {analysis['num_clusters']}  
                        - **Avg Cluster Size:** {analysis['avg_cluster_size']:.2f}  
                        - **Max Cluster Size:** {analysis['max_cluster_size']:.2f}  
                        - **Bright Pixels:** {analysis['total_bright_pixels']}  
                    """)

# Metric info section
st.markdown("---") 
with st.expander("📘 Learn more about match scoring metrics"):
    st.markdown("""
    The **final match score** is a weighted composite of several image analysis metrics (e.g., contrast, clustering, dip depth, brightness). It provides a rough estimate of how well the two hemidrivers overlap in expression, based on binary masks of their color depth MIPs.
                
    The **grade categories (Excellent, Very Good, Good, etc.)** are assigned based on score thresholds that were visually determined. These thresholds were manually tuned by inspecting existing split-GAL4 image combinations and correlating numerical scores with biologically meaningful expression patterns.
    | **Score Range** | **Grade** | **Meaning**                     |
    |-----------------|-----------|----------------------------------|
    | ≥ 750           | 🟢 **Excellent** | Highly Specific, clean and strong match                |
    | 700–749         | 🟢 **Very Good** | Specific, minimal off-target                       |
    | 650–699         | 🟡 **Good**      | Some off-target expression     |
    | 550–649         | 🟠 **Moderate**  | Likely off-targets             |
    | 440–549         | 🔴 **Low Confidence** | Diffuse or weak expression         |
    | < 440           | 🔴 **Poor Match** | Diffuse or weak expression     |

    **Score Metrics**                
    - **Contrast:** Measures variation in brightness. Higher contrast means clearer boundaries between signal and background.
        > *The standard deviation of pixel intensities in the grayscale image (`np.std(gray_arr)`).*
    - **Dip Depth:** Quantifies separation between signal and background in the intensity histogram. A deeper dip suggests a cleaner separation between background and signal.
        > *Calculated by smoothing the histogram and comparing the height of left/right peaks to the valley between them.*
    - **Mean Brightness:** Average pixel brightness. Lower values usually mean more specific expression (less diffuse glow).
        > *The average grayscale pixel value (`np.mean(gray_arr)`).*
    - **Clusters:** The number of distinct bright regions (expression areas). Fewer clusters often indicate higher specificity.
        > *The image is binarized (`gray_arr > 100`), connected components labeled, and are counted after passing a size filter (≥ 5 pixels).*
    - **Avg Cluster Size:** The average size of all detected expression clusters.
        > *The average size (in pixels) of the filtered bright clusters.*
    - **Max Cluster Size:** The largest detected expression cluster.
        > *The size of the largest filtered bright cluster.*
    - **Bright Pixels:** Total number of pixels above the brightness threshold. Helps indicate overall expression spread.
        > *The total number of pixels above threshold in the binary mask.*
    """)

st.markdown("---")
st.markdown(
    "📂 **Data Source:** Color depth MIPs are provided by the [Janelia FlyLight Project Team](https://www.janelia.org/project-team/flylight) via the [Janelia FlyLight Imagery Open Data Set](https://registry.opendata.aws/janelia-flylight-imagery/)."
)
