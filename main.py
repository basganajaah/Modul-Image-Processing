import os
import io
import numpy as np
import cv2
import matplotlib.pyplot as plt
from uuid import uuid4
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from skimage.exposure import match_histograms
from PIL import Image as PILImage
from typing import List
from skimage.metrics import structural_similarity as ssim
from tabulate import tabulate
import subprocess
from skimage import util
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

if not os.path.exists("static/uploads"):
    os.makedirs("static/uploads")

if not os.path.exists("static/histograms"):
    os.makedirs("static/histograms")

if not os.path.exists("static/dataset"):
    os.makedirs("static/dataset")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/rgb/", response_class=HTMLResponse)
async def rgb_form(request: Request):
    return templates.TemplateResponse("rgb.html", {"request": request})

@app.post("/rgb/", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    file_extension = file.filename.split(".")[-1]
    filename = f"{uuid4()}.{file_extension}"
    file_path = os.path.join("static", "uploads", filename)

    with open(file_path, "wb") as f:
        f.write(image_data)

    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)

    rgb_array = {"R": r.tolist(), "G": g.tolist(), "B": b.tolist()}

    return templates.TemplateResponse("display_rgb.html", {
        "request": request,
        "image_path": f"/static/uploads/{filename}",
        "rgb_array": rgb_array
    })

@app.get("/operation/", response_class=HTMLResponse)
async def arithmetics(request: Request):
    return templates.TemplateResponse("operation.html", {"request": request})

@app.post("/operation/", response_class=HTMLResponse)
async def perform_operation(
    request: Request,
    file: UploadFile = File(...),
    operation: str = Form(...),
    value: int = Form(...)
):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    original_path = save_image(img, "original")

    if operation == "add":
        result_img = cv2.add(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "subtract":
        result_img = cv2.subtract(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "max":
        result_img = np.maximum(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "min":
        result_img = np.minimum(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "inverse":
        result_img = cv2.bitwise_not(img)

    modified_path = save_image(result_img, "modified")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.post("/logic_operation/", response_class=HTMLResponse)
async def perform_logic_operation(
    request: Request,
    file1: UploadFile = File(...),
    file2: UploadFile = File(None),
    operation: str = Form(...)
):
    image_data1 = await file1.read()
    np_array1 = np.frombuffer(image_data1, np.uint8)
    img1 = cv2.imdecode(np_array1, cv2.IMREAD_COLOR)

    original_path = save_image(img1, "original")

    if operation == "not":
        result_img = cv2.bitwise_not(img1)
    else:
        if file2 is None:
            return HTMLResponse("Operasi AND dan XOR memerlukan dua gambar.", status_code=400)
        image_data2 = await file2.read()
        np_array2 = np.frombuffer(image_data2, np.uint8)
        img2 = cv2.imdecode(np_array2, cv2.IMREAD_COLOR)

        if operation == "and":
            result_img = cv2.bitwise_and(img1, img2)
        elif operation == "xor":
            result_img = cv2.bitwise_xor(img1, img2)

    modified_path = save_image(result_img, "modified")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })
@app.get("/grayscale/", response_class=HTMLResponse)
async def grayscale_form(request: Request):
    return templates.TemplateResponse("grayscale.html", {"request": request})

@app.post("/grayscale/", response_class=HTMLResponse)
async def convert_grayscale(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    original_path = save_image(img, "original")
    modified_path = save_image(gray_img, "grayscale")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.get("/histogram/", response_class=HTMLResponse)
async def histogram_form(request: Request):
    return templates.TemplateResponse("histogram.html", {"request": request})

@app.post("/histogram/", response_class=HTMLResponse)
async def generate_histogram(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if img is None:
        return HTMLResponse("Tidak dapat membaca gambar yang diunggah", status_code=400)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale_histogram_path = save_histogram(gray_img, "grayscale")

    color_histogram_path = save_color_histogram(img)

    return templates.TemplateResponse("histogram.html", {
        "request": request,
        "grayscale_histogram_path": grayscale_histogram_path,
        "color_histogram_path": color_histogram_path
    })

@app.get("/equalize/", response_class=HTMLResponse)
async def equalize_form(request: Request):
    return templates.TemplateResponse("equalize.html", {"request": request})

@app.post("/equalize/", response_class=HTMLResponse)
async def equalize_histogram(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)

    equalized_img = cv2.equalizeHist(img)

    original_path = save_image(img, "original")
    modified_path = save_image(equalized_img, "equalized")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.get("/specify/", response_class=HTMLResponse)
async def specify_form(request: Request):
    return templates.TemplateResponse("specify.html", {"request": request})

@app.post("/specify/", response_class=HTMLResponse)
async def specify_histogram(request: Request, file: UploadFile = File(...), ref_file: UploadFile = File(...)):
    image_data = await file.read()
    ref_image_data = await ref_file.read()

    np_array = np.frombuffer(image_data, np.uint8)
    ref_np_array = np.frombuffer(ref_image_data, np.uint8)

    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    ref_img = cv2.imdecode(ref_np_array, cv2.IMREAD_COLOR)


    if img is None or ref_img is None:
        return HTMLResponse("Gambar utama atau gambar referensi tidak dapat dibaca.", status_code=400)
	
    specified_img = match_histograms(img, ref_img, channel_axis=-1)
    specified_img = np.clip(specified_img, 0, 255).astype('uint8')

    original_path = save_image(img, "original")
    modified_path = save_image(specified_img, "specified")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.get("/statistics/", response_class=HTMLResponse)
async def statistics_form(request: Request):
    return templates.TemplateResponse("statistics.html", {"request": request})

@app.post("/statistics/", response_class=HTMLResponse)
async def calculate_statistics(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)

    mean_intensity = np.mean(img)
    std_deviation = np.std(img)

    image_path = save_image(img, "statistics")

    return templates.TemplateResponse("display_statistics.html", {
        "request": request,
        "mean_intensity": mean_intensity,
        "std_deviation": std_deviation,
        "image_path": image_path
    })

def save_image(image, prefix):
    filename = f"{prefix}_{uuid4()}.png"
    path = os.path.join("static/uploads", filename)
    cv2.imwrite(path, image)
    return f"/static/uploads/{filename}"

def save_histogram(image, prefix):
    histogram_path = f"static/histograms/{prefix}_{uuid4()}.png"
    plt.figure()
    plt.hist(image.ravel(), 256, [0, 256])
    plt.savefig(histogram_path)
    plt.close()
    return f"/{histogram_path}"

def save_color_histogram(image):
    color_histogram_path = f"static/histograms/color_{uuid4()}.png"
    plt.figure()
    for i, color in enumerate(['b', 'g', 'r']):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.savefig(color_histogram_path)
    plt.close()
    return f"/{color_histogram_path}"

@app.get("/convolution/", response_class=HTMLResponse)
async def convolution_form(request: Request):
    return templates.TemplateResponse("convolution.html", {"request": request})

@app.post("/convolution/", response_class=HTMLResponse)
async def show_convolution(request: Request, file: UploadFile = File(...), kernel_type: str = Form(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if img is None:
        return templates.TemplateResponse("error.html", {"request": request, "message": "Gagal membaca gambar."})

    if kernel_type == "average":
        modified_img = apply_convolution(img, kernel_type)
    elif kernel_type == "sharpen":
        modified_img = apply_convolution(img, kernel_type)
    elif kernel_type == "edge":
        modified_img = apply_convolution(img, kernel_type)

    original_path = save_image(img, "original")
    modified_path = save_image(modified_img, "modified")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

def apply_convolution(image, value):
    if value == "average":
        kernel = np.ones((3, 3), np.float32) / 9
    elif value == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    elif value == "edge":
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    else:
        raise ValueError("Jenis kernel tidak valid.")

    output_img = cv2.filter2D(image, -1, kernel)
    return output_img

@app.get("/zeropadding/", response_class=HTMLResponse)
async def zero_padding_form(request: Request):
    return templates.TemplateResponse("zeropadding.html", {"request": request})

@app.post("/zeropadding/", response_class=HTMLResponse)
async def show_zero_padding(request: Request, file: UploadFile = File(...), value: int = Form(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if img is None:
        return templates.TemplateResponse("error.html", {"request": request, "message": "Gagal membaca gambar."})

    modified_img = apply_zero_padding(img, value)

    original_path = save_image(img, "original")
    modified_path = save_image(modified_img, "modified")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

def apply_zero_padding(image, padding_size):
    padded_img = cv2.copyMakeBorder(image, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_img

@app.get("/passfilter/", response_class=HTMLResponse)
async def passfilter_form(request: Request):
    return templates.TemplateResponse("passfilter.html", {"request": request})

@app.post("/passfilter/", response_class=HTMLResponse)
async def show_passfilter(request: Request, file: UploadFile = File(...), filter_type: str = Form(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if img is None:
        return templates.TemplateResponse("error.html", {"request": request, "message": "Gagal membaca gambar."})

    if filter_type == "low":
        modified_img = apply_filter(img, filter_type)
    elif filter_type == "high":
        modified_img = apply_filter(img, filter_type)
    elif filter_type == "band":
        modified_img = apply_filter(img, filter_type)

    original_path = save_image(img, "original")
    modified_path = save_image(modified_img, "modified")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

def apply_filter(image, value):
    if value == "low":
        filtered_img = cv2.GaussianBlur(image, (5, 5), 0)
    elif value == "high":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        filtered_img = cv2.filter2D(image, -1, kernel)
    elif value == "band":
        low_pass = cv2.GaussianBlur(image, (9, 9), 0)
        high_pass = image - low_pass
        filtered_img = low_pass + high_pass

    return filtered_img

@app.get("/fouriertransform/", response_class=HTMLResponse)
async def fourier_form(request: Request):
    return templates.TemplateResponse("fouriertransform.html", {"request": request})

@app.post("/fouriertransform/", response_class=HTMLResponse)
async def show_fourier(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if img is None:
        return templates.TemplateResponse("error.html", {"request": request, "message": "Gagal membaca gambar."})

    modified_img = apply_fourier_transform(img)

    original_path = save_image(img, "original")
    modified_path = save_image(modified_img, "modified")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

def apply_fourier_transform(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    return magnitude_spectrum

@app.get("/periodicnoise/", response_class=HTMLResponse)
async def periodic_form(request: Request):
    return templates.TemplateResponse("periodicnoise.html", {"request": request})

@app.post("/periodicnoise/", response_class=HTMLResponse)
async def show_periodic(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if img is None:
        return templates.TemplateResponse("error.html", {"request": request, "message": "Gagal membaca gambar."})

    modified_img = reduce_periodic_noise(img)

    original_path = save_image(img, "original")
    modified_path = save_image(modified_img, "modified")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

def reduce_periodic_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    r = 30
    mask[crow-r:crow+r, ccol-r:ccol+r] = 0

    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back

@app.get("/facefilter/", response_class=HTMLResponse)
async def facefilter_form(request: Request):
    return templates.TemplateResponse("facefilter.html", {"request": request})

@app.post("/facefilter/", response_class=HTMLResponse)
async def facefilter_process(
    request: Request,
    files: List[UploadFile] = File(...),
    new_person: str = Form(...),
    filter_type: str = Form(...)
):
    if not new_person:
        return templates.TemplateResponse("facefilter.html", {
            "request": request,
            "error": "Silakan masukkan nama orang baru."
        })

    save_path = os.path.join('static/dataset', new_person)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    elif len(os.listdir(save_path)) >= 20:
        return templates.TemplateResponse("facefilter.html", {
            "request": request,
            "error": f"Nama {new_person} sudah ada di dataset dengan cukup gambar. Silakan pilih nama lain."
        })

    num_images = len(os.listdir(save_path))
    max_images = 20
    saved_images = []
    last_img = None
    last_faces = []

    for file in files:
        image_data = await file.read()
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if img is None:
            continue

        faces = detect_faces(img)
        if len(faces) > 0:
            last_img = img
            last_faces = faces

        for (x, y, w, h) in faces:
            if num_images >= max_images:
                break
            face = img[y:y+h, x:x+w]
            
            if filter_type == "saltpepper":
                processed_face = add_salt_and_pepper_noise(face, 0.02, 0.02)
            elif filter_type == "noises":
                processed_face = remove_noise(face, 5)
            elif filter_type == "sharpenimg":
                processed_face = sharpen_image(face)
            else:
                processed_face = face

            img_name = os.path.join(save_path, f"img_{num_images}.jpg")
            cv2.imwrite(img_name, processed_face)
            saved_images.append(save_image(processed_face, f"face_{num_images}"))
            num_images += 1

    if len(saved_images) == 0:
        return templates.TemplateResponse("facefilter.html", {
            "request": request,
            "error": "Tidak ada wajah yang terdeteksi dalam gambar yang diunggah."
        })

    if last_img is not None:
        for (x, y, w, h) in last_faces:
            cv2.rectangle(last_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        original_path = save_image(last_img, "original")
    else:
        original_path = None

    return templates.TemplateResponse("facefilter_result.html", {
        "request": request,
        "original_image_path": original_path,
        "saved_image_paths": saved_images,
        "num_saved": len(saved_images),
        "person_name": new_person
    })

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    colors = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(
        colors, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )
    return faces

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    total_pixels = image.shape[0] * image.shape[1]
    
    num_salt = int(total_pixels * salt_prob)
    num_pepper = int(total_pixels * pepper_prob)
    
    coords = [
        np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]
    ]
    noisy_image[coords[0], coords[1]] = 255
    
    coords = [
        np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]
    ]
    noisy_image[coords[0], coords[1]] = 0
    
    return noisy_image

def remove_noise(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)

def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

@app.get("/chaincode/", response_class=HTMLResponse)
async def chaincode_form(request: Request):
    return templates.TemplateResponse("chaincode.html", {"request": request})

@app.post("/chaincode/", response_class=HTMLResponse)
async def chaincode_process(
    request: Request,
    file: UploadFile = File(...)
):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return templates.TemplateResponse("chaincode.html", {
            "request": request,
            "error": "Gambar tidak valid atau tidak dapat dibaca."
        })

    threshold_value = 127
    _, binary_img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return templates.TemplateResponse("chaincode.html", {
            "request": request,
            "error": "Tidak ada kontur yang terdeteksi dalam gambar."
        })

    largest_contour = max(contours, key=cv2.contourArea)
    chain_code = generate_freeman_chain_code(largest_contour)

    img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_display, [largest_contour], -1, (0, 255, 0), 1)

    original_path = save_image(img, "original")
    binary_path = save_image(binary_img, "binary")
    contour_path = save_image(img_display, "contour")

    return templates.TemplateResponse("chaincode_result.html", {
        "request": request,
        "original_image_path": original_path,
        "binary_image_path": binary_path,
        "contour_image_path": contour_path,
        "chain_code": chain_code,
        "chain_code_length": len(chain_code),
        "num_contours": len(contours)
    })

def generate_freeman_chain_code(contour):
    chain_code = []
    if len(contour) < 2:
        return chain_code

    directions = {
        (1, 0): 0, (1, 1): 1, (0, 1): 2, (-1, 1): 3,
        (-1, 0): 4, (-1, -1): 5, (0, -1): 6, (1, -1): 7
    }

    for i in range(len(contour)):
        p1 = contour[i][0]
        p2 = contour[(i + 1) % len(contour)][0]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        norm_dx = np.sign(dx)
        norm_dy = np.sign(dy)
        code = directions.get((norm_dx, norm_dy))
        if code is not None:
            chain_code.append(code)

    return chain_code

@app.get("/crackcode/", response_class=HTMLResponse)
async def crackcode_form(request: Request):
    return templates.TemplateResponse("crackcode.html", {"request": request})

@app.post("/crackcode/", response_class=HTMLResponse)
async def show_crackcode(
    request: Request,
    file: UploadFile = File(...)
):
    
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if img is None:
        return templates.TemplateResponse("crackcode.html", {
            "request": request,
            "error": "Gambar tidak valid atau tidak dapat dibaca."
        })
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ksize = 5
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    original_path = save_image(img, "original")
    blurred_path = save_image(blurred, "blurred")
    edges_path = save_image(edges, "edges")

    return templates.TemplateResponse("crackcode_result.html", {
        "request": request,
        "original_image_path": original_path,
        "blurred_image_path": blurred_path,
        "edges_image_path": edges_path,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "ksize": ksize
    })

@app.get("/integralprojection/", response_class=HTMLResponse)
async def integralprojection_form(request: Request):
    return templates.TemplateResponse("integralprojection.html", {"request": request})

@app.post("/integralprojection/", response_class=HTMLResponse)
async def integralprojection_process(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return templates.TemplateResponse("integralprojection.html", {
            "request": request,
            "error": "Gambar tidak valid atau tidak dapat dibaca."
        })

    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_norm = binary_img / 255.0

    height, width = binary_norm.shape

    horizontal_projection = np.sum(binary_norm, axis=0)
    vertical_projection = np.sum(binary_norm, axis=1)

    binary_path = save_image(binary_img, "binary")

    hproj_path = save_plot(horizontal_projection, np.arange(width), horizontal_projection, 
                           "Proyeksi Horizontal (Profil Vertikal)", "Indeks Kolom", "Jumlah Piksel")
    vproj_path = save_plot(vertical_projection, vertical_projection, np.arange(height), 
                           "Proyeksi Vertikal", "Indeks Kolom", "Jumlah Piksel", invert_y=True)
    
    return templates.TemplateResponse("integralprojection_result.html", {
        "request": request,
        "binary_image_path": binary_path,
        "hproj_image_path": hproj_path,
        "vproj_image_path": vproj_path,
        "image_height": height,
        "image_width": width
    })

def save_plot(data, x, y, title, xlabel, ylabel, invert_y=False):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if invert_y:
        ax.invert_yaxis()
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    img = PILImage.open(buf)
    filename = f"plot_{uuid4()}.png"
    path = os.path.join("static/uploads", filename)
    img.save(path)
    return f"/static/uploads/{filename}"

def alternate_save_plot(fig, prefix):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img = PILImage.open(buf)
    filename = f"{prefix}_{uuid4()}.png"
    path = os.path.join("static/uploads", filename)
    img.save(path)
    return f"/static/uploads/{filename}"


@app.get("/lossy/", response_class=HTMLResponse)
async def lossy_form(request: Request):
    return templates.TemplateResponse("lossy.html", {"request": request})

@app.post("/lossy/", response_class=HTMLResponse)
async def lossy_process(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img_original_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if img_original_bgr is None:
        return templates.TemplateResponse("lossy.html", {
            "request": request,
            "error": "Gambar tidak valid atau tidak dapat dibaca."
        })
    is_color = len(img_original_bgr.shape) == 3
    img_original_cv = cv2.cvtColor(img_original_bgr, cv2.COLOR_BGR2RGB) if is_color else img_original_bgr
    original_size_bytes = len(image_data)
    min_dim = min(img_original_cv.shape[:2])
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    if win_size < 3:
        win_size = 3
    results = []
    output_folder = "static/uploads/"
    os.makedirs(output_folder, exist_ok=True)
    png_compression_levels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for level in png_compression_levels:
        basename = f"compressed_level{level}"
        png_path = os.path.join(output_folder, f"{basename}.png")
        img_to_save_png = cv2.cvtColor(img_original_cv, cv2.COLOR_RGB2BGR) if is_color else img_original_cv
        cv2.imwrite(png_path, img_to_save_png, [cv2.IMWRITE_PNG_COMPRESSION, level])
        png_size_bytes = os.path.getsize(png_path)
        img_png_compressed_bgr = cv2.imread(png_path)
        if img_png_compressed_bgr is None:
            continue
        img_png_compressed_cv = cv2.cvtColor(img_png_compressed_bgr, cv2.COLOR_BGR2RGB) if is_color else cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        psnr_png = cv2.PSNR(img_original_cv, img_png_compressed_cv)
        mse = np.mean((img_original_cv.astype(float) - img_png_compressed_cv.astype(float)) ** 2)
        psnr_manual = float('inf') if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
        try:
            ssim_png = ssim(img_original_cv, img_png_compressed_cv, channel_axis=2 if is_color else None, win_size=win_size, data_range=img_original_cv.max() - img_original_cv.min())
        except ValueError:
            ssim_png = None
        is_identical = np.array_equal(img_original_cv, img_png_compressed_cv)
        results.append([
            "Uploaded Image", 'PNG', f'Level {level}', round(png_size_bytes / 1024, 2),
            round(original_size_bytes / png_size_bytes, 2) if png_size_bytes > 0 else float('inf'),
            round(psnr_png, 2) if psnr_png != float('inf') else 'Infinity',
            round(ssim_png, 4) if ssim_png is not None else 'N/A', "Ya" if is_identical else "Tidak"
        ])
    headers = ["Citra Input", "Metode Kompresi", "Kualitas/Level", "Ukuran File (KB)", "Rasio Kompresi", "PSNR (dB)", "SSIM", "Identik?"]
    results.insert(0, ["Uploaded Image", 'Original', '-', round(original_size_bytes / 1024, 2), '1', 'Infinity', '1', '-'])
    png_path_vis = os.path.join(output_folder, "compressed_level9.png")
    img_png_vis = cv2.imread(png_path_vis)
    if is_color:
        img_png_vis = cv2.cvtColor(img_png_vis, cv2.COLOR_BGR2RGB)
    cmap_val = 'gray' if not is_color else None
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_original_cv, cmap=cmap_val)
    axes[0].set_title(f'Original ({original_size_bytes / 1024:.2f} KB)')
    axes[0].axis('off')
    if img_png_vis is not None:
        png_size_vis = os.path.getsize(png_path_vis)
        axes[1].imshow(img_png_vis, cmap=cmap_val)
        axes[1].set_title(f'PNG Level 9 ({png_size_vis / 1024:.2f} KB)')
        axes[1].axis('off')
    else:
        axes[1].set_title('PNG Level 9 (Error Loading)')
        axes[1].axis('off')
    plt.tight_layout()
    comp_comparison_path = alternate_save_plot(fig, "compression_comparison")
    levels = list(range(10))
    file_sizes = [row[3] for row in results[1:]]
    psnrs = [float(row[5]) if row[5] != 'Infinity' else 100 for row in results[1:]]
    ssims = [float(row[6]) if row[6] != 'N/A' else 1 for row in results[1:]]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Level Kompresi PNG')
    ax1.set_ylabel('Ukuran File (KB)', color='tab:blue')
    ax1.plot(levels, file_sizes, color='tab:blue', marker='o', label='Ukuran File (KB)')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    ax2.set_ylabel('PSNR (dB) / SSIM', color='tab:red')
    ax2.plot(levels, psnrs, color='tab:red', marker='x', label='PSNR (dB)')
    ax2.plot(levels, ssims, color='tab:green', marker='s', label='SSIM')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    fig.tight_layout()
    ax1.set_title('Level Kompresi PNG vs Ukuran File, PSNR, dan SSIM')
    fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.grid(True)
    graph_psnr_ssim_path = alternate_save_plot(fig, "graph_png_vs_psnr_ssim")
    labels = ['Original'] + [f'PNG-{lvl}' for lvl in levels]
    sizes = [results[0][3]] + file_sizes
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, sizes, color='skyblue')
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('Ukuran File (KB)')
    ax.set_title('Perbandingan Ukuran File: Original vs PNG Levels')
    ax.grid(axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom')
    plt.tight_layout()
    bar_file_size_path = alternate_save_plot(fig, "bar_file_size_comparison")
    return templates.TemplateResponse("lossy_result.html", {
        "request": request,
        "comp_comparison_path": comp_comparison_path,
        "graph_psnr_ssim_path": graph_psnr_ssim_path,
        "bar_file_size_path": bar_file_size_path,
        "results": results,
        "headers": headers,
        "original_size": round(original_size_bytes / 1024, 2)
    })

@app.get("/lossless/", response_class=HTMLResponse)
async def lossless_form(request: Request):
    return templates.TemplateResponse("lossless.html", {"request": request})

@app.post("/lossless/", response_class=HTMLResponse)
async def lossless_process(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img_original_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if img_original_bgr is None:
        return templates.TemplateResponse("lossless.html", {
            "request": request,
            "error": "Gambar tidak valid atau tidak dapat dibaca."
        })

    is_color = len(img_original_bgr.shape) == 3
    img_original_cv = cv2.cvtColor(img_original_bgr, cv2.COLOR_BGR2RGB) if is_color else img_original_bgr
    original_size_bytes = len(image_data)

    min_dim = min(img_original_cv.shape[:2])
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    if win_size < 3:
        win_size = 3

    results = []
    output_folder = "static/uploads/"
    os.makedirs(output_folder, exist_ok=True)

    # PNG Compression for all levels with optimization
    png_compression_levels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for level in png_compression_levels:
        basename = f"compressed_level{level}"
        png_path = os.path.join(output_folder, f"{basename}.png")

        img_to_save_png = cv2.cvtColor(img_original_cv, cv2.COLOR_RGB2BGR) if is_color else img_original_cv
        cv2.imwrite(png_path, img_to_save_png, [cv2.IMWRITE_PNG_COMPRESSION, level])
        png_size_bytes = os.path.getsize(png_path)

        try:
            subprocess.run(['optipng', '-o7', png_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            png_size_bytes_opt = os.path.getsize(png_path)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"optipng not found or failed for level {level}; using original size.")
            png_size_bytes_opt = png_size_bytes

        img_png_compressed_bgr = cv2.imread(png_path)
        if img_png_compressed_bgr is None:
            print(f"Error: Tidak dapat memuat citra terkompresi {png_path}")
            continue

        if is_color:
            img_png_compressed_cv = cv2.cvtColor(img_png_compressed_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_png_compressed_cv = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
            if img_png_compressed_cv is None:
                print(f"Error: Gagal memuat ulang {png_path} sebagai grayscale.")
                continue

        psnr_png = cv2.PSNR(img_original_cv, img_png_compressed_cv)
        mse = np.mean((img_original_cv.astype(float) - img_png_compressed_cv.astype(float)) ** 2)
        psnr_manual = float('inf') if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))

        try:
            ssim_png = ssim(
                img_original_cv,
                img_png_compressed_cv,
                channel_axis=2 if is_color else None,
                win_size=win_size,
                data_range=img_original_cv.max() - img_original_cv.min()
            )
        except ValueError as e:
            print(f"Error calculating SSIM for Level {level}: {e}")
            ssim_png = None

        is_identical = np.array_equal(img_original_cv, img_png_compressed_cv)

        results.append([
            "Uploaded Image",
            'PNG (Optimized)',
            f'Level {level}',
            round(png_size_bytes_opt / 1024, 2),
            round(original_size_bytes / png_size_bytes_opt, 2) if png_size_bytes_opt > 0 else float('inf'),
            round(psnr_png, 2) if psnr_png != float('inf') else 'Infinity',
            round(ssim_png, 4) if ssim_png is not None else 'N/A',
            "Ya" if is_identical else "Tidak"
        ])

    headers = ["Citra Input", "Metode Kompresi", "Kualitas/Level", "Ukuran File (KB)", "Rasio Kompresi", "PSNR (dB)", "SSIM", "Identik?"]
    results.insert(0, ["Uploaded Image", 'Original', '-', round(original_size_bytes / 1024, 2), '1', 'Infinity', '1', '-'])

    png_path_vis = os.path.join(output_folder, f"compressed_level9.png")
    img_png_vis = cv2.imread(png_path_vis)
    if is_color:
        img_png_vis = cv2.cvtColor(img_png_vis, cv2.COLOR_BGR2RGB)
    cmap_val = 'gray' if not is_color else None

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_original_cv, cmap=cmap_val)
    axes[0].set_title(f'Original ({original_size_bytes / 1024:.2f} KB)')
    axes[0].axis('off')
    if img_png_vis is not None:
        png_size_vis = os.path.getsize(png_path_vis)
        axes[1].imshow(img_png_vis, cmap=cmap_val)
        axes[1].set_title(f'PNG Level 9 (Optimized) ({png_size_vis / 1024:.2f} KB)')
        axes[1].axis('off')
    else:
        axes[1].set_title('PNG Level 9 (Error Loading)')
        axes[1].axis('off')
    plt.tight_layout()
    comp_comparison_path = alternate_save_plot(fig, "compression_comparison")

    levels = list(range(10))
    file_sizes = [row[3] for row in results[1:]]  # exclude original
    psnrs = [float(row[5]) if row[5] != 'Infinity' else 100 for row in results[1:]]
    ssims = [float(row[6]) if row[6] != 'N/A' else 1 for row in results[1:]]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Level Kompresi PNG')
    ax1.set_ylabel('Ukuran File (KB)', color='tab:blue')
    ax1.plot(levels, file_sizes, color='tab:blue', marker='o', label='Ukuran File (KB)')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    ax2.set_ylabel('PSNR (dB) / SSIM', color='tab:red')
    ax2.plot(levels, psnrs, color='tab:red', marker='x', label='PSNR (dB)')
    ax2.plot(levels, ssims, color='tab:green', marker='s', label='SSIM')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    fig.tight_layout()
    ax1.set_title('Level Kompresi PNG vs Ukuran File, PSNR, dan SSIM')
    fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.grid(True)
    graph_psnr_ssim_path = alternate_save_plot(fig, "graph_png_vs_psnr_ssim")

    labels = ['Original'] + [f'PNG-{lvl}' for lvl in levels]
    sizes = [results[0][3]] + file_sizes

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, sizes, color='skyblue')
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('Ukuran File (KB)')
    ax.set_title('Perbandingan Ukuran File: Original vs PNG Levels')
    ax.grid(axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom')
    plt.tight_layout()
    bar_file_size_path = alternate_save_plot(fig, "bar_file_size_comparison")

    return templates.TemplateResponse("lossless_result.html", {
        "request": request,
        "comp_comparison_path": comp_comparison_path,
        "graph_psnr_ssim_path": graph_psnr_ssim_path,
        "bar_file_size_path": bar_file_size_path,
        "results": results,
        "headers": headers,
        "original_size": round(original_size_bytes / 1024, 2)
    })

@app.get("/texture/", response_class=HTMLResponse)
async def texture_form(request: Request):
    return templates.TemplateResponse("texture.html", {"request": request})

@app.post("/texture/", response_class=HTMLResponse)
async def texture_process(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    image_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return templates.TemplateResponse("texture.html", {
            "request": request,
            "error": "Gambar tidak valid atau tidak dapat dibaca."
        })

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    def compute_texture_statistics(image, window_size=15):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.float32(image)
        mean = cv2.boxFilter(image, -1, (window_size, window_size))
        mean_sqr = cv2.boxFilter(image * image, -1, (window_size, window_size))
        variance = np.maximum(mean_sqr - mean * mean, 0)
        std_dev = np.sqrt(variance)
        norm_maps = {
            'mean': cv2.normalize(mean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            'variance': cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            'std_dev': cv2.normalize(std_dev, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        }
        return norm_maps

    stat_maps = compute_texture_statistics(image_gray)
    stat_images = [stat_maps['mean'], stat_maps['variance'], stat_maps['std_dev']]
    stat_titles = ['Rerata Lokal', 'Variansi Lokal', 'Deviasi Standar Lokal']

    def compute_glcm_features(image, distances, angles, levels=256, symmetric=True, normed=True):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = util.img_as_ubyte(image)
        glcm = graycomatrix(image, distances=distances, angles=angles, levels=levels, symmetric=symmetric, normed=normed)
        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        features = {prop: graycoprops(glcm, prop) for prop in props}
        return features, glcm

    distances = [1, 3, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm_features, glcm = compute_glcm_features(image_gray, distances, angles)
    glcm_image = glcm[:, :, 0, 0]

    # LBP Features
    def compute_lbp(image, radius=3, n_points=24, method='uniform'):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(image, n_points, radius, method)
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        return lbp, hist

    lbp_image, lbp_hist = compute_lbp(image_gray)

    # Gabor Filters
    def compute_gabor_filters(image, frequencies, orientations):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.float32(image)
        filtered_imgs = []
        titles = []
        magnitude = np.zeros_like(image)
        for frequency in frequencies:
            for theta in orientations:
                kernel_size = int(2 * np.ceil(frequency) + 1)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma=frequency/3, theta=theta, lambd=frequency, gamma=0.5, psi=0)
                filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
                magnitude += filtered * filtered
                filtered_norm = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                filtered_imgs.append(filtered_norm)
                titles.append(f'Gabor: f={frequency:.1f}, θ={theta:.1f}')
        magnitude = np.sqrt(magnitude)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        filtered_imgs.append(magnitude)
        titles.append('Gabor Magnitude')
        return filtered_imgs, titles, magnitude

    frequencies = [5, 10, 15]
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    gabor_images, gabor_titles, gabor_magnitude = compute_gabor_filters(image_gray, frequencies, orientations)
    gabor_display = gabor_images[:6] + [gabor_magnitude]
    gabor_display_titles = gabor_titles[:6] + ['Respons Magnitude Gabor']

    # Law's Texture Energy
    def compute_law_texture_energy(image):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.float32(image)
        L5 = np.array([1, 4, 6, 4, 1])
        E5 = np.array([-1, -2, 0, 2, 1])
        S5 = np.array([-1, 0, 2, 0, -1])
        R5 = np.array([1, -4, 6, -4, 1])
        W5 = np.array([-1, 2, 0, -2, 1])
        filters_1d = {'L5': L5, 'E5': E5, 'S5': S5, 'R5': R5, 'W5': W5}
        texture_maps = {}
        for name_i, filter_i in filters_1d.items():
            for name_j, filter_j in filters_1d.items():
                filter_name = f"{name_i}{name_j}"
                filter_2d = np.outer(filter_i, filter_j)
                filtered = cv2.filter2D(image, -1, filter_2d)
                energy = cv2.boxFilter(np.abs(filtered), -1, (15, 15), normalize=True)
                energy_norm = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                texture_maps[filter_name] = energy_norm
        selected_maps = ['L5E5', 'E5S5', 'S5S5', 'R5R5', 'L5S5', 'E5E5']
        selected_images = [texture_maps[name] for name in selected_maps]
        selected_titles = [f'Law: {name}' for name in selected_maps]
        return texture_maps, selected_images, selected_titles

    law_maps, selected_law_images, selected_law_titles = compute_law_texture_energy(image_gray)

    # Save visualizations
    fig1, ax1 = plt.subplots(1, 2, figsize=(10, 5))
    ax1[0].imshow(image_rgb)
    ax1[0].set_title('Citra RGB Asli')
    ax1[0].axis('off')
    ax1[1].imshow(image_gray, cmap='gray')
    ax1[1].set_title('Citra Grayscale')
    ax1[1].axis('off')
    plt.tight_layout()
    comp_comparison_path = alternate_save_plot(fig1, "texture_comparison")

    fig2, ax2 = plt.subplots(1, 3, figsize=(15, 5))
    for i, (img, title) in enumerate(zip(stat_images, stat_titles)):
        ax2[i].imshow(img, cmap='jet')
        ax2[i].set_title(title)
        ax2[i].axis('off')
    plt.tight_layout()
    stat_path = alternate_save_plot(fig2, "texture_stat_maps")

    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 7))
    ax3.imshow(glcm_image, cmap='viridis')
    ax3.set_title('Matriks GLCM (jarak=1, sudut=0)')
    plt.colorbar(ax3.imshow(glcm_image, cmap='viridis'), ax=ax3, label='Frekuensi')
    plt.tight_layout()
    glcm_path = alternate_save_plot(fig3, "texture_glcm")

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 5))
    ax4a.imshow(lbp_image, cmap='jet')
    ax4a.set_title('Peta Tekstur LBP')
    ax4a.axis('off')
    ax4b.bar(range(len(lbp_hist)), lbp_hist)
    ax4b.set_title('Histogram LBP')
    ax4b.set_xlabel('Nilai LBP')
    ax4b.set_ylabel('Frekuensi')
    plt.tight_layout()
    lbp_path = alternate_save_plot(fig4, "texture_lbp")

    fig5, ax5 = plt.subplots(2, 4, figsize=(16, 10))
    for i, (img, title) in enumerate(zip(gabor_display, gabor_display_titles)):
        ax5[i // 4, i % 4].imshow(img, cmap='jet')
        ax5[i // 4, i % 4].set_title(title)
        ax5[i // 4, i % 4].axis('off')
    plt.tight_layout()
    gabor_path = alternate_save_plot(fig5, "texture_gabor")

    fig6, ax6 = plt.subplots(2, 3, figsize=(15, 10))
    for i, (img, title) in enumerate(zip(selected_law_images, selected_law_titles)):
        ax6[i // 3, i % 3].imshow(img, cmap='jet')
        ax6[i // 3, i % 3].set_title(title)
        ax6[i // 3, i % 3].axis('off')
    plt.tight_layout()
    law_path = alternate_save_plot(fig6, "texture_law")

    return templates.TemplateResponse("texture_result.html", {
        "request": request,
        "comp_comparison_path": comp_comparison_path,
        "stat_path": stat_path,
        "glcm_path": glcm_path,
        "lbp_path": lbp_path,
        "gabor_path": gabor_path,
        "law_path": law_path
    })
    
@app.get("/color/", response_class=HTMLResponse)
async def texture_form(request: Request):
    return templates.TemplateResponse("color.html", {"request": request})

@app.post("/color/", response_class=HTMLResponse)
async def color_process(request: Request, file: UploadFile = File(...)):
    # Read and decode image
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    image_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return templates.TemplateResponse("color.html", {
            "request": request,
            "error": "Gambar tidak valid atau tidak dapat dibaca."
        })

    # Convert to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # RGB Channels
    R, G, B = cv2.split(image_rgb)

    # XYZ Color Space
    image_xyz = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2XYZ)
    X, Y, Z = cv2.split(image_xyz)

    # Lab Color Space
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2Lab)
    L, a, b = cv2.split(image_lab)

    # YCbCr Color Space
    image_ycbcr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    Y_ycbcr, Cr, Cb = cv2.split(image_ycbcr)

    # HSV Color Space
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(image_hsv)

    # YIQ Color Space
    def rgb_to_yiq(rgb):
        rgb_norm = rgb.astype(np.float32) / 255.0
        transform_matrix = np.array([
            [0.299, 0.587, 0.114],
            [0.596, -0.274, -0.322],
            [0.211, -0.523, 0.312]
        ])
        height, width, _ = rgb_norm.shape
        rgb_reshaped = rgb_norm.reshape(height * width, 3)
        yiq_reshaped = np.dot(rgb_reshaped, transform_matrix.T)
        yiq = yiq_reshaped.reshape(height, width, 3)
        return yiq

    image_yiq = rgb_to_yiq(image_rgb)
    Y_yiq = image_yiq[:, :, 0]
    I = image_yiq[:, :, 1]
    Q = image_yiq[:, :, 2]

    # Luminance Components
    luminance_components = {
        'Y dari YCbCr': Y_ycbcr,
        'L dari Lab': L,
        'Y dari YIQ': (Y_yiq * 255).astype(np.uint8),  # Scale back to 0-255
        'V dari HSV': V
    }

    # Save Visualizations
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    ax1.imshow(image_rgb)
    ax1.set_title('Citra RGB Asli')
    ax1.axis('off')
    plt.tight_layout()
    original_path = alternate_save_plot(fig1, "color_original")

    fig2, ax2 = plt.subplots(1, 3, figsize=(12, 4))
    ax2[0].imshow(R, cmap='gray')
    ax2[0].set_title('Kanal Red')
    ax2[0].axis('off')
    ax2[1].imshow(G, cmap='gray')
    ax2[1].set_title('Kanal Green')
    ax2[1].axis('off')
    ax2[2].imshow(B, cmap='gray')
    ax2[2].set_title('Kanal Blue')
    ax2[0].get_figure().colorbar(ax2[0].imshow(R, cmap='gray'), ax=ax2[0])
    ax2[1].get_figure().colorbar(ax2[1].imshow(G, cmap='gray'), ax=ax2[1])
    ax2[2].get_figure().colorbar(ax2[2].imshow(B, cmap='gray'), ax=ax2[2])
    plt.tight_layout()
    rgb_channels_path = alternate_save_plot(fig2, "color_rgb_channels")

    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6))
    ax3.imshow(image_xyz)
    ax3.set_title('Citra dalam Ruang Warna XYZ')
    ax3.axis('off')
    plt.tight_layout()
    xyz_path = alternate_save_plot(fig3, "color_xyz")

    fig4, ax4 = plt.subplots(1, 3, figsize=(12, 4))
    ax4[0].imshow(X, cmap='gray')
    ax4[0].set_title('Komponen X')
    ax4[0].axis('off')
    ax4[1].imshow(Y, cmap='gray')
    ax4[1].set_title('Komponen Y (Luminansi)')
    ax4[1].axis('off')
    ax4[2].imshow(Z, cmap='gray')
    ax4[2].set_title('Komponen Z')
    ax4[0].get_figure().colorbar(ax4[0].imshow(X, cmap='gray'), ax=ax4[0])
    ax4[1].get_figure().colorbar(ax4[1].imshow(Y, cmap='gray'), ax=ax4[1])
    ax4[2].get_figure().colorbar(ax4[2].imshow(Z, cmap='gray'), ax=ax4[2])
    plt.tight_layout()
    xyz_channels_path = alternate_save_plot(fig4, "color_xyz_channels")

    fig5, ax5 = plt.subplots(1, 1, figsize=(8, 6))
    ax5.imshow(image_lab)
    ax5.set_title('Citra dalam Ruang Warna Lab')
    ax5.axis('off')
    plt.tight_layout()
    lab_path = alternate_save_plot(fig5, "color_lab")

    fig6, ax6 = plt.subplots(1, 3, figsize=(12, 4))
    ax6[0].imshow(L, cmap='gray')
    ax6[0].set_title('Komponen L (Luminansi)')
    ax6[0].axis('off')
    ax6[1].imshow(a, cmap='gray')
    ax6[1].set_title('Komponen a (Hijau-Merah)')
    ax6[1].axis('off')
    ax6[2].imshow(b, cmap='gray')
    ax6[2].set_title('Komponen b (Biru-Kuning)')
    ax6[0].get_figure().colorbar(ax6[0].imshow(L, cmap='gray'), ax=ax6[0])
    ax6[1].get_figure().colorbar(ax6[1].imshow(a, cmap='gray'), ax=ax6[1])
    ax6[2].get_figure().colorbar(ax6[2].imshow(b, cmap='gray'), ax=ax6[2])
    plt.tight_layout()
    lab_channels_path = alternate_save_plot(fig6, "color_lab_channels")

    fig7, ax7 = plt.subplots(1, 1, figsize=(8, 6))
    ax7.imshow(image_ycbcr)
    ax7.set_title('Citra dalam Ruang Warna YCbCr')
    ax7.axis('off')
    plt.tight_layout()
    ycbcr_path = alternate_save_plot(fig7, "color_ycbcr")

    fig8, ax8 = plt.subplots(1, 3, figsize=(12, 4))
    ax8[0].imshow(Y_ycbcr, cmap='gray')
    ax8[0].set_title('Komponen Y (Luminansi)')
    ax8[0].axis('off')
    ax8[1].imshow(Cb, cmap='gray')
    ax8[1].set_title('Komponen Cb (Chrominance Blue)')
    ax8[1].axis('off')
    ax8[2].imshow(Cr, cmap='gray')
    ax8[2].set_title('Komponen Cr (Chrominance Red)')
    ax8[0].get_figure().colorbar(ax8[0].imshow(Y_ycbcr, cmap='gray'), ax=ax8[0])
    ax8[1].get_figure().colorbar(ax8[1].imshow(Cb, cmap='gray'), ax=ax8[1])
    ax8[2].get_figure().colorbar(ax8[2].imshow(Cr, cmap='gray'), ax=ax8[2])
    plt.tight_layout()
    ycbcr_channels_path = alternate_save_plot(fig8, "color_ycbcr_channels")

    fig9, ax9 = plt.subplots(1, 1, figsize=(8, 6))
    ax9.imshow(image_hsv)
    ax9.set_title('Citra dalam Ruang Warna HSV')
    ax9.axis('off')
    plt.tight_layout()
    hsv_path = alternate_save_plot(fig9, "color_hsv")

    fig10, ax10 = plt.subplots(1, 3, figsize=(12, 4))
    ax10[0].imshow(H, cmap='gray')
    ax10[0].set_title('Komponen Hue')
    ax10[0].axis('off')
    ax10[1].imshow(S, cmap='gray')
    ax10[1].set_title('Komponen Saturation')
    ax10[1].axis('off')
    ax10[2].imshow(V, cmap='gray')
    ax10[2].set_title('Komponen Value')
    ax10[0].get_figure().colorbar(ax10[0].imshow(H, cmap='gray'), ax=ax10[0])
    ax10[1].get_figure().colorbar(ax10[1].imshow(S, cmap='gray'), ax=ax10[1])
    ax10[2].get_figure().colorbar(ax10[2].imshow(V, cmap='gray'), ax=ax10[2])
    plt.tight_layout()
    hsv_channels_path = alternate_save_plot(fig10, "color_hsv_channels")

    fig11, ax11 = plt.subplots(1, 1, figsize=(8, 6))
    ax11.imshow(image_yiq)
    ax11.set_title('Citra dalam Ruang Warna YIQ')
    ax11.axis('off')
    plt.tight_layout()
    yiq_path = alternate_save_plot(fig11, "color_yiq")

    fig12, ax12 = plt.subplots(1, 3, figsize=(12, 4))
    ax12[0].imshow(Y_yiq, cmap='gray')
    ax12[0].set_title('Komponen Y (Luminansi)')
    ax12[0].axis('off')
    ax12[1].imshow(I, cmap='gray')
    ax12[1].set_title('Komponen I (In-phase)')
    ax12[1].axis('off')
    ax12[2].imshow(Q, cmap='gray')
    ax12[2].set_title('Komponen Q (Quadrature)')
    ax12[0].get_figure().colorbar(ax12[0].imshow(Y_yiq, cmap='gray'), ax=ax12[0])
    ax12[1].get_figure().colorbar(ax12[1].imshow(I, cmap='gray'), ax=ax12[1])
    ax12[2].get_figure().colorbar(ax12[2].imshow(Q, cmap='gray'), ax=ax12[2])
    plt.tight_layout()
    yiq_channels_path = alternate_save_plot(fig12, "color_yiq_channels")

    fig13, ax13 = plt.subplots(2, 2, figsize=(12, 8))
    i = 0
    for name, component in luminance_components.items():
        ax13[i // 2, i % 2].imshow(component, cmap='gray')
        ax13[i // 2, i % 2].set_title(name)
        ax13[i // 2, i % 2].axis('off')
        i += 1
    plt.tight_layout()
    luminance_path = alternate_save_plot(fig13, "color_luminance")

    return templates.TemplateResponse("color_result.html", {
        "request": request,
        "original_path": original_path,
        "rgb_channels_path": rgb_channels_path,
        "xyz_path": xyz_path,
        "xyz_channels_path": xyz_channels_path,
        "lab_path": lab_path,
        "lab_channels_path": lab_channels_path,
        "ycbcr_path": ycbcr_path,
        "ycbcr_channels_path": ycbcr_channels_path,
        "hsv_path": hsv_path,
        "hsv_channels_path": hsv_channels_path,
        "yiq_path": yiq_path,
        "yiq_channels_path": yiq_channels_path,
        "luminance_path": luminance_path
    })
