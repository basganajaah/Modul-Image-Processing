import os
from uuid import uuid4
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from skimage.exposure import match_histograms
from IPython.display import display, Image
from io import BytesIO
from PIL import Image as PILImage
from typing import List

import numpy as np
import cv2
import matplotlib.pyplot as plt

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

    # Pastikan gambar berhasil diimpor
    if img is None:
        return HTMLResponse("Tidak dapat membaca gambar yang diunggah", status_code=400)

    # Buat histogram grayscale dan berwarna
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
    # Menampilkan halaman untuk upload gambar untuk equalisasi histogram
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
    # Baca gambar yang diunggah dan gambar referensi
    image_data = await file.read()
    ref_image_data = await ref_file.read()

    np_array = np.frombuffer(image_data, np.uint8)
    ref_np_array = np.frombuffer(ref_image_data, np.uint8)
		
	#jika ingin grayscale
    #img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
    #ref_img = cv2.imdecode(ref_np_array, cv2.IMREAD_GRAYSCALE)

    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # Membaca gambar dalam format BGR
    ref_img = cv2.imdecode(ref_np_array, cv2.IMREAD_COLOR)  # Membaca gambar referensi dalam format BGR


    if img is None or ref_img is None:
        return HTMLResponse("Gambar utama atau gambar referensi tidak dapat dibaca.", status_code=400)

    # Spesifikasi histogram menggunakan match_histograms dari skimage #grayscale
    # specified_img = match_histograms(img, ref_img, multichannel=False)
	
    # Spesifikasi histogram menggunakan match_histograms dari skimage untuk gambar berwarna
    specified_img = match_histograms(img, ref_img, channel_axis=-1)
    # Konversi kembali ke format uint8 jika diperlukan
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
    
    original_path = save_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "original")
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