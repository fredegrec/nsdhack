import streamlit as st
import pandas as pd
import io
# from io import StringIO
import numpy as np

import os
import pathlib
import sys
import base64
from PIL import Image

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import cv2

import pytesseract

def get_df(path_):
    # Load model
    config_file = '/content/CascadeTabNet/Config/cascade_mask_rcnn_hrnetv2p_w32_20e.py'
    checkpoint_file = '/content/epoch_36.pth'
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # Test a single image 
    #path_ = "vypiska_2.png"#"/content/CascadeTabNet/Demo/demo.png"
    image = cv2.imread(path_)
    div = 3000//image.shape[0]
    image=cv2.resize(image,dsize=(image.shape[1]*div, image.shape[0]*div), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 210, 255,cv2.THRESH_BINARY)[1]
    #kernel = np.ones((3,3),np.uint8)
    #gray = cv2.dilate(gray, kernel, iterations=3)
    #plt.figure(figsize=(20, 30))
    #plt.imshow(gray)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    #cv2.imwrite('img.png', image)
    # Run Inference
    result = inference_detector(model, gray)

    result = result[0]
    full_border = len(result[0]) != 0
    result = [x for x in result if len(x) !=0]
    result = sorted(result, key=len)
    res = [int(x) for x in result[0][0]]
    table = image[res[1]:res[3],res[0]:res[2]]
    cells = [[int(cell[0] - res[0]), int(cell[1] - res[1]),
              int(cell[2] - res[0]), int(cell[3] - res[1])] for cell in result[-1]]
    cells = [cell for cell in cells if cell[0] >= 0 and cell[1] >=0]

    if full_border:
      img = table

      im = cv2.imread(path_)[res[1]//div:res[3]//div,res[0]//div:res[2]//div]
      im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
      dst = cv2.adaptiveThreshold(~im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)

      horizontal = dst.copy()
      vertical = dst.copy()
      scale = 13 # play with this variable in order to increase/decrease the amount of lines to be detected

      # Specify size on horizontal axis
      #print(horizontal.shape)
      horizontalsize = horizontal.shape[1] // scale
      horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
      horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
      horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))
      #plt.figure(figsize=(20, 30))
      #plt.imshow( horizontal)

      # vertical
      verticalsize = vertical.shape[0] // scale
      verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
      vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
      vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
      #plt.figure(figsize=(20, 30))
      #plt.imshow( vertical)

      # table line
      grid = horizontal + vertical
    else:
      im = table.copy()
      gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
      im = cv2.threshold(im, 170, 255,cv2.THRESH_BINARY)[1]

      grid = np.zeros(im.shape, np.uint8)
      arr = np.zeros(im.shape[1])
      for cell in cells:
        #print(cell)
        arr[cell[0]:cell[2]] += 1

      start = 0
      end = 0
      intervals = []
      for i in range(len(arr)-1):
        if start is not None and (arr[i] == arr[i-1]) and arr[i+1] > arr[i]:
          end = i
          interval = int((end + start)/2)
          print(interval)
          cv2.line(grid, (interval, 0),
                        (interval, im.shape[0]), (255,255,255),3)
          start = None
        if arr[i-1] > arr[i] and arr[i+1] == arr[i]:
          start = i

      end = len(arr) - 1

      arr = np.zeros(im.shape[0])
      for cell in cells:
        #print(cell)
        arr[cell[1]:cell[3]] += 1

      start = 0
      end = 0
      intervals = []
      prev_start = arr[0]
      for i in range(1, len(arr)-1):
        if start is not None and (arr[i] == arr[i-1]) and arr[i+1] > arr[i]:
          end = i
          interval = int((end + start)/2)
          print(interval)
          cv2.line(grid, ( 0, interval),
                        (im.shape[1], interval), (0,255,0),3)
          start = None
        if arr[i-1] > arr[i] and arr[i+1] == arr[i]:
          start = i

      end = len(arr) - 1
      #interval = int((end + start)/2)
      cv2.line(grid, ( 0, im.shape[0]),
                        (im.shape[1], im.shape[0]), (0,255,0),3)
      
    
    contours,hierarchy = cv2.findContours(grid,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, find the bounding rectangle and draw it
    cells = []
    for cnt in contours:
      x,y,w,h = cv2.boundingRect(cnt)

      if w < 0.9*img.shape[1] and h < 0.95*img.shape[0]:
        cells.append((x, y, w, h))
        #print(x, y, w, h)
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)

    #print(sorted(lines, key=lambda x: x[0]))
    img = table.copy()#cv2.imread(path_)
    div1 = 12000//3000
    div *= div1
    print(div)
    image=cv2.resize(img,dsize=(img.shape[1]*div1, img.shape[0]*div1), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.threshold(gray, 170, 255,
    cv2.THRESH_BINARY)[1]

    i=0

    clusters = []
    for cl in sorted(cells, key=lambda x: (x[1])):
      if cl[2] < 5 or cl[3] < 5:
        continue
      if len(clusters) == 0:
        clusters.append([cl])
      elif abs(cl[1] - clusters[-1][0][1]) < 3 :
        clusters[-1].append(cl) 
      else:
        clusters.append([cl])

    res_text = []
    for cluster in clusters:
      cluster = sorted(cluster, key=lambda x: (x[0]))
      cur_text = []
      for (x, y, w, h) in cluster:
        cell_ = image[y*div:y*div+h*div, x*div:x*div+w*div]
        text = ' '.join([phrase.strip() for phrase in pytesseract.image_to_string(cell_, lang='rus').split('\n')])
        print(text)
        cur_text.append(text)
      res_text.append('|'.join(cur_text))

    return res_text

# @st.cache
def tesseract_results(img, selected_languages):
  return pytesseract.image_to_string(img, lang=selected_languages)

# @st.cache
def show_image_results(original, tesseract, grumpy_ocr, **kwargs):
    col1, col2, col3 = st.beta_columns(3)
    col1.header("Original")
    col1.image(original, use_column_width=True)
    
    col2.header("Tesseract")
    # col2.text()
    col2.write(pd.DataFrame([x for x in tesseract.split('\n')]))
    
    col3.header("Grumpy OCR")
    col3.write(grumpy_ocr)

# def read_dataframe(filename: str) -> pd.DataFrame:
#     return pd.read_excel(filename)

# st.title('Grumpy OCR')
LOGO_IMAGE = '/content/grumpy_logo.png'

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    .container {
        display: flex;
        align-items: center;
        justify-content: center
    }
    .logo-text {
        font-weight:700 !important;
        font-size:100px !important;
    }
    .logo-img {
        float:right;
        height: 150px;
        width: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="container">
        <img class="logo-img" 
        src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
        <p class="logo-text">GrumpyOCR</p>
    </div>
    """,
    unsafe_allow_html=True
)

tesseract_version = pytesseract.get_tesseract_version()
tesseract_languages = pytesseract.get_languages(config='')
st.markdown(f"Version: {tesseract_version}, Langs: {tesseract_languages}")

st.sidebar.markdown('### Настройки')
languages = ['eng+rus', 'rus', 'eng']
selected_languages = st.sidebar.selectbox('Язык', languages)


# Выбор файла с признаками обязателен для дальнейшего выполнения.
st.markdown("### 1 Выберите или перенесите файл с текстом")
upload_file_object = st.file_uploader('', ['pdf', 'png', 'jpg', 'jpeg'])
if not upload_file_object:
    st.stop()

# Небольшой трюк с кэшированием в файл.
filename = upload_file_object.name
if not upload_file_object.closed:
    print(f'Dumping file to cache {filename}...')
    with open(filename, 'wb') as outfile:
        outfile.write(upload_file_object.getvalue())

st.markdown("### 2 Результат преобразования")

if filename.endswith('pdf'):
    import pdf2image
    images = pdf2image.convert_from_path(filename)
    
    for page_num, img in enumerate(images):
        path = './name.png'
        img.save(path)
        st.markdown(f"**Страница #{page_num + 1}**")
        text_representation = pytesseract.image_to_string(img, lang=selected_languages)
        grumpy_ocr = pd.DataFrame(data=get_df(path))
        show_image_results(img, text_representation, grumpy_ocr)
        # st.text(text_representation)
else:
    img = Image.open(filename)
    text_representation = pytesseract.image_to_string(filename, lang=selected_languages)
    # st.text(text_representation)
    grumpy_ocr = pd.DataFrame(data=get_df(filename))#pd.DataFrame(data=np.random.randn(10, 10))
    show_image_results(img, text_representation, grumpy_ocr)
# data_load_state = st.text('Loading data...')

st.stop()

