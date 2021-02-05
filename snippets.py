# Authored by Aparna <aparnaabhi2@gmail.com>

'''
Django reusable snippets
'''

import hashlib
import imghdr
import time
from io import BytesIO

from django.conf import settings
from django.contrib.auth import authenticate, login, logout
from django.http import JsonResponse
from django.middleware.csrf import get_token
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, parser_classes, \
    permission_classes
from rest_framework.parsers import JSONParser
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.status import HTTP_401_UNAUTHORIZED


# login user
@csrf_exempt
@permission_classes((AllowAny,))
@api_view(["POST"])
def login_user(request):
    username = request.data.get("username")
    password = request.data.get("password")

    user = authenticate(username=username, password=password)
    if user is not None:
        # the password verified for the user
        if user.is_active:
            # request.session.set_expiry(0)  # sets the exp. value of the session
            login(request, user)
            return Response(
                {
                    "success": "logged in",
                    "user_detail": {
                        "username": user.username,
                        "user_id": user.id,
                    },
                }
            )

        else:
            return Response({"error": "user is not active or removed"})

    return Response(
        {"error": "Login failed,please check username or password"},
        status=HTTP_401_UNAUTHORIZED,
    )


# login authentication API

@csrf_exempt
def now(request):
    get_token(request)
    return JsonResponse(
        {"now": time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime())}
    )


# logout user


# logout authentication API
@csrf_exempt
@parser_classes((JSONParser,))
@api_view(["POST"])
def logout_user(request):
    logout(request)
    return Response({"success": "logged out"})


# Get logged In user
# me user
@csrf_exempt
@api_view(["GET"])
def me(request):
    if request.user.is_authenticated:
        try:
            user_details = User.objects.get(
                username__exact=request.user.username
            )
            response = dict(
                message="success",
                user_name=user_details.username,
            )
            return Response(response)
        except User.DoesNotExist:
            return Response("user does not exists")

    return Response({"error": "user not logged in"},
                    status=HTTP_401_UNAUTHORIZED)


'''
GCP Image Text Detection
'''
from google.cloud import vision
from google.cloud.vision import types
from PIL import Image as PILImage

image = PILImage.open(file_path)
buffer = BytesIO()
image.save(buffer, format="JPEG")
image_bytes = buffer.getvalue()
image = types.Image(content=image_bytes)
image_request = {
    "image": image,
    "features": [
        {"type": vision.enums.Feature.Type.DOCUMENT_TEXT_DETECTION}],
}
client = vision.ImageAnnotatorClient()
batch_response = client.annotate_image(image_request)
input = batch_response.full_text_annotation.text
input_list.append(input.replace('\n', ' '))

# run google vision , get boundary of text
client = vision.ImageAnnotatorClient()
bounds = []
with io.open(img_path, 'rb') as image_file:
    content = image_file.read()

image = types.Image(content=content)
response = client.document_text_detection(image=image)
document = response.full_text_annotation

page = document.pages[0]
width = page.width
height = page.height
image_dimension = (width, height)
for block in page.blocks:
    bounds.append(block.bounding_box)
for idx, bound in enumerate(bounds):
    image.crop([bound.vertices[0].x, bound.vertices[0].y,
                bound.vertices[2].x,
                bound.vertices[2].y])

'''Azure image Text detection'''

image = PILImage.open(file_path)
buffer = BytesIO()
image.save(buffer, format="JPEG")
image_bytes = buffer.getvalue()

response = requests.post(
    "https://centralindia.api.cognitive.microsoft.com/vision/v2.0/recognizeText?mode=Printed",
    headers=headers,
    data=image_bytes
)
response_api = response.headers['Operation-Location']

# read image and store in GCP bucket
# to read image from start
image.seek(0)
image_hash_value = hashlib.sha256(image.read()).hexdigest()  # to covert image to hashvalue
# to get image type
image_type = imghdr.what(None, h=image.read())

'''GCP Storage'''

gcs_client = storage.Client()
bucket = gcs_client.bucket(settings.GS_BUCKET_NAME)
image_blob = bucket.blob(f"{image_hash_value}")
if not image_blob.exists():
    image.seek(0)
    image_blob.upload_from_string(
        image.read(), content_type=f"image/{image_type}"
    )
    image_blob.make_public()  # to get public url to download image

''' Google tramnslator'''
client = translate.Client()
translation = client.translate(text, target_language, input_language)
translated_text = translation['translatedText']

'''image manipulation'''
# numpy
# find max width
max_width = image_list[0].shape[1]
for img in image_list:
    if img.shape[1] > max_width:
        max_width = img.shape[1]
# pad white space to the right of all images
for i, img in enumerate(image_list):
    if not img.shape[1] == max_width:
        blank = np.full(shape=(img.shape[0], max_width - img.shape[1], 3),
                        fill_value=255, dtype='uint8')
        image_list[i] = np.concatenate([blank, img], axis=1)

stitched_image = np.concatenate(image_list, axis=0)

'''CELERY for parallel processing'''
from celery import group

# from django_celery_results.models import TaskResult ->to track task results in db

jobs = group(tasks)
response_task = jobs.apply_async()
response_data = response_task.join(interval=0.1)

'''
Ecommerce website Django pagination
'''

from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

page = request.GET.get("page", 1)
size = request.GET.get('size', 10)

paginator = Paginator(gtin_list, len(gtin_list))
try:
    products = paginator.page(page)
except PageNotAnInteger:
    page = 1
    products = paginator.page(page)
except EmptyPage:
    page = paginator.num_pages
    products = paginator.page(page)

# products = list
total_pages = paginator.num_pages

''' Image and Files'''

from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage

try:
    im = PILImage.open(BytesIO(image_data)).convert("RGB")
except (ValueError, OSError):
    return False
im = resize_im(im, 480)

# image url to image
content = requests.get(img_url).content
print(requests.get(img_url).status_code)
file_content = ContentFile(content)

# read file in django
file = request.FILES.get("csv")
fs = FileSystemStorage(location="media/csv")
filename = fs.save(file.name, file)
with open(path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for idx, row in enumerate(reader):
        pass

# image file to hashvalue to validate existing file
file = request.FILES.get("img_files")
m = hashlib.sha256()
m.update(file.read())
hash_value = m.digest()


def split_image(image_dimension, img_path, bounds):
    half_width = image_dimension[0] / 2
    image_file = io.open(img_path, 'rb')
    im = PILImage.open(image_file)
    for idx, bound in enumerate(bounds):
        tmp1 = half_width - bound.vertices[0].x
        tmp2 = bound.vertices[1].x - half_width


def get_image_extension(image_data: bytes) -> str:
    valid_types = ["jpeg", "gif", "png", "tiff", "web/p"]
    magic_output = magic.from_buffer(image_data)
    magic_output = str(magic_output).lower()
    if not any(x in magic_output for x in valid_types):
        return "jpeg"
    extension = [x for x in valid_types if x in magic_output][0]
    extension = extension.replace("/", "")
    return extension


def image_http_response(image_data, file_name):
    ext = get_image_extension(image_data)
    response = HttpResponse(content_type=f"image/{ext}")
    response["Content-Disposition"] = f"attachment; " f"filename={file_name}.{ext}"
    response.write(image_data)
    return response


# scrapping

from typing import AnyStr, Union

import requests
from parsel import Selector

from w3lib.html import (
    remove_tags_with_content,
)


def download(self, url: AnyStr) -> Union[AnyStr, None]:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:64.0) Gecko/20100101 Firefox/64.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    try:
        response = requests.get(url=url, headers=headers)
        response.raise_for_status()
    except Exception:
        return

    return remove_tags_with_content(
        text=response.text, which_ones=("script",)
    )


website = requests.get(url, headers=headers, proxies=proxies, verify=False, stream=True)
if website.status_code == 200:
    website_sel = Selector(
        text=website.text)
data = website_sel.css('div#id::text').get
data_list = website_sel.css(
    'div.class a::text').getall()
page_sel.css(
    'table.description tr th::text').getall()

#django Custom Commands
import csv
from django.core.management.base import BaseCommand
from tqdm import tqdm
class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument(
            'csv',
            help='Path to a CSV file import. '
        )

    def handle(self, *args, **options):
        with open(options['csv'], mode='r') as f:
            reader = csv.DictReader(f)
            for idx, row in tqdm(enumerate(reader)):
                result = dict(row)