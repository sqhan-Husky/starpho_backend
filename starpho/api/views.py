from django.shortcuts import render
from django.http import HttpResponse
import os
from static.get_single_score import get_s_score


# Create your views here.
def get_single_score(request):
    if request.method == 'POST':
        obj = request.FILES.get('img')
        path = os.path.join('api/img/', obj.name)
        f = open(path, 'wb')
        for line in obj.chunks():
            f.write(line)
        f.close()

        score = get_s_score(obj.name)

        os.remove(path)

    return HttpResponse(score)


'''
def upload_and_get_res(request):
    if request.method == 'GET':
        return HttpResponse("服务器不接受GET请求!")
    else:
        #获取图像数据信息
        image_file = request.FILES.get('file')
        # file_name = image_file.name
        # file_size = image_file.size
        f = open('123', 'wb')
        for chunk in image_file.chunks():
            f.write(chunk)
        f.close() 
'''