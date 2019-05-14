from django.conf.urls import include, url
from rest_framework import routers
from api import views

# 定义路由地址
#route = routers.DefaultRouter()

# 注册新的路由地址
#route.register(r'singlePic', views.get_single_score)

# 注册上一级的路由地址并添加
urlpatterns = [
    url('singlePic', views.get_single_score),
    #url('api/', include(route.urls)),
]
