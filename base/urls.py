from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name='home'),
    path("predict/",views.predictpage, name='predictpage'),
    path("modelHistory/",views.modelHistory, name='modelHistory'),
    path("bitcoinData/",views.bitcoinData, name='bitcoinData'),
    path("otherCrypto/",views.otherCrypto, name='otherCrypto'),
    path("about/",views.about, name='about'),

]