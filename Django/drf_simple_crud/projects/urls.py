from rest_framework import routers
from .api import ProjectViewSet

router = routers.DefaultRouter()

router.register('api/projects', ProjectViewSet, 'projects') # urls api crear con router

urlpatterns = router.urls # generar la urls
