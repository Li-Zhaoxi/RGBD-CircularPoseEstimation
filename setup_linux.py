from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np


opencv_include = '/home/expansion/lizhaoxi/anaconda3/envs/pcd/include/'
opencv_lib_dirs = '/home/expansion/lizhaoxi/anaconda3/envs/pcd/lib/'
opencv_libs = ['opencv_core', 'opencv_highgui', 'opencv_imgproc', 'opencv_imgcodecs']

print('opencv_include: ', opencv_include)
print('opencv_lib_dirs: ', opencv_lib_dirs)
print('opencv_libs: ', opencv_libs)




class custom_build_ext(build_ext):
    def build_extensions(self):
        build_ext.build_extensions(self)

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


ext_modules = [
    Extension(
        "lib.cpp_tools",
        ["./cpp/CalculateOverlap.cpp",
         "./cpp/CalculateRangeAtY.cpp",
         "./cpp/CalculateRangeOfY.cpp",
         "./cpp/ELPShape2Equation.cpp",
         "./cpp/fasterCalculateOverlap.cpp",
         "./cpp/GetCirclePos.cpp",
         "./cpp/FitCircle.cpp",
         "./cpp/cpp_tools.pyx",],
        include_dirs = [numpy_include, opencv_include],
        language='c++',
        libraries=opencv_libs,
        library_dirs=[opencv_lib_dirs]
        ),
    Extension(
        "lib.pyAAMED",
        ["./cpp/AAMED/adaptApproximateContours.cpp",
         "./cpp/AAMED/adaptApproxPolyDP.cpp",
         "./cpp/AAMED/Contours.cpp",
         "./cpp/AAMED/EllipseNonMaximumSuppression.cpp",
         "./cpp/AAMED/FLED.cpp",
         "./cpp/AAMED/FLED_drawAndWriteFunctions.cpp",
         "./cpp/AAMED/FLED_Initialization.cpp",
         "./cpp/AAMED/FLED_PrivateFunctions.cpp",
         "./cpp/AAMED/Group.cpp",
         "./cpp/AAMED/LinkMatrix.cpp",
         "./cpp/AAMED/Node_FC.cpp",
         "./cpp/AAMED/Segmentation.cpp",
         "./cpp/AAMED/Validation.cpp",
         "./cpp/AAMED/aamed.pyx"],
        include_dirs = [numpy_include, opencv_include],
        language='c++',
        libraries=opencv_libs,
        library_dirs=[opencv_lib_dirs]
        ),
    ]

setup(
    name='cpp_tools',
    ext_modules=ext_modules,
    cmdclass={'build_ext': custom_build_ext},
)

print('Build done')
