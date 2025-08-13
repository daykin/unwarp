import cv2
import sympy
from sympy import symbols
from sympy.abc import x, y
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input')
parser.add_argument('--output', default='unwarped_out.tiff')
parser.add_argument('--theta', type=float, default=10.0, help='Rotation angle in degrees')
parser.add_argument('--fx', type=float, default=0.000100)
parser.add_argument('--fy', type=float, default=0.000200)
parser.add_argument('--cx', type=int, default=1024)
parser.add_argument('--cy', type=int, default=768)
args = parser.parse_args()

warped = cv2.imread(args.input)

theta = np.deg2rad(args.theta)  # rotation angle in radians

fx = args.fx
fy = args.fy

cx = args.cx
cy = args.cy

xo, yo, xk, yk, xr, yr = symbols('xo yo xk yk xr yr')
xk = xo-cx
yk = yo-cy

solution = sympy.solve([xr + (fx*xr*xr) + (fy*xr*yr)-xk,\
                        yr + (fx*yr*xr) + (fy*yr*yr)-yk], (xr, yr), dict=True)

print(solution)

s0 = sympy.lambdify((xo, yo), (solution[0][xr], solution[0][yr]))
s1 = sympy.lambdify((xo, yo), (solution[1][xr], solution[1][yr]))

def unwarp_pixel(x_in, y_in, soln):   
    try:
        #shift+undo quadratic mess
        xr_sub, yr_sub = soln(x_in, y_in)

        #un-rotate
        un_theta = -theta
        xr_rot = xr_sub*np.cos(un_theta) - yr_sub*np.sin(un_theta)
        yr_rot = xr_sub*np.sin(un_theta) + yr_sub*np.cos(un_theta)
        #back to image coordinates

        xu = int(xr_rot+cx)
        yu = int(yr_rot+cy)
    except:
        xu, yu = -1, -1
    return xu, yu



def unwarp(image, soln):
    h, w = image.shape[:2]
    unwarped = np.zeros_like(image)
    for y in range(h):
        for x in range(w):
            xr, yr = unwarp_pixel(x, y, soln)
            if 0 <= xr < w and 0 <= yr < h:
                unwarped[y, x] = image[yr, xr]
    return unwarped

#want the positive root
unwarped_1 = unwarp(warped, s1)

cv2.imwrite(args.output, unwarped_1)