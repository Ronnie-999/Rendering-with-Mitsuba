import drjit as dr
import mitsuba as mi

mi.set_variant('cuda_ad_rgb')

scene = mi.load_file(r'C:\Users\HTCV_DIRME\Desktop\Projectwork\scene.xml', res=128, integrator='prb')
