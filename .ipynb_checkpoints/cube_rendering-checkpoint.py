import drjit as dr
import mitsuba as mi
import numpy as np

mi.set_variant('cuda_ad_rgb')

scene = mi.load_file(r'C:\Users\HTCV_DIRME\Desktop\Projectwork\Blender_xml\Swimming_monke.xml')

print('I am here')

image = mi.render(scene, spp =512)

#bitmap = mi.util.convert_to_bitmap(image)
image_8bit = (np.clip(image, 0, 1) * 255).astype(np.uint8)
bitmap = mi.util.convert_to_bitmap(image_8bit)
bitmap.write(r'C:\Users\HTCV_DIRME\Desktop\Projectwork\blender_images\rendered_reference_img_1.png')
print('Image rendered')

params = mi.traverse(scene)

print(params)