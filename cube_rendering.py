import drjit as dr
import mitsuba as mi
import numpy as np
import os

mi.set_variant('cuda_ad_rgb')

scene = mi.load_file(r'C:\\Users\\HTCV_DIRME\\Desktop\\Projectwork\\Blender_xml\\Swimming_monke.xml')

print('I am here')

image = mi.render(scene, spp=512)

mi.util.convert_to_bitmap(image)

params = mi.traverse(scene)
print(params)
key = 'elm__8.bsdf.brdf_0.base_color.value'
params_ref = mi.Color3f(params[key])
print(params_ref)
params[key] = mi.Color3f(0.9, 0.3, 0.5)
params.update()

rendered_img = mi.render(scene, spp=128)
mi.util.convert_to_bitmap(rendered_img)

opt = mi.ad.Adam(lr=0.1)
opt[key] = params[key]
params.update(opt)

def mse(image1):
    return dr.mean(dr.sqr(image1 - image))

iteration_count = 50
print("full")

errors = []
for it in range(iteration_count):
    image1 = mi.render(scene, params, spp=4)
    loss = mse(image1)
    dr.backward(loss)
    opt.step()
    opt[key] = dr.clamp(opt[key], 0.0, 1.0)
    params.update(opt)
    err_ref = dr.sum(dr.sqr(params_ref - params[key]))
    print(f"Iteration{it:02d}: parameter error = {err_ref[0]:6f}", end='\r')
    errors.append(err_ref)
    print("\n Optimization complete.")

# Render the final image
image_final = mi.render(scene, spp=128)

# Convert to bitmap and save
bitmap = mi.util.convert_to_bitmap(image_final)

# Define the output directory
output_directory = r"C:\\Users\\HTCV_DIRME\\Desktop\\Projectwork\\Rendered_Images"
os.makedirs(output_directory, exist_ok=True)

# Save the image
output_file = os.path.join(output_directory, "lr=0.1.png")
bitmap.write(output_file)

print(f"Final image saved to {output_file}")
