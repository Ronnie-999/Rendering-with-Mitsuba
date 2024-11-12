import drjit as dr
import mitsuba as mi

mi.set_variant('cuda_ad_rgb')

# Load the scene without the 'res' parameter
scene = mi.load_file(r"C:/Users/HTCV_DIRME/Downloads/scene.xml", integrator='prb')

# Render the reference image
image_ref = mi.render(scene, spp=512)

# Preview the reference image
mi.util.convert_to_bitmap(image_ref)

# Access and modify parameters
params = mi.traverse(scene)
key = 'red.reflectance.value'

# Save the original value
param_ref = mi.Color3f(params[key])

# Set another color value and update the scene
params[key] = mi.Color3f(0.01, 0.2, 0.9)
params.update()

# Render and preview the modified image
image_init = mi.render(scene, spp=128)
mi.util.convert_to_bitmap(image_init)
