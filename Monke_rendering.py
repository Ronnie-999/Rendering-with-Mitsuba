import drjit as dr
import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt

mi.set_variant('cuda_ad_rgb')

scene = mi.load_file(r'C:\Users\HTCV_DIRME\Desktop\Projectwork\Blender_xml\Swimming_monke.xml')

print('I am here')

initial_image = mi.render(scene, spp=512)
initial_image_8bit = (np.clip(initial_image, 0, 1) * 255).astype(np.uint8)
bitmap = mi.util.convert_to_bitmap(initial_image_8bit)
bitmap.write(r'C:\Users\HTCV_DIRME\Desktop\Projectwork\rendered and invrend images\initial_rendered_image.png')
print('Initial image rendered and saved')

mi.util.convert_to_bitmap(initial_image)

params = mi.traverse(scene)
#print(params)
keys = {
        'Monke_color' : 'elm__8.bsdf.brdf_0.base_color.value',
       'Sphere_color' : 'elm__6.bsdf.brdf_0.base_color.value', 
        'Monke_metallic' : 'elm__6.bsdf.brdf_0.metallic.value' 
        }
original_values = {
    key: params[keys[key]] if 'color' not in key else mi.Color3f(params[keys[key]])
    for key in keys
}
params[keys['Sphere_color']] = mi.Color3f(0.7, 0.2, 0.1)  
params[keys['Monke_metallic']] = 0.5  
params[keys['Monke_color']] = mi.Color3f(0.8, 0.3, 0.5)  
params.update()
print(f"Updated parameters: Monkey Metallic = {params[keys['Monke_metallic']]}, "
      f"Monke Color = {params[keys['Monke_color']]}, "
      f"Sphere Color = {params[keys['Sphere_color']]}")

updated_image = mi.render(scene, spp=512)
updated_image_8bit = (np.clip(updated_image, 0, 1) * 255).astype(np.uint8)
bitmap = mi.util.convert_to_bitmap(updated_image_8bit)
bitmap.write(r'C:\Users\HTCV_DIRME\Desktop\Projectwork\rendered and invrend images\updated_image_with_color_and_metallic_properties.png')
print('Updated image rendered and saved')
mi.util.convert_to_bitmap(updated_image_8bit)

opt = mi.ad.Adam(lr=0.05)
for key in keys:
    opt[keys[key]] = params[keys[key]]

params.update(opt)

def mse(image):
    return dr.mean(dr.sqr(image - initial_image))

iterations = 50
error = {key: [] for key in keys}
intermediate_images = [initial_image_8bit]

for it in range(iterations):
    image = mi.render(scene, params, spp=4)
    loss = mse(image)
    
    print(f"Iteration {it}: Loss = {loss}")  # Check loss value

    dr.backward(loss)
    opt.step()

    # Update and clamp parameter values
    for key in keys:
        if 'color' not in key:
            opt[keys[key]] = dr.clamp(opt[keys[key]], 0.0, 1.0)  # Clamp scalar parameters
        else:
            opt[keys[key]] = mi.Color3f(
                dr.clamp(opt[keys[key]][0], 0.0, 1.0),
                dr.clamp(opt[keys[key]][1], 0.0, 1.0),
                dr.clamp(opt[keys[key]][2], 0.0, 1.0)
            )  # Clamp color values
        params.update(opt)

    # Compute parameter errors
    for key in keys:
        if 'color' not in key:
            err_ref = dr.sum(dr.sqr(original_values[key] - params[keys[key]]))
        else:
            err_ref = dr.sum(dr.sqr(original_values[key] - mi.Color3f(params[keys[key]])))
        print(f"Iteration {it:02d}: {key} parameter error = {err_ref[0]:6f}")
        error[key].append(err_ref)

    # Save intermediate images every 5 iterations
    if (it + 1) % 5 == 0:
        intermediate_image_8bit = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        intermediate_images.append(intermediate_image_8bit)
        save_path = rf'C:\Users\HTCV_DIRME\Desktop\Projectwork\rendered and invrend images\intermediate_image_{it + 1}.png'
        bitmap = mi.util.convert_to_bitmap(intermediate_image_8bit)
        bitmap.write(save_path)
        print(f'\nIntermediate image saved at iteration {it + 1}: {save_path}')

final_image = mi.render(scene, spp=512)
final_image_8bit = (np.clip(final_image, 0, 1) * 255).astype(np.uint8)

bitmap = mi.util.convert_to_bitmap(final_image_8bit)
bitmap.write(r'C:\Users\HTCV_DIRME\Desktop\Projectwork\rendered and invrend images\recovered_image_with_color_and_metallic_property.png')
print('\nFinal image rendered and saved')
mi.util.convert_to_bitmap(final_image_8bit)

plt.figure(figsize=(10, 6))
for key in keys:
    plt.plot(error[key], marker='o', label=key)
plt.xlabel('Iteration')
plt.ylabel('MSE(param)')
plt.title('Parameter Error Plot')
plt.legend()
plt.grid(True)

# Save the error plot
save_path_error_plot = r'C:\Users\HTCV_DIRME\Desktop\Projectwork\rendered and invrend images\parameter_error_plot_with_color.png'
plt.savefig(save_path_error_plot, format='png', dpi=300)
plt.show()

print(f"Error plot saved at: {save_path_error_plot}")