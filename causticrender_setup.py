#Met deze code kan je een lens renderen in Mitsuba op basis van een .ply bestand
# dat je hebt gemaakt met de Poisson-methode of via optimaal transport. 
# De code zal automatisch de afstand van de muur en de hoeken van de lens uit de bestandsnaam halen

import os
import re
from os.path import realpath
import mitsuba as mi
import matplotlib.pyplot as plt


mi.set_variant('llvm_ad_rgb') 


LENS_FILE = realpath(r"C:......") #pad naar de zojuist aangemaakte lens .ply file
lens_basisnaam = os.path.splitext(os.path.basename(LENS_FILE))[0]
RENDER_RESOLUTION = (512, 512)
SAMPLES_PER_PIXEL = 2048 #hoger is betere kwaliteit render
scene_set = re.search(r"muur([0-9.]+)_", lens_basisnaam)
afstand_muur_origineel = float(scene_set.group(1))
LENS_SCHAAL = 5.0 
werkelijke_muur_afstand = afstand_muur_origineel


match_hx = re.search(r"_hx([0-9.-]+)", lens_basisnaam)
hoek_x = float(match_hx.group(1)) if match_hx else 0.0


match_hy = re.search(r"_hy([0-9.-]+)", lens_basisnaam)
hoek_y = float(match_hy.group(1)) if match_hy else 0.0

if not os.path.exists(LENS_FILE):
    raise FileNotFoundError(f"Could not find the lens mesh at: {LENS_FILE}")

inv_rot_x = -hoek_x
inv_rot_y = -hoek_y 

pivot = [0.0, -werkelijke_muur_afstand, 0.0]

orbit_transform = mi.ScalarTransform4f() \
    .translate(pivot) \
    .rotate(axis=[1, 0, 0], angle=inv_rot_x) \
    .rotate(axis=[0, 0, 1], angle=inv_rot_y) \
    .translate([0.0, werkelijke_muur_afstand, 0.0])


lens_base = mi.ScalarTransform4f().scale((LENS_SCHAAL, LENS_SCHAAL, LENS_SCHAAL)).rotate(axis=[1, 0, 0], angle=90)
lens_to_world = orbit_transform @ lens_base

light_rotation = mi.ScalarTransform4f().rotate(axis=[1, 0, 0], angle=inv_rot_x).rotate(axis=[0, 0, 1], angle=inv_rot_y)
nieuwe_licht_richting = light_rotation @ [0.0, -1.0, 0.0]



camera_base_origin = [0.0, -0.3, 0.0] #plaats sensor moet hier aangepast worden
nieuwe_camera_origin = camera_base_origin

sensor_to_world = mi.ScalarTransform4f().look_at(
    target=pivot,                   
    origin=list(nieuwe_camera_origin), 
    up=[0, 0, 1]                  
)




print(f"muur locatie : {pivot}")


print(f"camera locatie       : {nieuwe_camera_origin}")
print(f"camera kijkt naar    : {pivot}")

lens_pos = lens_to_world @ [0.0, 0.0, 0.0]
lens_pos_list = [lens_pos.x, lens_pos.y, lens_pos.z]
print(f"[-] lens locatie         : [ {lens_pos_list[0]:.4f},  {lens_pos_list[1]:.4f},  {lens_pos_list[2]:.4f} ]")


licht_richting_list = [nieuwe_licht_richting.x, nieuwe_licht_richting.y, nieuwe_licht_richting.z]
print(f"richting licht : [ {licht_richting_list[0]:.4f},  {licht_richting_list[1]:.4f},  {licht_richting_list[2]:.4f} ]")

scene_dict = {
    'type': 'scene',
    'integrator': {
        'type': 'ptracer',
        'max_depth': 4,
        'hide_emitters': False,
    },
    'sensor': {
        'type': 'perspective',
        'near_clip': 0.01,
        'far_clip': 1000,
        'fov': 120, #breedte van de sensor in graden
        'to_world': sensor_to_world,
        'sampler': {
            'type': 'independent',
            'sample_count': SAMPLES_PER_PIXEL
        },
        'film': {
            'type': 'hdrfilm',
            'width': RENDER_RESOLUTION[0],
            'height': RENDER_RESOLUTION[1],
            'pixel_format': 'rgb',
            'rfilter': {'type': 'gaussian'}
        },
    },
    'simple-glass': {
        'type': 'dielectric',
        'id': 'simple-glass-bsdf',
        'ext_ior': 'air',
        'int_ior': 1.49,
        'specular_reflectance': { 'type': 'spectrum', 'value': 0 },
    },
    'white-bsdf': {
        'type': 'diffuse',
        'id': 'white-bsdf',
        'reflectance': { 'type': 'rgb', 'value': (1, 1, 1) },
    },
    
    
    'receiving-plane': {
        'type': 'rectangle',
        'to_world': mi.ScalarTransform4f().look_at(
            target=[0, 1, 0],
            origin=pivot,
            up=[0, 0, 1]
        ).scale((5, 5, 5)),
        'bsdf': {'type': 'ref', 'id': 'white-bsdf'},
    },
    
 
    'lens': {
        'type': 'ply',
        'filename': LENS_FILE,
        'to_world': lens_to_world,
        'face_normals': True,
        'bsdf': {'type': 'ref', 'id': 'simple-glass'},
    },
    
    # LICHT: 
    'focused-emitter': {
        'type': 'directional',
        'direction': list(nieuwe_licht_richting),
        'irradiance': {
            'type': 'spectrum',
            'value': 0.5 
        }
    }
}

scene = mi.load_dict(scene_dict)

print(f"Afbeelding aan het renderen op {RENDER_RESOLUTION[0]}x{RENDER_RESOLUTION[1]} met {SAMPLES_PER_PIXEL} spp")
image = mi.render(scene)

lens_basisnaam = os.path.splitext(os.path.basename(LENS_FILE))[0]

bestandsnaam = f"render_{lens_basisnaam}_SPP{SAMPLES_PER_PIXEL}_RenderRes{RENDER_RESOLUTION[0]}x{RENDER_RESOLUTION[1]}.png"

render_map = r"C:....."
os.makedirs(render_map, exist_ok=True)


volledig_output_pad = os.path.join(render_map, bestandsnaam)

mi.util.write_bitmap(volledig_output_pad, image)


plt.figure(figsize=(8, 8))
plt.imshow(mi.util.convert_to_bitmap(image))
plt.axis('off')
plt.title(f"Caustic Projection\n({bestandsnaam})")
plt.tight_layout()
plt.show()