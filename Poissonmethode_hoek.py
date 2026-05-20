#Deze code maakt de lens op basis van de Poisson-methode en werkt ook onder andere hoeken
import numpy as np
from PIL import Image
from stl import mesh
import os
import trimesh
from PIL import ImageOps


pad_in = r"C:....."
map_uit = r"C:...."

if not os.path.exists(pad_in):
    print(f"afbeelding niet gevonden op: {pad_in}")
else:
    afstand_muur = 0.5  #projectie afstand in meter
    hoek_x = 0          #verticale rotatie rond x-as(pitch)    
    hoek_y = 0          #horizontale rotatie rond y-as(yaw)
    n1 = 1.53           #brekingindex lens  
    n2 = 1.0            #brekingindex lucht  
    max_verschuiving = 0.03 #hoger is meer contrast maar ook meer vervorming
    
    lens_breedte = 0.12 
    lens_hoogte = 0.081875 
    
    resolutie_x = 750
    resolutie_y = int(resolutie_x * (lens_hoogte / lens_breedte))
    D_basis = 0.01
    pixel_grootte = lens_breedte / resolutie_x
    
    iteraties = 20
    bestandsnaam = f"lens_muur{afstand_muur}_hx{hoek_x}_hy{hoek_y}_versch{max_verschuiving}_resx{resolutie_x}resy{resolutie_y}_iter{iteraties}.ply"
    pad_uit_ply = os.path.join(map_uit, bestandsnaam)

    img = Image.open(pad_in).convert('L')
    img = img.resize((resolutie_x, resolutie_y)) 
    img_array = np.array(img, dtype=np.float64)
    
    brightness = img_array / (np.sum(img_array) or 1.0)
    h_grid, w_grid = brightness.shape
    loss = (np.full((h_grid, w_grid), 1.0 / (h_grid * w_grid))) - brightness
    loss = loss - np.mean(loss)

    loss_fft = np.fft.fft2(loss)
    kx = np.fft.fftfreq(w_grid, d=pixel_grootte) * 2 * np.pi
    ky = np.fft.fftfreq(h_grid, d=pixel_grootte) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2
    K2[0, 0] = 1
    
    Phi = np.real(np.fft.ifft2(-loss_fft / K2))
    dy_phi, dx_phi = np.gradient(Phi, pixel_grootte)
    
    schaal = max_verschuiving / (np.max(np.sqrt(dx_phi**2 + dy_phi**2)) or 1)
    print(schaal)
    Dx, Dy = dx_phi * schaal, dy_phi * schaal

    x_as = np.linspace(-lens_breedte/2, lens_breedte/2, w_grid)
    y_as = np.linspace(-lens_hoogte/2, lens_hoogte/2, h_grid)
    X, Y = np.meshgrid(x_as, y_as)
    Y = -Y
    
    Z_muur = afstand_muur + X * np.tan(np.radians(hoek_y)) + Y * np.tan(np.radians(hoek_x))

    h_xy = np.zeros((h_grid, w_grid))

    for it in range(iteraties):
        d_effectief = Z_muur - h_xy

        L = np.sqrt(Dx**2 + Dy**2 + d_effectief**2)
        
        noemer = (n1 * L) - (n2 * d_effectief)

        noemer = np.where(np.abs(noemer) < 1e-12, np.sign(noemer) * 1e-12, noemer)

        Sx = -(n2 * Dx) / noemer
        Sy = -(n2 * Dy) / noemer

        div = np.gradient(Sx, pixel_grootte, axis=1) + np.gradient(Sy, pixel_grootte, axis=0)
        div_fft = np.fft.fft2(div)
        h_nieuw = np.real(np.fft.ifft2(-div_fft / K2))

        h_nieuw = h_nieuw - np.min(h_nieuw)
        verschil = np.max(np.abs(h_nieuw - h_xy))

        h_xy = h_nieuw

    print('stl aan het maken')
    Z_lens = D_basis + h_xy
    
    if np.min(Z_lens) <= 0:
        D_basis = np.max(h_xy) + 2.0 
        Z_lens = D_basis - h_xy
    faces = []
    
    for i in range(h_grid - 1):
        for j in range(w_grid - 1):
            v1 = [X[i, j], Y[i, j], Z_lens[i, j]]
            v2 = [X[i+1, j], Y[i+1, j], Z_lens[i+1, j]]
            v3 = [X[i, j+1], Y[i, j+1], Z_lens[i, j+1]]
            v4 = [X[i+1, j+1], Y[i+1, j+1], Z_lens[i+1, j+1]]
            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])
    for i in range(h_grid - 1):
        for j in range(w_grid - 1):
            v1_b = [X[i, j], Y[i, j], 0]
            v2_b = [X[i+1, j], Y[i+1, j], 0]
            v3_b = [X[i, j+1], Y[i, j+1], 0]
            v4_b = [X[i+1, j+1], Y[i+1, j+1], 0]
            faces.append([v1_b, v3_b, v2_b])
            faces.append([v2_b, v3_b, v4_b])
    for j in range(w_grid - 1):
        faces.extend([
            [[X[0,j], Y[0,j], Z_lens[0,j]], [X[0,j], Y[0,j], 0], [X[0,j+1], Y[0,j+1], Z_lens[0,j+1]]],
            [[X[0,j], Y[0,j], 0], [X[0,j+1], Y[0,j+1], 0], [X[0,j+1], Y[0,j+1], Z_lens[0,j+1]]]
        ])
    for j in range(w_grid - 1):
        faces.extend([
            [[X[-1,j+1], Y[-1,j+1], Z_lens[-1,j+1]], [X[-1,j+1], Y[-1,j+1], 0], [X[-1,j], Y[-1,j], Z_lens[-1,j]]],
            [[X[-1,j+1], Y[-1,j+1], 0], [X[-1,j], Y[-1,j], 0], [X[-1,j], Y[-1,j], Z_lens[-1,j]]]
        ])
    for i in range(h_grid - 1):
        faces.extend([
            [[X[i+1,0], Y[i+1,0], Z_lens[i+1,0]], [X[i+1,0], Y[i+1,0], 0], [X[i,0], Y[i,0], Z_lens[i,0]]],
            [[X[i+1,0], Y[i+1,0], 0], [X[i,0], Y[i,0], 0], [X[i,0], Y[i,0], Z_lens[i,0]]]
        ])
    for i in range(h_grid - 1):
        faces.extend([
            [[X[i,-1], Y[i,-1], Z_lens[i,-1]], [X[i,-1], Y[i,-1], 0], [X[i+1,-1], Y[i+1,-1], Z_lens[i+1,-1]]],
            [[X[i,-1], Y[i,-1], 0], [X[i+1,-1], Y[i+1,-1], 0], [X[i+1,-1], Y[i+1,-1], Z_lens[i+1,-1]]]
        ])

    faces = np.array(faces)
    lens_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            lens_mesh.vectors[i][j] = f[j]

    faces_array = np.array(faces)
    vertices_soup = faces_array.reshape(-1, 3).astype(np.float32) # Mitsuba verwacht meestal float32
    

    unique_vertices, inverse_indices = np.unique(vertices_soup, axis=0, return_inverse=True)
    
    num_vertices = len(unique_vertices)
    num_faces = len(faces_array)
    face_data_indices = inverse_indices.reshape(-1, 3).astype(np.int32)

    with open(pad_uit_ply, 'wb') as f:
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_vertices}
property float x
property float y
property float z
element face {num_faces}
property list uchar int vertex_index
end_header
"""
        f.write(header.encode('utf-8'))
        
        # Schrijf de vertices (binair)
        unique_vertices.tofile(f)
        
        
        threes = np.full((num_faces, 1), 3, dtype=np.uint8) 
        for i in range(num_faces):
            f.write(threes[i].tobytes())
            f.write(face_data_indices[i].tobytes())

    print(f"ply opgeslagen als: {pad_uit_ply}")