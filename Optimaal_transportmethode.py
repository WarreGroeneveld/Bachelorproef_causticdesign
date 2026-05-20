#Deze code maakt de lens op basis van de optimaaltransportmethode
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized
import scipy.sparse.linalg as spla
from scipy.optimize import minimize_scalar
from stl import mesh
from PIL import Image, ImageOps
import os
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

def load_and_prepare_image(image_path, max_dim=512):
    img = Image.open(image_path).convert('L')
    
    w, h = img.size
    if w > h:
        new_w = max_dim
        new_h = int(max_dim * (h / w))
    else:
        new_h = max_dim
        new_w = int(max_dim * (w / h))
    
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    
    u_photo = np.array(img, dtype=np.float64) / 255.0

    u_photo = (u_photo - np.min(u_photo)) / (np.max(u_photo) - np.min(u_photo))

    u_photo = gaussian_filter(u_photo, sigma=1.0)

    u_photo = np.clip(u_photo, a_min=0.0005, a_max=1.0)

    u = u_photo ** 1.8 #contrastverhoging

    u = u / np.mean(u)
    
    return u

def build_laplacian_matrix(rows, cols, h):
    N = rows * cols
    L = sp.lil_matrix((N, N))
    
    
    factor = 0.5 /(h**2)
    
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            L[idx, idx] -= 4 * factor
            
            neighbors = [
                (i - 1, j - 1),
                (i + 1, j - 1),
                (i + 1, j + 1),
                (i - 1, j + 1)
            ]
            
            for ni, nj in neighbors:
                ni_clamped = max(0, min(rows - 1, ni))
                nj_clamped = max(0, min(cols - 1, nj))
                n_idx = ni_clamped * cols + nj_clamped
                L[idx, n_idx] += factor
                
    return L.tocsc()

def compute_hessian_determinant(psi_grid, h):
    rows, cols = psi_grid.shape
    psi_padded = np.pad(psi_grid, pad_width=1, mode='edge')

    top_left     = psi_padded[0:-1, 0:-1]
    top_right    = psi_padded[0:-1, 1:  ]
    bottom_left  = psi_padded[1:  , 0:-1]
    bottom_right = psi_padded[1:  , 1:  ]

    gx = (top_right + bottom_right - top_left - bottom_left) / (2 * h)
    gy = (bottom_left + bottom_right - top_left - top_right) / (2 * h)

    gx_TL = gx[:-1, :-1];  gy_TL = gy[:-1, :-1]
    gx_TR = gx[:-1, 1: ];  gy_TR = gy[:-1, 1: ]
    gx_BR = gx[1:,  1: ];  gy_BR = gy[1:,  1: ]
    gx_BL = gx[1:,  :-1];  gy_BL = gy[1:,  :-1]

    def cross_product(x1, y1, x2, y2):
        return x1 * y2 - y1 * x2

    area = 0.5 * (
        cross_product(gx_TL, gy_TL, gx_TR, gy_TR) +
        cross_product(gx_TR, gy_TR, gx_BR, gy_BR) +
        cross_product(gx_BR, gy_BR, gx_BL, gy_BL) +
        cross_product(gx_BL, gy_BL, gx_TL, gy_TL)
    )
    return area 


def compute_residual(psi_flat, u_flat, L, rows, cols, h):

    psi_grid = psi_flat.reshape((rows, cols))
    q_grid = compute_hessian_determinant(psi_grid, h)
    q_flat = q_grid.flatten()
    return L.dot(psi_flat) + (q_flat / (h**2)) + (1.0 - u_flat)

def solve_optimal_transport(u, h, max_iter=100, tol=1e-5):

    rows, cols = u.shape
    N = rows * cols
    u_flat = u.flatten()
    

    print("Laplaciaan aan het bouwen")
    L = build_laplacian_matrix(rows, cols, h)
    

    L_reg = L - 1e-7 * sp.eye(N) #om singulariteit te voorkomen
    

    solve_L = spla.factorized(-L_reg)

    psi = np.zeros(N)
    r = compute_residual(psi, u_flat, L, rows, cols, h)
    
    d_prev = np.zeros(N)
    r_prev = np.zeros(N)
    alpha_prev = 1.0
    epsilon = 1e-3  
    
    for k in range(max_iter):
        residue_norm = (np.sum(r**2))
        print(f"Iteratie {k}, Residue Norm: {residue_norm:.6f}")
        
        if residue_norm < tol:
            break

        d_hat = solve_L(r)
    
        if k == 0:
            beta = 0.0
            d = d_hat
        else:
            max_d = np.max(np.abs(d_hat))
            h_eps = 1e-7 / max_d if max_d > 1e-15 else 1e-7
            r_eps = compute_residual(psi + h_eps * d_hat, u_flat, L, rows, cols, h)
            dr = r - r_prev
            J_d_hat = (r_eps - r) / h_eps
            
            teller = alpha_prev * np.dot(dr, J_d_hat)
            
            noemer = np.dot(dr, dr)

            beta = -teller / noemer if noemer != 0 else 0.0

            d = d_hat + beta * d_prev
            
        def objective(alpha_test): #Line search
            r_temp = compute_residual(psi + alpha_test * d, u_flat, L, rows, cols, h)
            return np.sum(r_temp**2)
            
        res = minimize_scalar(objective, bounds=(0.0, 2.0), method='bounded')
        alpha = res.x
        
        psi = psi + alpha * d
        
        psi = psi - np.mean(psi)
        
        r_prev = r.copy()
        r = compute_residual(psi, u_flat, L, rows, cols, h)
        d_prev = d.copy()
        alpha_prev = alpha
    return psi.reshape((rows, cols))

def calculate_heightmap(psi_grid, lens_breedte_mm, Z_distance_mm, refractive_index=1.49, base_thickness=3.0):
    psi_phys = psi_grid * (lens_breedte_mm ** 2)
    height_map = -psi_phys / (Z_distance_mm * (refractive_index - 1.0))
    height_map = height_map - np.min(height_map) 

    height_map = height_map + base_thickness
    
    return height_map

def export_to_ply(height_map, pixel_size_mm, filename="caustic_lens_solid.ply"):

    rows, cols = height_map.shape

    breedte = cols * pixel_size_mm
    hoogte = rows * pixel_size_mm

    x = (np.arange(cols) * pixel_size_mm) - (breedte / 2.0)
    y = (np.arange(rows) * pixel_size_mm) - (hoogte / 2.0)
    xv, yv = np.meshgrid(x, y)
    verts_top = np.column_stack((xv.flatten(), yv.flatten(), height_map.flatten()))
    verts_bottom = np.column_stack((xv.flatten(), yv.flatten(), np.zeros_like(height_map.flatten())))
    vertices = np.vstack((verts_top, verts_bottom)).astype(np.float32)
    num_vertices = len(vertices)


    faces_list = []
    offset = rows * cols 
    
    for i in range(rows - 1):
        for j in range(cols - 1):
            tl = i * cols + j
            tr = tl + 1
            bl = (i + 1) * cols + j
            br = bl + 1
            
         
            faces_list.append([tl, bl, tr])
            faces_list.append([tr, bl, br])

            tl_b = tl + offset; tr_b = tr + offset
            bl_b = bl + offset; br_b = br + offset
            faces_list.append([tl_b, tr_b, bl_b])
            faces_list.append([tr_b, br_b, bl_b])
            
    for j in range(cols - 1): 
        faces_list.append([j, j+1, j+offset])
        faces_list.append([j+1, j+1+offset, j+offset])
        idx = (rows - 1) * cols + j
        faces_list.append([idx, idx+offset, idx+1])
        faces_list.append([idx+1, idx+offset, idx+1+offset])
        
    for i in range(rows - 1): 
        idx = i * cols
        idx_next = (i + 1) * cols
        faces_list.append([idx, idx+offset, idx_next])
        faces_list.append([idx_next, idx+offset, idx_next+offset])
        idx_r = i * cols + (cols - 1)
        idx_r_next = (i + 1) * cols + (cols - 1)
        faces_list.append([idx_r, idx_r_next, idx_r+offset])
        faces_list.append([idx_r_next, idx_r_next+offset, idx_r+offset])


    faces_array = np.array(faces_list, dtype=np.int32)
    num_faces = len(faces_array)

    ply_face_dtype = np.dtype([('n_verts', 'u1'), ('v1', 'i4'), ('v2', 'i4'), ('v3', 'i4')])
    faces_binary = np.empty(num_faces, dtype=ply_face_dtype)
    faces_binary['n_verts'] = 3  
    faces_binary['v1'] = faces_array[:, 0]
    faces_binary['v2'] = faces_array[:, 1]
    faces_binary['v3'] = faces_array[:, 2]

    with open(filename, 'wb') as f:
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_vertices}
property float x
property float y
property float z
element face {num_faces}
property list uchar int vertex_indices
end_header
"""
        f.write(header.encode('utf-8'))
        vertices.tofile(f)
        faces_binary.tofile(f)

    print(f"filename: {filename}")


if __name__ == "__main__":
    pad_naar_afbeelding = r"C:\Users\warre\Desktop\Bachelorproef\melencolia.jpg"
    map_uit = r"C:\Users\warre\Desktop\Bachelorproef\lenzen"
    
    foto_naam = os.path.splitext(os.path.basename(pad_naar_afbeelding))[0]

    maximale_resolutie = 768
    lens_breedte_m = 0.12       
    projectie_afstand_m = 0.5  
    refractive_index_val = 1.53
    basis_dikte_m = 0.01
    

    print("\nlaad afbeelding")
    u_input = load_and_prepare_image(pad_naar_afbeelding, max_dim=maximale_resolutie)

    rows, cols = u_input.shape

    bestandsnaam = f"OT_Melencolia_{foto_naam}_{cols}x{rows}_muur{projectie_afstand_m}_breedte{lens_breedte_m}.ply"
    
    volledig_pad_uit = os.path.join(map_uit, bestandsnaam)

    h_stapgrootte = 1.0 / cols
     
    print("\noptimaal transport aan het oplossen")
    psi_result = solve_optimal_transport(u_input, h_stapgrootte, max_iter=250, tol=1e-5/h_stapgrootte**2)
        
    print("\nheightmap aan het berekenen")
    hoogtekaart = calculate_heightmap(
        psi_grid=psi_result, 
        lens_breedte_mm=lens_breedte_m, 
        Z_distance_mm=projectie_afstand_m,
        refractive_index=refractive_index_val, 
        base_thickness=basis_dikte_m
    )

    print("\n ply aan het exporteren")
    pixel_grootte_mm = lens_breedte_m / cols 
        
    export_to_ply(hoogtekaart, pixel_grootte_mm, filename=volledig_pad_uit)
        
    print(f"\nlens is gegenereerd en opgeslagen als:\n{volledig_pad_uit}")
