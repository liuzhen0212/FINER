import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modules
import torch
import numpy as np
from tqdm import tqdm
from inside_mesh.metrics import compute_trimesh_chamfer
from inside_mesh.metrics import compute_iou
from inside_mesh.metrics import check_mesh_contains
import mcubes
import trimesh
from glob import glob
from joblib import Parallel, delayed
from tabulate import tabulate
from pykdtree.kdtree import KDTree
import mrcfile


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout



def export_model(ckpt_path, model_name, N=512, model_type='bacon', hidden_layers=8,
                 hidden_size=256, w0=30, pe=8, s0 = 40, fbs=None,
                 scaling=None, return_sdf=False):


    if os.path.exists(f"outputs/shapes/{model_name}_{1}.obj"):
        print(f'[Exist]: outputs/shapes/{model_name}_{1}.obj')
        return 

    with HiddenPrints():
        if model_type == 'siren':
            num_outputs = 1
            model = modules.CoordinateNet(in_features=3, hidden_features=hidden_size,
                                            out_features=1, nl='sine',
                                            num_hidden_layers=hidden_layers,
                                            w0=w0,
                                            is_sdf=True)
        elif model_type == 'ff':
            num_outputs = 1
            model = modules.CoordinateNet(nl='relu',
                                            in_features=3,
                                            out_features=1,
                                            num_hidden_layers=hidden_layers,
                                            hidden_features=hidden_size,
                                            is_sdf=True,
                                            pe_scale=pe,
                                            use_sigmoid=False)
        elif model_type == 'wire':
            num_outputs = 1
            model = modules.Wire(
                in_features=3,
                out_features=1,
                hidden_layers=hidden_layers,
                hidden_features=hidden_size,
                first_omega_0=w0,
                hidden_omega_0=w0,
                scale=s0,
                is_sdf=True,
            )   

        elif model_type == 'nerfpe':
            num_outputs = 1
            model = modules.CoordinateNet(nl='relu',
                                            in_features=3,
                                            out_features=1,
                                            num_hidden_layers=hidden_layers,
                                            hidden_features=hidden_size,
                                            is_sdf=True,
                                            pe_scale=pe, nerfpe=True,
                                            use_sigmoid=False)
        elif model_type == 'finer':
            num_outputs = 1
            model = modules.FinerB(
                                in_features=3,
                                out_features=1,
                                hidden_layers=hidden_layers,
                                hidden_features=hidden_size,
                                first_omega_0=w0,
                                hidden_omega_0=w0,
                                first_bias_scale=fbs,
                                is_sdf=True,
                                )
        elif model_type == 'gauss':
            num_outputs = 1
            model = modules.Gauss(in_features=3, out_features=1, 
                              hidden_layers=hidden_layers,
                              hidden_features=hidden_size,
                              scale=30,
                              is_sdf=True
                              )

    print(model)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    model.cuda()

    # write output
    x = torch.linspace(-0.5, 0.5, N)
    if return_sdf:
        x = torch.arange(-N//2, N//2) / N
        x = x.float()
    # x = 2*(torch.arange(N) / N - 0.5)
    x, y, z = torch.meshgrid(x, x, x)
    render_coords = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=-1).cuda()
    sdf_values = [np.zeros((N**3, 1)) for i in range(num_outputs)]

    # render in a batched fashion to save memory
    bsize = int(128**2)
    for i in tqdm(range(int(N**3 / bsize))):
        coords = render_coords[i*bsize:(i+1)*bsize, :]
        out = model({'coords': coords})['model_out']

        if not isinstance(out, list):
            out = [out,]

        for idx, sdf in enumerate(out):
            sdf_values[idx][i*bsize:(i+1)*bsize] = sdf.detach().cpu().numpy()

    if return_sdf:
        return [sdf.reshape(N, N, N) for sdf in sdf_values]

    for idx, sdf in enumerate(sdf_values):
        
        # sdf = np.ones((N, N, N))
        # sdf[:, 0, 0] = 0
        # sdf[0, :, 0] = 0
        # sdf[0, 0, :] = 0
        # sdf[-1, -1, :] = 0
        # sdf[-1, :, -1] = 0
        # sdf[:, -1, -1] = 0

        sdf = sdf.reshape(N, N, N)
        vertices, triangles = mcubes.marching_cubes(-sdf, 0)
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        mesh.vertices = (mesh.vertices / N - 0.5) + 0.5/N

        if 'sphere' in model_name:
            mesh.vertices = mesh.vertices * 0.25 / np.mean(np.linalg.norm(mesh.vertices, axis=-1))

        os.makedirs('outputs/shapes', exist_ok=True)
        mesh.export(f"outputs/shapes/{model_name}_{idx+1}.obj")


def extract_gt(mesh_path, out_name, N=512):
    mesh = trimesh.load(mesh_path)
    mesh.vertices, _, _ = normalize(mesh.vertices)
    x = torch.linspace(-1, 1, N)
    x, y, z = torch.meshgrid(x, x, x)
    coords = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=-1)
    out = check_mesh_contains(mesh, coords.numpy())
    out = out.reshape(N, N, N)

    print(np.any(out))
    print(np.min(out), np.max(out))

    vertices, triangles = mcubes.marching_cubes(out, 0.5)
    # vertices, triangles = mcubes.marching_cubes(out, 0)
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    mesh.vertices = 2 * (mesh.vertices / N - 0.5)
    mesh.export(f"outputs/shapes/ref_{out_name}.obj")


def normalize(coords, scaling=0.9):
    coords = np.array(coords).copy()
    cmean = np.mean(coords, axis=0, keepdims=True)
    coords -= cmean
    coord_max = np.amax(coords)
    coord_min = np.amin(coords)
    coords = (coords - coord_min) / (coord_max - coord_min)
    coords -= 0.5
    coords *= scaling

    scale = scaling / (coord_max - coord_min)
    offset = -scaling * (cmean + coord_min) / (coord_max - coord_min) - 0.5*scaling
    return coords, scale, offset


def export_reference_mesh(scenes, reference_dir='../../data'):
    # we need to realign the meshes to ground truth...
    # we usually normalize the points during training, so we have to undo
    # the normalization for the ground truth mesh to
    # match the training output

    for scene in scenes:
        # this is the xyz file used to train siren/bacon
        # we need to calculate the resizing scale and apply it to the 
        # corresponding .obj file
        # gt_mesh = trimesh.load(f'./outputs/shapes/gt_{scene}.xyz')
        gt_mesh = trimesh.load(os.path.join(reference_dir, f'gt_{scene}.xyz'))
        gt_mesh.vertices, scale, offset = normalize(gt_mesh.vertices)

        # rescale the .obj file
        # gt_mesh = trimesh.load(f'./outputs/shapes/gt_{scene}.obj')
        gt_mesh = trimesh.load(f'./outputs/shapes/gt_{scene}.ply')
        gt_mesh.vertices = gt_mesh.vertices * scale + offset
        gt_mesh.export(f'./outputs/shapes/ref_{scene}.obj')


# compute chamfer distances
def run_metrics(name):
    if 'gt' in name or 'ref' in name:
        return

    mesh_pred = trimesh.load(f'./outputs/shapes/{name}.obj')

    if 'armadillo' in name:
        gt_name = 'armadillo'
    elif 'lucy' in name:
        gt_name = 'lucy'
    elif 'thai' in name:
        gt_name = 'thai'
    elif 'sphere' in name:
        gt_name = 'sphere'
    elif 'dragon' in name:
        gt_name = 'dragon'
    elif 'statue' in name:
        gt_name = 'statue'
    elif 'BeardedMan' in name:
        gt_name = 'BeardedMan'
    else:
        raise ValueError("don't have ground truth mesh")

    mesh_gt = trimesh.load(f'./outputs/shapes/ref_{gt_name}.obj')
    chamfer = compute_trimesh_chamfer(mesh_pred, mesh_gt, sphere=gt_name == 'sphere')
    iou = compute_iou(mesh_gt, mesh_pred, sphere=gt_name == 'sphere')

    # mesh_gt.export('test_gt.obj')
    # mesh_pred.export('test_pred.obj')

    metrics = {'chamfer': chamfer, 'iou': iou}
    print(f'{name}: chamfer: {chamfer}, iou: {iou}')
    np.save(f'./outputs/shapes/{name}.npy', metrics)
    
    return 

    # return chamfer, iou


def calc_all_metrics(scenes=['armadillo', 'dragon', 'lucy', 'thai']):
    # scenes = ['sphere', 'thai', 'lucy', 'armadillo', 'dragon']

    for scene in tqdm(scenes):
        tqdm.write(scene)
        filenames = glob(f'./outputs/shapes/*{scene}*.obj')
        filenames = [os.path.splitext(os.path.basename(name))[0] for name in filenames]

        print(filenames)
        Parallel(n_jobs=5)(delayed(run_metrics)(name) for name in filenames)


def aggregate_metrics(
    scenes = ['dragon', 'armadillo', 'thai', 'lucy'], 
    methods = ['finer', 'siren', 'wire', 'nerfpeL10', 'gauss']
):

    chamfer = np.zeros((len(methods), len(scenes)))
    iou = np.zeros((len(methods), len(scenes)))

    for sidx, scene in enumerate(scenes):
        for midx, method in enumerate(methods):
            if method == 'finer':
                path = f'./outputs/shapes/{scene}_{method}_2x256_w30_fbs_1.0_1.npy'
            elif method == 'wire':
                path = f'./outputs/shapes/{scene}_{method}_2x256_w20s10_1.npy'
            elif method == 'nerfpeL10':
                path = f'./outputs/shapes/{scene}_{"nerfpe"}_2x256_L10_1.npy'
            elif method == 'gauss':
                path = f'./outputs/shapes/{scene}_{"gauss"}_2x256_s30_1.npy'
            else:
                path = f'./outputs/shapes/{scene}_{method}_2x256_1.npy'
            chamfer[midx, sidx] = np.load(path, allow_pickle=True)[()]['chamfer']
            iou[midx, sidx]     = np.load(path, allow_pickle=True)[()]['iou']
            
    metrics = np.zeros((len(methods), 2*len(scenes)))
    metrics[:, ::2] = chamfer
    metrics[:, 1::2] = iou
    # scenes = 2*scenes
    headers = []
    headers.append('Methods')
    for idx, scene in enumerate(scenes):
        headers.append(f"{scene}(chamfer)")
        headers.append(f"(iou)")
    
    print(metrics.shape)
    content = []
    for i in range(metrics.shape[0]):
        line = []
        line.append(methods[i])
        for k in range(metrics.shape[1]):
            line.append(metrics[i, k]) 
        line.append(np.mean(chamfer[i]))
        line.append(np.mean(iou[i]))
        content.append(line)
        
    metrics = np.concatenate((metrics, np.mean(chamfer, axis=-1)[:, None], np.mean(iou, axis=-1)[:, None]), axis=-1)
    with open(f'./outputs/shapes/metrics.txt', 'w') as f:
        # out_str = tabulate(metrics, tablefmt='latex', floatfmt='.3e', headers=headers + ['avg. chamfer', 'avg. iou'])
        out_str = tabulate(content, tablefmt='latex', floatfmt='.3e', headers=headers + ['avg. chamfer', 'avg. iou'])
        f.write(out_str)


def export_meshes(scenes=['armadillo', 'dragon', 'lucy', 'thai']):
    # export meshes siren & finer
    
    results = []
    for scene in scenes:

        ## finer
        results.append({
            'type': 'finer',
            'ckpt': f'../logs/{scene}_finer_2x256_w30_fbs_{1.0}/checkpoints/model_final.pth',
            'name': f'{scene}_finer_2x256_w30_fbs_{1.0}',
            'hidden_layers': 2,
            'hidden_size': 256,
            'fbs': 1.0,
            'w0': 30,
        })
        ## siren
        results.append({
            'type': 'siren',
            'ckpt': f'../logs/{scene}_siren_2x256/checkpoints/model_final.pth',
            'name': f'{scene}_siren_2x256',
            'hidden_layers': 2,
            'hidden_size': 256,
            'w0': 30,
        })
        ## gauss
        results.append({
            'type': 'gauss',
            'ckpt': f'../logs/{scene}_gauss_2x256_s30/checkpoints/model_final.pth',
            'name': f'{scene}_gauss_2x256_s30',
            'hidden_layers': 2,
            'hidden_size': 256,
            'w0': 30,
        })     
        results.append({
            'type': 'wire',
            'ckpt': f'../logs/{scene}_wire_2x256_w20s10/checkpoints/model_final.pth',
            'name': f'{scene}_wire_2x256_w20s10',
            'hidden_layers': 2,
            'hidden_size': 256,
            'w0': 20,
            's0': 10,
        })  
        results.append({
            'type': 'nerfpe',
            'ckpt': f'../logs/{scene}_nerfpe_2x256_L10/checkpoints/model_final.pth',
            'name': f'{scene}_nerfpe_2x256_L10', 
            'hidden_layers': 2,
            'hidden_size': 256,
        })
    
    for res in results:
        if not os.path.exists(res['ckpt']):
            continue
        print(res['ckpt'])
        
        if res['type'] == 'nerfpe':
            print('Exporting nerfpe')
            export_model(res['ckpt'], res['name'], model_type='nerfpe',
                         hidden_layers=res['hidden_layers'], hidden_size=res['hidden_size'])
        if res['type'] == 'siren':
            print('Exporting SIREN')
            export_model(res['ckpt'], res['name'], model_type='siren', w0=res['w0'],
                         hidden_layers=res['hidden_layers'], hidden_size=res['hidden_size'])
        if res['type'] == 'gauss':
            print('Exporting SIREN')
            export_model(res['ckpt'], res['name'], model_type='gauss', w0=res['w0'],
                         hidden_layers=res['hidden_layers'], hidden_size=res['hidden_size'])
        if res['type'] == 'wire':
            print('Exporting wire')
            export_model(res['ckpt'], res['name'], model_type='wire', w0=res['w0'], s0=res['s0'],
                         hidden_layers=res['hidden_layers'], hidden_size=res['hidden_size'])
        if res['type'] == 'finer':
            print('Exporting Finer')
            export_model(res['ckpt'], res['name'], model_type='finer', w0=res['w0'], fbs=res['fbs'],
                         hidden_layers=res['hidden_layers'], hidden_size=res['hidden_size'])
    



if __name__ == '__main__':
    print('start')
    
    export_reference_mesh(['armadillo', 'dragon', 'lucy', 'thai']) 
    export_meshes(['armadillo', 'dragon', 'lucy', 'thai']) 
    calc_all_metrics()
    aggregate_metrics(scenes = ['armadillo', 'dragon', 'lucy', 'thai'])
    
    # scenes = ['armadillo']
    # export_reference_mesh(scenes)
    # export_meshes(scenes) 
    # calc_all_metrics(scenes)
    # aggregate_metrics(scenes)
    
    

    
    