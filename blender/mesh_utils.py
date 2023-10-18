import atexit
import subprocess
import os

def genreate_mesh_from_texture(bpy, image):
    import torch
    import cv2
    import numpy as np

    image = np.copy(image[0].numpy())
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = (gray * 255).astype(np.uint8)
    # Find contours
    contours, _ = cv2.findContours(
        gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour
    areas = [cv2.contourArea(contour) for contour in contours]

    max_area_index = areas.index(max(areas))
    largest_contour = contours[max_area_index]
    contours = [largest_contour]

    def normalize_vertices(vertices, max_value):
        return vertices / float(max_value) * 2 - 1

    # Get the image width and height
    height, width = image.shape[:2]

    # Normalize the vertices
    normalized_contours = []
    for contour in contours:
        normalized_contour = []
        for vertex in contour:
            normalized_vertex = [normalize_vertices(
                vertex[0][0], width), normalize_vertices(vertex[0][1], height) * -1]
            normalized_contour.append(normalized_vertex)
        normalized_contours.append(
            np.array(normalized_contour, dtype=np.float32))

    meshes = []
    # print(len(normalized_contours))
    for i, contour in enumerate(normalized_contours):
        # Create a new mesh for each contour
        mesh = bpy.data.meshes.new(name=f"NewMesh{i}")
        # Create a new object for each mesh
        obj = bpy.data.objects.new(f"NewObject{i}", mesh)
        # Link the object to the current collection
        bpy.context.collection.objects.link(obj)

        # Add a z coordinate to each vertex
        ordered_vertices = [(*vertex, 0) for vertex in contour]
        # Create a face from the vertices
        face = list(range(len(ordered_vertices)))

        # Create the mesh from the vertices and face
        mesh.from_pydata(ordered_vertices, [], [face])

        # Create a default shape key for the mesh
        sk_basis = obj.shape_key_add(name='Basis')

        meshes.append(obj)  # Add the object to the list of meshes

    # Draw contours on the original image
    if not image.flags['C_CONTIGUOUS']:
        image = np.ascontiguousarray(image)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

    # Convert image back to tensor
    image = torch.from_numpy(np.expand_dims(image, axis=0))
    return (image, meshes[0])


def assign_texture(bpy, BPY_OBJ, texture, texture_name):
    import numpy as np
    import time
    
    # Start the timer
    start_time = time.time()

    # Convert image to numpy
    texture = texture[0].numpy()

    # Flip the image vertically
    texture = np.flipud(texture)

    # Create an image with the required dimensions
    img = bpy.data.images.new(
        texture_name, width=texture.shape[1], height=texture.shape[0], alpha = True)

    # If there is no alpha channel, append one full of 1's
    if texture.shape[2] == 3:
        alpha_channel = np.ones((*texture.shape[:2], 1))
        texture = np.concatenate((texture, alpha_channel), axis=2)

    end_time = time.time()
    print(f"Time taken (np.concatenate) : {end_time - start_time} seconds")

    # Flatten image data and rearrange color channels for blender
    img.pixels = texture.ravel()

    end_time = time.time()
    print(f"Time taken (texture.ravel) : {end_time - start_time} seconds")

    # End the timer and print the time taken
  
    # Pack image to store it within .blend file
    img.pack()

    # Save image to a file
    # img.filepath_raw = 'test.png'
    # img.file_format = 'PNG'
    # img.save()

    # Get the active object
    obj = BPY_OBJ

    # Create a material
    mat = bpy.data.materials.new("MaterialName")
    mat.use_nodes = True
    mat.blend_method = 'BLEND'
    nodes = mat.node_tree.nodes
    for node in nodes:
        nodes.remove(node)

    # Add a new texture node
    texture_node = nodes.new(type='ShaderNodeTexImage')
    texture_node.image = img

    # Add a new BSDF node
    bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')

    # Add a new output node
    output_node = nodes.new(type='ShaderNodeOutputMaterial')

    # Link nodes together
    links = mat.node_tree.links
    links.new(bsdf_node.inputs['Base Color'],
              texture_node.outputs['Color'])
    links.new(output_node.inputs['Surface'], bsdf_node.outputs['BSDF'])

    links.new(bsdf_node.inputs['Alpha'], texture_node.outputs['Alpha'])

    # Assign the material to the active object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    end_time = time.time()
    print(f"Time taken (Full) : {end_time - start_time} seconds")


blender_process_global = []


def open_in_blender(blender_process, blender_path, output_file, camera_location=(0, 0, 0), camera_rotation=(0, 0, 0), shading="Material"):
    import global_bpy
    import mathutils
    bpy = global_bpy.get_bpy()

    # Change shading mode and viewport
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = shading.upper()
                    rv3d = space.region_3d
                    rv3d.view_location = camera_location
                    rv3d.view_rotation = mathutils.Euler(
                        camera_rotation).to_quaternion()

    # Open blender
    if blender_process != None:
        blender_process.kill()
        # remove from global list so it doesn't get garbage collected
        blender_process_global.remove(blender_process)

    # Save as .blend
    if os.path.exists(output_file):
        os.remove(output_file)
    bpy.ops.wm.save_as_mainfile(filepath=output_file)

    print('blender_path', blender_path)
    print('output_file', output_file)
    blender_process = subprocess.Popen([blender_path, output_file])
    # append to global list so it doesn't get garbage collected
    blender_process_global.append(blender_process)

    return blender_process


# detects when the python process is killed, and kills the blender process

@atexit.register
def kill_blender_process():
    print('blender_process_global', blender_process_global)
    for process in blender_process_global:
        process.kill()


def export_gltf(output_dir, bpy_objects, filename, model_type, write_mode, metadata):
    import global_bpy
    bpy = global_bpy.get_bpy()
    # print(bpy, bpy_objects)
    
    # deselect all objects
    override = bpy.context.copy()
    override["selected_objects"] = list(bpy_objects)
    override["active_object"] = list(bpy_objects)[0]

    bpy_objects[0]["metadata"] = metadata
    for obj in bpy_objects:
        obj.select_set(True)

    def get_file_extension(model_type):
        if model_type == "GLB":
            return ".glb"
        elif model_type == "GLTF_EMBEDDED":
            return ".gltf"
        elif model_type == "GLTF_SEPARATE":
            return ".gltf"
        elif model_type == "AVA":
            return ".ava"

    ext = get_file_extension(model_type)
    filepath = output_dir + "/" + filename + ext + (".glb" if model_type == "AVA" else "")

    if write_mode == "Increment":
        count = 0
        # while file exists, increment count
        while os.path.exists(output_dir + "/" + filename + '_' + str(count) + ext):
            count += 1

        filepath = output_dir + "/" + filename + '_' + str(count) + ext + (".glb" if model_type == "AVA" else "")
    
    with bpy.context.temp_override(**override):
        bpy.ops.export_scene.gltf(filepath=filepath, export_format="GLB" if model_type == "AVA" else model_type, use_selection=True, export_extras=True)
        # print(filepath)
        if filepath.endswith('.ava.glb'):
            new_filepath = filepath.replace('.ava.glb', '.ava')
            os.replace(filepath, new_filepath)
            filepath = new_filepath

    return filepath