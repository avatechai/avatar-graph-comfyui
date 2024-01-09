import blender_node
import math
import folder_paths
import torch
import numpy as np
import os
from PIL import Image, ImageOps

def get_incremented_filename(folder_path, base_filename):
    # Initialize the counter and create the full initial path
    counter = 0
    output_path = f"{folder_path}/{base_filename}.png"

    # Check if the file exists and increment the counter until the file does not exist
    while os.path.exists(output_path):
        counter += 1
        output_path = f"{folder_path}/{base_filename}_{counter}.png"

    return output_path

class BlenderRenderImage(blender_node.ObjectOps):
    def __init__(self):
        pass

    EXTRA_INPUT_TYPES = {
    }

    # OUTPUT_NODE = True
    RETURN_TYPES = ("BPY_OBJ", "IMAGE")

    def add_light(self, bpy):
        # Check if there is at least one light source in the scene
        light_exists = any(ob for ob in bpy.data.objects if ob.type == 'LIGHT')
        if not light_exists:
            # Create a new Area light datablock for ambient light
            light_data = bpy.data.lights.new(name='AmbientLight', type='AREA')
            light_object = bpy.data.objects.new(name='AmbientLight', object_data=light_data)
            bpy.context.collection.objects.link(light_object)
            # Position the light in the scene
            light_object.location = (0, 0, 10)
            # Set light size for soft shadows and ambient effect
            light_data.size = 10
            light_data.energy = 1000
            print("Added an ambient light source to the scene.")
    
    def add_camera(self, bpy):
        # Check if there is a camera in the scene
        if bpy.context.scene.camera:
            return bpy.context.scene.camera 

        # If not, create a new camera
        cam_data = bpy.data.cameras.new(name='Camera')
        cam = bpy.data.objects.new(name='Camera', object_data=cam_data)
        bpy.context.collection.objects.link(cam)
        # Set the new camera to the active camera
        bpy.context.scene.camera = cam
        # Position the camera to a default view
        cam.location = (0, 0, 10)
        return cam
    
    def get_texture_size(self, obj):
        # Get the first material slot
        mat = obj.data.materials[0]
        
        # Check if the material has a node tree
        if mat.node_tree:
            nodes = mat.node_tree.nodes
            # Find an image texture node in the node tree
            for node in nodes:
                if node.type == 'TEX_IMAGE':
                    texture = node.image
                    if texture:
                        return texture.size
            print("No image texture node found in the material's node tree.")
        else:
            print("Material has no node tree.")


    def blender_process(self, bpy, BPY_OBJ=None):
        cam = self.add_camera(bpy)

        plane = BPY_OBJ
        if plane:
            tex_width, tex_height = self.get_texture_size(plane)
            # Calculate the aspect ratio of the plane
            aspect_ratio_plane = tex_width / tex_height

            # Set the render resolution to match the plane's aspect ratio
            # Choose an arbitrary resolution for the longer side of the plane
            base_resolution = 512

            if aspect_ratio_plane > 1:
                # Plane is wider than it is tall
                bpy.context.scene.render.resolution_x = base_resolution
                bpy.context.scene.render.resolution_y = int(base_resolution / aspect_ratio_plane)
            else:
                # Plane is taller than it is wide
                bpy.context.scene.render.resolution_x = int(base_resolution * aspect_ratio_plane)
                bpy.context.scene.render.resolution_y = base_resolution
            bpy.context.scene.render.resolution_percentage = 100


        ortho_scale = max(plane.dimensions.x, plane.dimensions.y)
        cam.data.type = 'ORTHO'
        cam.data.ortho_scale = ortho_scale

        self.add_light(bpy)

        # Update the scene to reflect changes
        bpy.context.view_layer.update()

        # Set render engine (e.g., 'BLENDER_EEVEE', 'CYCLES', 'BLENDER_WORKBENCH')
        bpy.context.scene.render.engine = "BLENDER_EEVEE"

        # Specify the render output path
        
        output_path = get_incremented_filename(folder_paths.get_output_directory(), "render")
        bpy.context.scene.render.filepath = output_path

        # Render the image
        bpy.ops.render.render(write_still=True)

        # Load the image
        i = Image.open(output_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        # print(image.shape)

        return (BPY_OBJ, image)
