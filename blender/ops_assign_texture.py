import blender_node


class Object_AssignTexture(blender_node.ObjectOps):
    EXTRA_INPUT_TYPES = {
        "texture": ("IMAGE",),
        "texture_name": ("STRING", {
            "multiline": False,
            "default": "my_image"
        }),
    }

    def blender_process(self, bpy, BPY_OBJ, texture, texture_name):
        import numpy as np

        # Convert image to numpy
        texture = texture[0].numpy()

        # Flip the image vertically
        texture = np.flipud(texture)

        # Create an image with the required dimensions
        img = bpy.data.images.new(
            texture_name, width=texture.shape[1], height=texture.shape[0])

        # If there is no alpha channel, append one full of 1's
        if texture.shape[2] == 3:
            alpha_channel = np.ones((*texture.shape[:2], 1))
            texture = np.concatenate((texture, alpha_channel), axis=2)

        # Flatten image data and rearrange color channels for blender
        img.pixels = texture.ravel()

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

        # Assign the material to the active object
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)