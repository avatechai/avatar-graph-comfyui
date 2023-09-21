import blender_node


class GetImageWidthHeight(blender_node.ObjectOps):
    BASE_INPUT_TYPES = {
        "image": ('IMAGE',),
        "scale": ('FLOAT', {'default': 1.0,})
    }

    RETURN_TYPES = ('FLOAT', 'FLOAT',)
    RETURN_NAMES = ('width', 'height',)

    def blender_process(self, bpy, image, scale, **props):

        height, width  = image[0].shape[:2]

        # print(height, width)
        # print((width * scale, height * scale, ))

        return (width * scale, height * scale, )
