class MeshFromTexture:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                # "min_threshold": ("INT", {"default": 50, "min": 0, "max": 1024}),
                # "max_threshold": ("INT", {"default": 150, "min": 0, "max": 1024}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), # For disabling cache
            },
        }

    RETURN_TYPES = ("IMAGE","BPY_OBJS")
    RETURN_NAMES = ("image","bpy_objs")

    FUNCTION = "process"

    #OUTPUT_NODE = False

    CATEGORY = "mesh"

    def process(self, image, seed):
        import torch
        import cv2
        import numpy as np
        import global_bpy
        bpy = global_bpy.get_bpy()

        image = np.copy(image[0].numpy())

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = (gray * 255).astype(np.uint8)
        # edges = cv2.Canny(gray, min_threshold, max_threshold)

        # Find contours
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        def normalize_vertices(vertices, max_value):
            return vertices / float(max_value) * 2 - 1

        # Get the image width and height
        height, width = image.shape[:2]

        # Normalize the vertices
        normalized_contours = []
        for contour in contours:
            normalized_contour = []
            for vertex in contour:
                normalized_vertex = [normalize_vertices(vertex[0][0], width), normalize_vertices(vertex[0][1], height) * -1]
                normalized_contour.append(normalized_vertex)
            normalized_contours.append(np.array(normalized_contour, dtype=np.float32))

        meshes = []
        # print(len(normalized_contours))
        for i, contour in enumerate(normalized_contours):
            mesh = bpy.data.meshes.new(name=f"NewMesh{i}")  # Create a new mesh for each contour
            obj = bpy.data.objects.new(f"NewObject{i}", mesh)  # Create a new object for each mesh
            bpy.context.collection.objects.link(obj)  # Link the object to the current collection

            ordered_vertices = [(*vertex, 0) for vertex in contour]  # Add a z coordinate to each vertex
            face = list(range(len(ordered_vertices)))  # Create a face from the vertices

            mesh.from_pydata(ordered_vertices, [], [face])  # Create the mesh from the vertices and face



            # Create a default shape key for the mesh
            sk_basis = obj.shape_key_add(name='Basis')

            meshes.append(obj)  # Add the object to the list of meshes

        # Convert edges back to an image
        # image_from_edges = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        # image_from_edges = image_from_edges.astype(np.float32)
        # image_from_edges = [torch.from_numpy(image_from_edges)]

        # Draw contours on the original image
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

        # print(image, meshes)

        # Convert image back to tensor
        image = [torch.from_numpy(image)]
        return (image,meshes)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "MeshFromTexture": MeshFromTexture
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshFromTexture": "Mesh from texture"
}
