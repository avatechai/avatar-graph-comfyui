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
    image = [torch.from_numpy(image)]
    return (image, meshes[0])
