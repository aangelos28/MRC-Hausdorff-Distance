def morton3D(k, x, y, z):
    """
    Computes and returns the morton code of the x, y, z coordinates each
    represented with k bits

    Args:
        k: Bits to represent each coordinate with
        x: X coordinate
        y: Y coordinate
        z: Z coordinate

    Returns:
        The morton code in integer format of the x, y, z coordinates of size 3*k
    """

    result = (x << 2*k) + (y << k) + z

    return result

def morton2D(k, x, y):
    """
    Computes and returns the morton code of the x, y coordinates each
    represented with k bits

    Args:
        k: Bits to represent each coordinate with
        x: X coordinate
        y: Y coordinate

    Returns:
        The morton code in integer format of the x, y coordinates of size 2*k
    """

    result = (x << k) + y 

    return result

def morton_to_string(dim, k, morton_code):
    """
    Creates a string representation of the morton code

    Args:
        dim: Dimension represented by morton code
        k: Bits to represent each coordinate with
        morton_code: The morton code in integer format

    Returns:
        String representation of the binary morton code
    """

    return "{0:0{digits}b}".format(morton_code, digits=dim*k)

def extract_morton_coords_bin(dim, k, morton_code):
    """
    Creates a list containing the extracted coordinates in binary format 
    from the morton code in the order in which they were encoded

    Args:
        dim: Dimension represented by morton code
        k: Bits to represent each coordinate with
        morton_code: The morton code in integer format

    Returns:
        List containing the extracted coordinates in binary format
    """

    morton_code_str = morton_to_string(dim, k, morton_code)

    # 3D
    if dim == 3:
        return [morton_code_str[0:k], morton_code_str[k:2*k], morton_code_str[2*k:3*k]]

    # 2D
    elif dim == 2:
        return [morton_code_str[0:k], morton_code_str[k:2*k]]
    else:
        return []

def extract_morton_coords_int(dim, k, morton_code):
    """
    Creates a list containing the extracted coordinates in integer format 
    from the morton code in the order in which they were encoded

    Args:
        dim: Dimension represented by morton code
        k: Bits to represent each coordinate with
        morton_code: The morton code in integer format

    Returns:
        List containing the extracted coordinates in integer format
    """

    # 3D
    if dim == 3:
        x = morton_code >> 2*k
        y = (morton_code >> k) & ((1 << (2*k - k)) - 1)
        z = morton_code & ((1 << k) - 1)

        return [x, y, z]

    # 2D
    elif dim == 2:
        x = morton_code >> k
        y = morton_code & ((1 << k) - 1)
        
        return [x, y]    
    else:
        return []