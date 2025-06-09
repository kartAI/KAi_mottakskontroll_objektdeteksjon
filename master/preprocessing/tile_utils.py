def should_skip_tile(clipped_gdf, annotations, include_empty_tiles):
    if include_empty_tiles:
        return False
    return clipped_gdf.empty or not annotations
